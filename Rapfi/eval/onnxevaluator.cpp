/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022  Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "onnxevaluator.h"

#include "../core/utils.h"
#include "../game/board.h"
#include "weightloader.h"

#include <onnxruntime_cxx_api.h>

namespace Evaluation::onnx {

enum OnnxDevice {
    DEFAULT_DEVICE = 0,
    CPU            = 1,
    CUDA_0         = 2,
    CUDA_1         = 3,
    CUDA_2         = 4,
    CUDA_3         = 5,
    CUDA_MAX       = 9,
};

class OnnxModel
{
public:
    enum ModelIOVersion { VERSION_START = 1, VERSION_1 = 1, VERSION_MAX };

    OnnxModel(std::istream &modelStream, OnnxDevice device)
    {
        std::string modelData = readAllFromStream(modelStream);

        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetInterOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        initExecutionProvider(device);

        session = Ort::Session(getGlobalEnvInstance(),
                               modelData.data(),
                               modelData.size(),
                               sessionOptions);

        auto    metainfo     = session.GetModelMetadata();
        int64_t majorVersion = (metainfo.GetVersion() >> 48) & 0xffff;
        if (majorVersion >= VERSION_START && majorVersion < VERSION_MAX)
            modelVersion = static_cast<ModelIOVersion>(majorVersion);
        else
            throw std::runtime_error("unsupported onnx model");
    }

    ModelIOVersion getVersion() const { return modelVersion; }

    std::vector<std::string> getInputNames() const
    {
        std::vector<std::string>         inputNames;
        Ort::AllocatorWithDefaultOptions allocator;
        size_t                           numInputs = session.GetInputCount();
        for (size_t i = 0; i < numInputs; i++) {
            auto inputName = session.GetInputNameAllocated(i, allocator);
            inputNames.emplace_back(inputName.get());
        }
        return inputNames;
    }

    std::vector<std::string> getOutputNames() const
    {
        std::vector<std::string>         outputNames;
        Ort::AllocatorWithDefaultOptions allocator;
        size_t                           numOutputs = session.GetOutputCount();
        for (size_t i = 0; i < numOutputs; i++) {
            auto outputName = session.GetOutputNameAllocated(i, allocator);
            outputNames.emplace_back(outputName.get());
        }
        return outputNames;
    }

    void run(const std::vector<const char *> &inputNames,
             const std::vector<Ort::Value>   &inputValues,
             const std::vector<const char *> &outputNames,
             std::vector<Ort::Value>         &outputValues)
    {
        Ort::RunOptions runOptions;
        session.Run(runOptions,
                    inputNames.data(),
                    inputValues.data(),
                    inputNames.size(),
                    outputNames.data(),
                    outputValues.data(),
                    outputNames.size());
    }

private:
    std::string readAllFromStream(std::istream &in)
    {
        std::string ret;
        char        buffer[4096];
        while (in.read(buffer, sizeof(buffer)))
            ret.append(buffer, sizeof(buffer));
        ret.append(buffer, in.gcount());

        return ret;
    }

    void initExecutionProvider(OnnxDevice device)
    {
        if (device == CPU)
            return;  // Do nothing
        else if (CUDA_0 <= device && device <= CUDA_MAX) {
#ifdef USE_CUDA
            OrtCUDAProviderOptions cudaOptions;
            cudaOptions.device_id                 = device - CUDA_0;
            cudaOptions.arena_extend_strategy     = 0;
            cudaOptions.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchExhaustive;
            cudaOptions.do_copy_in_default_stream = 1;
            sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
#else
            throw std::runtime_error("cuda device is not supported in this build");
#endif
        }
        else {
            throw std::runtime_error("unsupported device " + std::to_string(device));
        }
    }

    static Ort::Env &getGlobalEnvInstance()
    {
        static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "rapfi");
        return env;
    }

    Ort::Session        session {nullptr};
    Ort::SessionOptions sessionOptions;
    ModelIOVersion      modelVersion;
};

class OnnxAccumulatorV1 : public OnnxAccumulator
{
public:
    OnnxAccumulatorV1(int boardSize, Color side)
        : stmInput {side == BLACK ? -1.0f : 1.0f}
        , sizeInput {(int8_t)boardSize, (int8_t)boardSize}
        , sideToMove {side}
    {
        boardInput.resize(BatchSize * 2 * boardSize * boardSize);
        policyOutput.resize(BatchSize * boardSize * boardSize);
    }

    void init(OnnxModel &model) override
    {
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        inputTensors.clear();
        inputNames.clear();
        for (const auto &name : model.getInputNames()) {
            if (name == InputNames[0]) {
                const std::vector<int64_t> boardInputShape {BatchSize,
                                                            2,
                                                            (int64_t)sizeInput[0],
                                                            (int64_t)sizeInput[1]};

                inputTensors.push_back(Ort::Value::CreateTensor<int8_t>(memoryInfo,
                                                                        boardInput.data(),
                                                                        boardInput.size(),
                                                                        boardInputShape.data(),
                                                                        boardInputShape.size()));
                inputNames.push_back(InputNames[0]);
            }
            else if (name == InputNames[1]) {
                const std::vector<int64_t> stmInputShape {BatchSize, 1};
                inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
                                                                       &stmInput,
                                                                       1,
                                                                       stmInputShape.data(),
                                                                       stmInputShape.size()));
                inputNames.push_back(InputNames[1]);
            }
            else if (name == InputNames[2]) {
                const std::vector<int64_t> boardSizeShape {BatchSize, 2};
                inputTensors.push_back(Ort::Value::CreateTensor<int8_t>(memoryInfo,
                                                                        sizeInput,
                                                                        2,
                                                                        boardSizeShape.data(),
                                                                        boardSizeShape.size()));
                inputNames.push_back(InputNames[2]);
            }
            else {
                throw std::runtime_error("unknown input name in onnx model (v1): " + name);
            }
        }

        outputTensors.clear();
        outputNames.clear();
        for (const auto &name : model.getOutputNames()) {
            if (name == OutputNames[0]) {
                const std::vector<int64_t> valueShape {BatchSize, 3};
                outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
                                                                        valueOutput,
                                                                        3,
                                                                        valueShape.data(),
                                                                        valueShape.size()));
                outputNames.push_back(OutputNames[0]);
            }
            else if (name == OutputNames[1]) {
                const std::vector<int64_t> policyShape {BatchSize,
                                                        (int64_t)sizeInput[0],
                                                        (int64_t)sizeInput[1]};
                outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
                                                                        policyOutput.data(),
                                                                        policyOutput.size(),
                                                                        policyShape.data(),
                                                                        policyShape.size()));
                outputNames.push_back(OutputNames[1]);
            }
            else {
                throw std::runtime_error("unknown output name in onnx model (v1): " + name);
            }
        }
    }

    void clear(OnnxModel &model) override
    {
        std::fill(boardInput.begin(), boardInput.end(), 0);
        std::fill(policyOutput.begin(), policyOutput.end(), 0.0f);
        dirty = true;
    }

    void update(OnnxModel &model, Color oldColor, Color newColor, int x, int y) override
    {
        const size_t channelStride = (size_t)sizeInput[0] * (size_t)sizeInput[1];
        const size_t posOffset     = y * (size_t)sizeInput[1] + x;

        boardInput[0 * channelStride + posOffset] = newColor == sideToMove;   // self plane
        boardInput[1 * channelStride + posOffset] = newColor == ~sideToMove;  // oppo plane
    }

    ValueType evaluateValue(OnnxModel &model) override
    {
        if (dirty)
            runInference(model);

        return ValueType(valueOutput[0], valueOutput[1], valueOutput[2], true);
    }

    void evaluatePolicy(OnnxModel &model, PolicyBuffer &policyBuffer) override
    {
        if (dirty)
            runInference(model);

        for (int y = 0; y < sizeInput[0]; y++)
            for (int x = 0; x < sizeInput[1]; x++) {
                if (policyBuffer.getComputeFlag(x, y))
                    policyBuffer(x, y) = policyOutput[y * sizeInput[1] + x];
            }
    }

private:
    static constexpr int64_t     BatchSize     = 1;
    static constexpr const char *InputNames[]  = {"board_input", "stm_input", "board_size"};
    static constexpr const char *OutputNames[] = {"value", "policy"};

    void runInference(OnnxModel &model)
    {
        model.run(inputNames, inputTensors, outputNames, outputTensors);
    }

    std::vector<int8_t> boardInput;
    float               stmInput;
    int8_t              sizeInput[2];
    Color               sideToMove;
    bool                dirty;
    float               valueOutput[3];
    std::vector<float>  policyOutput;

    std::vector<Ort::Value>   inputTensors;
    std::vector<const char *> inputNames;
    std::vector<Ort::Value>   outputTensors;
    std::vector<const char *> outputNames;
};

}  // namespace Evaluation::onnx

namespace {

using namespace Evaluation;
using namespace Evaluation::onnx;

OnnxDevice parseDeviceString(std::string device)
{
    if (device.empty())
        return DEFAULT_DEVICE;

    upperInplace(device);
    if (device == "CPU")
        return CPU;
    else if (device == "CUDA")
        return CUDA_0;
    else if (device.rfind("CUDA:", 0) == 0) {
        int cudaDeviceId = std::stoi(device.substr(5));
        if (CUDA_0 + cudaDeviceId <= CUDA_MAX)
            return OnnxDevice(CUDA_0 + cudaDeviceId);
        else
            throw std::runtime_error("invalid cuda device " + device);
    }
    else
        throw std::runtime_error("unknown device " + device);
}

std::string deviceString(OnnxDevice device)
{
    if (device == CPU)
        return "cpu";
    else if (CUDA_0 <= device && device <= CUDA_MAX)
        return "cuda:" + std::to_string(device - CUDA_0);
    else
        return "unknown";
}

OnnxDevice pickDefaultDevice()
{
    return CPU;
}

static WeightRegistry<OnnxModel, OnnxDevice> OnnxModelRegistry;

struct OnnxModelLoader : WeightLoader<OnnxModel, OnnxDevice>
{
    std::unique_ptr<OnnxModel> load(std::istream &is, OnnxDevice args) override
    {
        try {
            return std::make_unique<OnnxModel>(is, args);
        }
        catch (const std::exception &e) {
            ERRORL("Failed to create onnx model: " << e.what());
            return nullptr;
        }
    }
};

std::unique_ptr<OnnxAccumulator>
makeOnnxAccumulator(OnnxModel &model, int boardSize, Rule rule, Color side)
{
    std::unique_ptr<OnnxAccumulator> ptr;
    switch (model.getVersion()) {
    case OnnxModel::VERSION_1: ptr = std::make_unique<OnnxAccumulatorV1>(boardSize, side); break;
    default: throw std::runtime_error("unknown onnx model version"); break;
    }

    ptr->init(model);
    return ptr;
}

}  // namespace

namespace Evaluation::onnx {

OnnxEvaluator::OnnxEvaluator(int                   boardSize,
                             Rule                  rule,
                             std::filesystem::path onnxModelPath,
                             std::string           device)
    : Evaluator(boardSize, rule)
{
    if (!std::filesystem::exists(onnxModelPath))
        throw std::runtime_error("Onnx model file not found: " + pathToString(onnxModelPath));

    OnnxDevice dev = parseDeviceString(device);
    if (dev == DEFAULT_DEVICE)
        dev = pickDefaultDevice();

    OnnxModelLoader loader;
    model = OnnxModelRegistry.loadWeightFromFile(loader, onnxModelPath, dev);
    if (!model)
        throw std::runtime_error("Failed to load onnx model from " + pathToString(onnxModelPath));

    accumulator[BLACK] = makeOnnxAccumulator(*model, boardSize, rule, BLACK);
    accumulator[WHITE] = makeOnnxAccumulator(*model, boardSize, rule, WHITE);

    MESSAGEL("Initialized onnx model on device: " << deviceString(dev));
}

OnnxEvaluator::~OnnxEvaluator()
{
    OnnxModelRegistry.unloadWeight(model);
}

void OnnxEvaluator::initEmptyBoard()
{
    accumulator[BLACK]->clear(*model);
    accumulator[WHITE]->clear(*model);
}

void OnnxEvaluator::beforeMove(const Board &board, Pos pos)
{
    for (Color c : {BLACK, WHITE}) {
        accumulator[c]->update(*model, EMPTY, board.sideToMove(), pos.x(), pos.y());
    }
}

void OnnxEvaluator::afterUndo(const Board &board, Pos pos)
{
    for (Color c : {BLACK, WHITE}) {
        accumulator[c]->update(*model, board.sideToMove(), EMPTY, pos.x(), pos.y());
    }
}

ValueType OnnxEvaluator::evaluateValue(const Board &board)
{
    Color self = board.sideToMove();
    return accumulator[self]->evaluateValue(*model);
}

void OnnxEvaluator::evaluatePolicy(const Board &board, PolicyBuffer &policyBuffer)
{
    Color self = board.sideToMove();
    accumulator[self]->evaluatePolicy(*model, policyBuffer);
}

}  // namespace Evaluation::onnx
