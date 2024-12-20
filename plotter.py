from matplotlib import pyplot as plt
import json

def plotAblation(jsonFilenames, title="Train"):
    data = {}

    for jsonFilename in jsonFilenames:
        with open("{}.json".format(jsonFilename), 'r') as inFile:
            data[jsonFilename] = json.load(inFile)

    for (key, runData) in data.items():
        epochData = runData["epoch times"]
        X = [i+1 for i in range(len(epochData))]
        plt.plot(X, epochData, label=key)

    controlVal = sum(data[jsonFilenames[0]]["epoch times"][-10:])
    compileVal = sum(data[jsonFilenames[1]]["epoch times"][-10:])
    print("Epoch Time Improvement: ", 100 * (controlVal - compileVal) / compileVal, "%")

    plt.yscale('log')

    plt.title("Optimization Strategies Ablation: Epoch Times")
    plt.xlabel("Epoch")
    plt.ylabel("Epoch Time (s)")

    plt.legend()
    plt.savefig("{}Epoch.png".format(title))

    plt.clf()
    for (key, runData) in data.items():
        batchData = runData["batch times"]
        batchData = [sum(allEpochBatches) for allEpochBatches in batchData]
        X = [i+1 for i in range(len(batchData))]
        plt.plot(X, batchData, label=key)

    controlVal = sum([sum(epochBatches) for epochBatches in data[jsonFilenames[0]]["batch times"]][-10:])
    compileVal = sum([sum(epochBatches) for epochBatches in data[jsonFilenames[1]]["batch times"]][-10:])
    print("Batch Processing Time Improvement: ", 100 * (controlVal - compileVal) / compileVal, "%")

    plt.yscale('log')

    plt.title("Optimization Strategies Ablation: Batch Times")
    plt.xlabel("Epoch")
    plt.ylabel("Total Batch Time per Epoch (s)")

    plt.legend()
    plt.savefig("{}Batch.png".format(title))

def plotTestAblation():
    print("Plotting Test Ablation")
    jsonFilenames = ["test_control", "test_compile", "test_quant", "test_compile+quant"]
    plotAblation(jsonFilenames, title="Test")

def plotTrainAblation():
    print("Plotting Train Ablation")
    jsonFilenames = ["train_control", "train_compile", "train_amp", "train_compile+amp"]
    plotAblation(jsonFilenames, title="Train")
    
def plotAccuracies(accs):
    items = accs.items()
    keys = [item[0] for item in items]
    values = [item[1] for item in items]
    plt.bar(keys, values)

    plt.title("Test Loss")
    plt.xlabel("Optimization Type")
    plt.ylabel("Test Loss")

    plt.savefig("TestLoss.png")

test_losses = {"control": -78.16336919762789, "compile": -75.64637800851249, "quantized": -6.353716566571718, "compile+quantized": -7.511720192686234}
plotTrainAblation()
plotTestAblation()
plotAccuracies(test_losses)
