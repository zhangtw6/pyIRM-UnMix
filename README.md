
# pyIRM-UnMix

**pyIRM-UnMix** is a package designed for unmixing isothermal remanent magnetization acquisition curves (IRM acquisition curves) in rock magnetism and environmental magnetism.

It provides a practical workflow for decomposing IRM acquisition curves into magnetic components, helping users identify coercivity distributions and interpret magnetic mineral assemblages in geological and environmental samples.

## Main Features

- **Single-sample IRM unmixing**  
  `Run_single_sample` provides a streamlined workflow for fitting and unmixing IRM acquisition curves from individual samples.

- **Batch processing for large datasets**  
  `Run_multiple_samples` is designed for efficient processing of large environmental magnetism datasets.

- **Full parameters optimization**  
  The package using full parameters of SGG model to perform IRM unmixing.

## Data Format

Example files are provided in the repository.

Supported input formats include:

- AGM2900 / AGM3900 files
- VSM8600 `.csv` files
- VSM8600 `.vers` files
- General text files with two columns:
  - Field
  - Remanence

For the general text format, the file should contain two columns separated by commas, for example:

```txt
Field,Remanence
0,0.0012
10,0.0045
20,0.0081
...

```

## Installation

No installation is required.

Simply download the package and unzip it to any directory on your computer. Then run the corresponding program or script according to the prompts.

Recommended steps:
1、Download or clone this repository.
2、Unzip the package to any local folder.
3、Open the folder.
4、Run the desired workflow:
- Run_single_sample for single-sample IRM unmixing
- Run_multiple_samples for batch processing of multiple samples
5、Follow the prompts to load data and complete the unmixing process.

## Example Files

Please refer to the example files included in the repository for supported data structures and formatting requirements.

## License

This project is licensed under the MIT License.
