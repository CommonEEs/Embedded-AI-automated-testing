# Embedded-AI-automated-testing
This is the github repository for our UTEP senior capstone project  
###### Members
Nathan Lee: nslee@miners.utep.edu  
Noel Cruz: ncruz2@miners.utep.edu  
Herman Ramey: hframey@miners.utep.edu  
Chris Ramirez: coramirez5@miners.utep.edu  
Jorge Rodriguez: jarodriguez44@miners.utep.edu
## Project Goal
Our aim is to implement a neural network to detect anomalous waveforms on an embedded system (Xilinx FPGA platform)

### Dataset  1
#### RC circuit dataset
Oscilloscope data from a simple RC circuit built on a breadboard. Acceptable values set at ??Ohms and ??Farads. Dataset divided into three folders:  
- Training data (Good)
- Training data (Bad)
- Test data (Mixed)

Limited in applicability and difficulty. Obsolete.

### Dataset 2
#### Radar dataset
Simulated data from a virtual antenna to detect approaching  targets using matlab platform. Acceptable signals have positive velocity . Dataset divided into three folders:  
- Training data (False-alarm)
- Training data (Approaching)
- Test data (Mixed)

### Dataset 3
#### UART Dataset
Sent a letter Q and a generated sine wave into a summing amplifier to modify UART signals. Acceptable signals have slight noise on the square waves being sent. Dataset divided into three folders:
- Training Data (Good)
- Training Data (Bad)
- Test Data (Mixed)
