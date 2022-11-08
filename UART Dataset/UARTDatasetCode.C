#include "msp.h"
#include "stdio.h"

void transmit(char x);
void sender();
char voltage;                               //use char to putty
int main(void)
{
volatile unsigned int i;
WDT_A->CTL = WDT_A_CTL_PW | WDT_A_CTL_HOLD; // Stop WDT

// GPIO Setup
P1->DIR &= ~BIT4;                           // P1.4 Button
//Configuring pins 1.2 and 1.3 to use for UART
P1->SEL0 |= BIT2 | BIT3;

// Enable global interrupt
__enable_irq();

// Enable ADC interrupt in NVIC module
// NVIC->ISER[0] = 1 << ((ADC14_IRQn) & 31);

// Initialize UART
EUSCI_A0->CTLW0 |= EUSCI_A_CTLW0_SWRST;     // Reset eUSCI
EUSCI_A0->CTLW0 = EUSCI_A_CTLW0_SWRST |     // Keep eUSCI in reset
EUSCI_B_CTLW0_SSEL__SMCLK;                  // Use SMCLK as the eUSCI clock source
EUSCI_A0->BRW = 19;                         //Baud rate register adjusted for nominal baud rate,3Meg/9600/16=19
EUSCI_A0->CTLW0 |=0;
EUSCI_A0->MCTLW = 0xB601;                   //adjusting for 3MHz
EUSCI_A0->CTLW0 &= ~EUSCI_A_CTLW0_SWRST;    // Initialize eUSCI
EUSCI_A0->IFG &= ~EUSCI_A_IFG_RXIFG;        // Clear eUSCI RX interrupt flag
EUSCI_A0->IE = 0;                           //Disable Interrupts will use polling instead

__DSB();

while (1)
{
     if ((P1->IN & BIT4) == 1){             //Press P1.4 to transmit character
         sender();
     }
}
}

void sender()
{
voltage = 'Q';                              // Character to transmit
transmit(voltage);                          //Output to putty
}



void transmit(char x)                       //function to transmit bit per bit
{
while(!(EUSCI_A0->IFG & EUSCI_A_IFG_TXIFG));
//printf("N");                                // Print to console
EUSCI_A0->TXBUF = x;                        // Transmits
}
