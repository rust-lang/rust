// Checks the signature of the implicitly generated native main()
// entry point. It must match C's `int main(int, char **)`.

// This test is for targets with 16bit c_int only.
//@ revisions: avr msp
//@[avr] only-avr
//@[msp] only-msp430

fn main() {}

// CHECK: define i16 @main(i16, i8**)
