// Checks the signature of the implicitly generated native main()
// entry point. It must match C's `int main(int, char **)`.

// This test is for targets with 32bit c_int only.
//@ ignore-msp430
//@ ignore-avr
//@ ignore-wasi wasi codegens the main symbol differently

fn main() {}

// CHECK: define{{( hidden| noundef)*}} i32 @main(i32{{( %0)?}}, ptr{{( %1)?}})
