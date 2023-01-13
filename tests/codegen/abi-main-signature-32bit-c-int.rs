// Checks the signature of the implicitly generated native main()
// entry point. It must match C's `int main(int, char **)`.

// This test is for targets with 32bit c_int only.
// ignore-msp430

fn main() {
}

// CHECK: define{{( hidden)?}} i32 @main(i32{{( %0)?}}, {{i8\*\*|ptr}}{{( %1)?}})
