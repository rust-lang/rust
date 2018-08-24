// Checks the signature of the implicitly generated native main()
// entry point. It must match C's `int main(int, char **)`.

// This test is for targets with 16bit c_int only.
// ignore-aarch64
// ignore-arm
// ignore-asmjs
// ignore-hexagon
// ignore-mips
// ignore-mips64
// ignore-powerpc
// ignore-powerpc64
// ignore-s390x
// ignore-sparc
// ignore-sparc64
// ignore-wasm32
// ignore-x86
// ignore-x86_64
// ignore-xcore

fn main() {
}

// CHECK: define i16 @main(i16, i8**)
