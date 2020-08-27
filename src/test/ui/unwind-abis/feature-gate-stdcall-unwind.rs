// ignore-arm stdcall isn't supported
// ignore-aarch64 stdcall isn't supported
// ignore-riscv64 stdcall isn't supported

// Test that the "stdcall-unwind" ABI is feature-gated, and cannot be used when
// the `c_unwind` feature gate is not used.

extern "stdcall-unwind" fn f() {}
//~^ ERROR stdcall-unwind ABI is experimental and subject to change [E0658]

fn main() {
    f();
}
