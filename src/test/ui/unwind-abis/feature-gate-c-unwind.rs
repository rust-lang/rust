// Test that the "C-unwind" ABI is feature-gated, and cannot be used when the
// `c_unwind` feature gate is not used.

#![allow(ffi_unwind_calls)]
//~^ WARNING unknown lint: `ffi_unwind_calls`
//~| WARNING unknown lint: `ffi_unwind_calls`

extern "C-unwind" fn f() {}
//~^ ERROR C-unwind ABI is experimental and subject to change [E0658]

fn main() {
    f();
}
