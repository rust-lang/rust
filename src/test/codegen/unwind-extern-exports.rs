// compile-flags: -C opt-level=0
// ignore-wasm32-bare compiled with panic=abort by default

#![crate_type = "lib"]
#![feature(c_unwind)]

// Make sure these all do *not* get the attribute.
// We disable optimizations to prevent LLVM from inferring the attribute.
// CHECK-NOT: nounwind

// "C" ABI
pub extern "C-unwind" fn foo_unwind() {}

// "Rust"
// (`extern "Rust"` could be removed as all `fn` get it implicitly; we leave it in for clarity.)
pub fn bar() {}
