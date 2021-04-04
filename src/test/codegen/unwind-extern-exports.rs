// compile-flags: -C opt-level=0
// ignore-wasm32-bare compiled with panic=abort by default

#![crate_type = "lib"]
#![feature(unwind_attributes)]

// Make sure these all do *not* get the attribute.
// We disable optimizations to prevent LLVM from infering the attribute.
// CHECK-NOT: nounwind

// "C" ABI
// pub extern fn foo() {} // FIXME right now we don't abort-on-panic but add `nounwind` nevertheless
#[unwind(allowed)]
pub extern "C" fn foo_allowed() {}

// "Rust"
// (`extern "Rust"` could be removed as all `fn` get it implicitly; we leave it in for clarity.)
pub extern "Rust" fn bar() {}
#[unwind(allowed)]
pub extern "Rust" fn bar_allowed() {}
