// compile-flags: -C opt-level=0

#![crate_type = "lib"]
#![feature(unwind_attributes)]

// make sure these all do *not* get the attribute
// CHECK-NOT: nounwind

pub extern fn foo() {} // right now we don't abort-on-panic, so we also shouldn't have `nounwind`
#[unwind(allowed)]
pub extern fn foo_allowed() {}

pub extern "Rust" fn bar() {}
#[unwind(allowed)]
pub extern "Rust" fn bar_allowed() {}
