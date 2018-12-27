// compile-flags:-Clink-dead-code

#![crate_type = "rlib"]

// This test makes sure that, when -Clink-dead-code is specified, we generate
// code for functions that would otherwise be skipped.

// CHECK-LABEL: define hidden i32 @_ZN14link_dead_code8const_fn
const fn const_fn() -> i32 { 1 }

// CHECK-LABEL: define hidden i32 @_ZN14link_dead_code9inline_fn
#[inline]
fn inline_fn() -> i32 { 2 }

// CHECK-LABEL: define hidden i32 @_ZN14link_dead_code10private_fn
fn private_fn() -> i32 { 3 }
