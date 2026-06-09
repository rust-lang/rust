//@ compile-flags:-Clink-dead-code

#![crate_type = "rlib"]

// This test makes sure that, when -Clink-dead-code is specified, we generate
// code for functions that would otherwise be skipped.

// CHECK-LABEL: ; link_dead_code::const_fn
// CHECK-NEXT: ; Function Attrs:
// CHECK-NEXT: define hidden
const fn const_fn() -> i32 {
    1
}

// CHECK-LABEL: ; link_dead_code::inline_fn
// CHECK-NEXT: ; Function Attrs:
// CHECK-NEXT: define hidden
#[inline]
fn inline_fn() -> i32 {
    2
}

// CHECK-LABEL: ; link_dead_code::private_fn
// CHECK-NEXT: ; Function Attrs:
// CHECK-NEXT: define hidden
fn private_fn() -> i32 {
    3
}
