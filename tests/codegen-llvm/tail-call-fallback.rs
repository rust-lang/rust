//@ add-minicore
//@ compile-flags: -Zforce-tail-call-fallback -Copt-level=3
#![crate_type = "lib"]
#![feature(explicit_tail_calls)]

// Check the guaranteed tail call desugaring. The shims return continuations, and don't make any
// calls themselves. The transformed functions (that retain the symbol name) use an indirect
// call as part of the trampoline structure.

// CHECK-LABEL: is_odd::{shim:tail_call#0}
// CHECK-NOT: {{call[[:space:]]}}

// CHECK-LABEL: is_even::{shim:tail_call#0}
// CHECK-NOT: {{call[[:space:]]}}

// CHECK-LABEL: define{{.*}} i1 @is_even
#[unsafe(no_mangle)]
fn is_even(n: u32) -> bool {
    // CHECK-NOT: musttail
    // The indirect call made by (the inlined) tail_eval.
    // CHECK: call void %4
    match n {
        0 => true,
        _ => become is_odd(n - 1),
    }
}

// CHECK-LABEL: define{{.*}} i1 @is_odd
#[unsafe(no_mangle)]
fn is_odd(n: u32) -> bool {
    // CHECK-NOT: musttail
    // The indirect call made by (the inlined) tail_eval.
    // CHECK: call void %4
    match n {
        0 => false,
        _ => become is_even(n - 1),
    }
}
