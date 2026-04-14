//! Regression test for <https://github.com/rust-lang/rust/issues/155241>.
//!
//! When a closure with the `extern "rust-call"` ABI is invoked with a constant
//! tuple whose layout is passed indirectly, `codegen_arguments_untupled` used
//! to hand the callee a pointer straight into the caller's operand (typically
//! promoted into `.rodata`). Because the callee owns its argument storage by
//! ABI contract, a mutation inside the closure would write through a read-only
//! pointer and abort with SIGBUS / `STATUS_HEAP_CORRUPTION`.
//!
//! The safety-copy added in PR #45996 for `codegen_call_terminator` was never
//! ported to the tupled `rust-call` path; this test locks in that fix.
//
//@ run-pass
//@ revisions: opt0 opt3
//@[opt0] compile-flags: -Copt-level=0
//@[opt3] compile-flags: -Copt-level=3

#![feature(fn_traits)]

use std::hint::black_box;

#[derive(Copy, Clone)]
struct Thing {
    x: usize,
    y: usize,
    z: usize,
}

// A tuple large enough to be passed indirectly (BackendRepr::Memory).
const VALUE: (Thing,) = (Thing { x: 0, y: 0, z: 0 },);

fn main() {
    // The original 2017 reproducer: invoke the closure through `Fn::call`,
    // which forces the tupled `rust-call` code path in the caller.
    let observed = (|mut thing: Thing| {
        thing.z = 1;
        // Prevent the optimizer from eliminating the write entirely.
        black_box(&thing);
        thing.z
    })
    .call(VALUE);
    assert_eq!(observed, 1);

    // Same shape, but exercised through `FnMut::call_mut` to cover the other
    // trait method that lowers to the same tupled path.
    let mut sum = 0usize;
    (|t: Thing| {
        sum = sum.wrapping_add(t.x).wrapping_add(t.y).wrapping_add(t.z);
    })
    .call_mut(VALUE);
    assert_eq!(sum, 0);

    // And via `FnOnce::call_once` for good measure.
    let taken = (|mut t: Thing| {
        t.z = 42;
        black_box(&t);
        t.z
    })
    .call_once(VALUE);
    assert_eq!(taken, 42);
}
