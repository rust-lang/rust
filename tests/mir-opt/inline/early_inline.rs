//@ compile-flags: -Copt-level=0 -Zmir-opt-level=1 -Cdebuginfo=limited
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(rustc_attrs)]

#[rustc_early_inline]
fn do_early(x: i32, y: i32) -> i32 {
    x + y
}

// EMIT_MIR early_inline.early_as_fn.ForceInline.diff
fn early_as_fn() -> fn(i32, i32) -> i32 {
    // CHECK-LABEL: fn early_as_fn() -> fn(i32, i32) -> i32
    // CHECK: _0 = do_early as fn(i32, i32) -> i32 (PointerCoercion(ReifyFnPointer, Implicit));
    do_early
}

// EMIT_MIR early_inline.call_early.ForceInline.diff
fn call_early(x: i32) -> i32 {
    // CHECK-LABEL: fn call_early(_1: i32) -> i32
    // CHECK: (inlined do_early)
    // CHECK: _2 = const 42_i32;
    // CHECK: _3 = AddWithOverflow(copy _1, copy _2);
    // CHECK: _0 = move (_3.0: i32);
    do_early(x, 42)
}

fn main() {
    let f = early_as_fn();
    let _z = f(1, 2);
    call_early(7);
}
