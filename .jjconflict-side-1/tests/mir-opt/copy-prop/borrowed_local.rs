// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: CopyProp

#![feature(custom_mir, core_intrinsics, freeze)]
#![allow(unused_assignments)]
extern crate core;
use core::intrinsics::mir::*;
use core::marker::Freeze;

fn opaque(_: impl Sized) -> bool {
    true
}

fn cmp_ref(a: &u8, b: &u8) -> bool {
    std::ptr::eq(a as *const u8, b as *const u8)
}

#[custom_mir(dialect = "analysis", phase = "post-cleanup")]
fn compare_address() -> bool {
    // CHECK-LABEL: fn compare_address(
    // CHECK: bb0: {
    // CHECK-NEXT: _1 = const 5_u8;
    // CHECK-NEXT: _2 = &_1;
    // CHECK-NEXT: _3 = copy _1;
    // CHECK-NEXT: _4 = &_3;
    // CHECK-NEXT: _0 = cmp_ref(copy _2, copy _4)
    // CHECK: bb1: {
    // CHECK-NEXT: _0 = opaque::<u8>(copy _3)
    mir! {
        {
            let a = 5_u8;
            let r1 = &a;
            let b = a;
            // We cannot propagate the place `a`.
            let r2 = &b;
            Call(RET = cmp_ref(r1, r2), ReturnTo(next), UnwindContinue())
        }
        next = {
            // But we can propagate the value `a`.
            Call(RET = opaque(b), ReturnTo(ret), UnwindContinue())
        }
        ret = {
            Return()
        }
    }
}

/// Generic type `T` is `Freeze`, so shared borrows are immutable.
#[custom_mir(dialect = "analysis", phase = "post-cleanup")]
fn borrowed<T: Copy + Freeze>(x: T) -> bool {
    // CHECK-LABEL: fn borrowed(
    // CHECK: bb0: {
    // CHECK-NEXT: _3 = &_1;
    // CHECK-NEXT: _0 = opaque::<&T>(copy _3)
    // CHECK: bb1: {
    // CHECK-NEXT: _0 = opaque::<T>(copy _1)
    mir! {
        {
            let a = x;
            let r1 = &x;
            Call(RET = opaque(r1), ReturnTo(next), UnwindContinue())
        }
        next = {
            Call(RET = opaque(a), ReturnTo(ret), UnwindContinue())
        }
        ret = {
            Return()
        }
    }
}

/// Generic type `T` is not known to be `Freeze`, so shared borrows may be mutable.
#[custom_mir(dialect = "analysis", phase = "post-cleanup")]
fn non_freeze<T: Copy>(x: T) -> bool {
    // CHECK-LABEL: fn non_freeze(
    // CHECK: bb0: {
    // CHECK-NEXT: _2 = copy _1;
    // CHECK-NEXT: _3 = &_1;
    // CHECK-NEXT: _0 = opaque::<&T>(copy _3)
    // CHECK: bb1: {
    // CHECK-NEXT: _0 = opaque::<T>(copy _2)
    mir! {
        {
            let a = x;
            let r1 = &x;
            Call(RET = opaque(r1), ReturnTo(next), UnwindContinue())
        }
        next = {
            Call(RET = opaque(a), ReturnTo(ret), UnwindContinue())
        }
        ret = {
            Return()
        }
    }
}

/// We must not unify a borrowed local with another that may be written-to before the borrow is
/// read again. As we have no aliasing model yet, this means forbidding unifying borrowed locals.
fn borrow_in_loop() {
    // CHECK-LABEL: fn borrow_in_loop(
    // CHECK: debug c => [[c:_.*]];
    // CHECK: debug p => [[p:_.*]];
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[b:_.*]];
    // CHECK-NOT: &[[a]]
    // CHECK-NOT: &[[b]]
    // CHECK: [[a]] = Not({{.*}});
    // CHECK-NOT: &[[a]]
    // CHECK-NOT: &[[b]]
    // CHECK: [[b]] = Not({{.*}});
    // CHECK-NOT: &[[a]]
    // CHECK-NOT: &[[b]]
    // CHECK: &[[c]]
    // CHECK-NOT: &[[a]]
    // CHECK-NOT: &[[b]]
    let mut c;
    let mut p = &false;
    loop {
        let a = !*p;
        let b = !*p;
        c = a;
        p = &c;
        if a != b {
            return;
        }
    }
}

fn main() {
    assert!(!compare_address());
    non_freeze(5);
    borrow_in_loop();
}

// EMIT_MIR borrowed_local.compare_address.CopyProp.diff
// EMIT_MIR borrowed_local.borrowed.CopyProp.diff
// EMIT_MIR borrowed_local.non_freeze.CopyProp.diff
// EMIT_MIR borrowed_local.borrow_in_loop.CopyProp.diff
