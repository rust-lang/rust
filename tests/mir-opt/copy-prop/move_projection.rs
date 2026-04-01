// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: CopyProp

#![feature(custom_mir, core_intrinsics)]
#![allow(unused_assignments)]
extern crate core;
use core::intrinsics::mir::*;

fn opaque(_: impl Sized) -> bool {
    true
}

struct Foo(u8);

#[custom_mir(dialect = "runtime")]
fn f(a: Foo) -> bool {
    // CHECK-LABEL: fn f(
    // CHECK-SAME: [[a:_.*]]: Foo)
    // CHECK: bb0: {
    // CHECK-NOT: _2 = copy [[a]];
    // CHECK-NOT: _3 = move (_2.0: u8);
    // CHECK: [[c:_.*]] = copy ([[a]].0: u8);
    // CHECK: _0 = opaque::<Foo>(copy [[a]])
    // CHECK: bb1: {
    // CHECK: _0 = opaque::<u8>(move [[c]])
    mir! {
        {
            let b = a;
            // This is a move out of a copy, so must become a copy of `a.0`.
            let c = Move(b.0);
            Call(RET = opaque(Move(a)), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Call(RET = opaque(Move(c)), ReturnTo(ret), UnwindUnreachable())
        }
        ret = {
            Return()
        }
    }
}

fn main() {
    assert!(f(Foo(0)));
}

// EMIT_MIR move_projection.f.CopyProp.diff
