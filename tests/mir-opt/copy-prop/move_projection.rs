// skip-filecheck
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
