// unit-test: CopyProp

#![feature(custom_mir, core_intrinsics)]
#![allow(unused_assignments)]
extern crate core;
use core::intrinsics::mir::*;

struct NotCopy(bool);

// EMIT_MIR custom_move_arg.f.CopyProp.diff
#[custom_mir(dialect = "analysis", phase = "post-cleanup")]
fn f(_1: NotCopy) {
    mir!({
        let _2 = _1;
        Call(RET, bb1, opaque(Move(_1)))
    }
    bb1 = {
        let _3 = Move(_2);
        Call(RET, bb2, opaque(_3))
    }
    bb2 = {
        Return()
    })
}

#[inline(never)]
fn opaque<T>(_t: T) {}

fn main() {
    f(NotCopy(true));
    println!("hi");
}
