// skip-filecheck
#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;

// EMIT_MIR assume.assume_local.built.after.mir
#[custom_mir(dialect = "built")]
fn assume_local(x: bool) {
    mir! {
        {
            Assume(x);
            Return()
        }
    }
}

// EMIT_MIR assume.assume_place.built.after.mir
#[custom_mir(dialect = "built")]
fn assume_place(p: (bool, u8)) {
    mir! {
        {
            Assume(p.0);
            Return()
        }
    }
}

// EMIT_MIR assume.assume_constant.built.after.mir
#[custom_mir(dialect = "built")]
fn assume_constant() {
    mir! {
        {
            Assume(true);
            Return()
        }
    }
}

fn main() {
    assume_local(true);
    assume_place((true, 50));
    assume_constant();
}
