// skip-filecheck
//@ test-mir-pass: ElaborateBoxDerefs

#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;

// EMIT_MIR elaborate_box_deref_in_debuginfo.pointee.ElaborateBoxDerefs.diff
#[custom_mir(dialect = "built")]
fn pointee(opt: Box<i32>) {
    mir!(
        debug foo => *opt;
        {
            Return()
        }
    )
}

fn main() {}
