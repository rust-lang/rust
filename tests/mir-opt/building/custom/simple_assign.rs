// skip-filecheck
#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;

// EMIT_MIR simple_assign.simple.built.after.mir
#[custom_mir(dialect = "built")]
pub fn simple(x: i32) -> i32 {
    mir! {
        let temp1: i32;
        let temp2: _;
        {
            StorageLive(temp1);
            temp1 = x;
            Goto(exit)
        }
        exit = {
            temp2 = Move(temp1);
            StorageDead(temp1);
            RET = temp2;
            Return()
        }
    }
}

// EMIT_MIR simple_assign.simple_ref.built.after.mir
#[custom_mir(dialect = "built")]
pub fn simple_ref(x: &mut i32) -> &mut i32 {
    mir! {
        {
            RET = Move(x);
            Return()
        }
    }
}

fn main() {
    assert_eq!(5, simple(5));
}
