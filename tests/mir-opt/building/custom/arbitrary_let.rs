// skip-filecheck
#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;
use core::ptr::{addr_of, addr_of_mut};

// EMIT_MIR arbitrary_let.arbitrary_let.built.after.mir
#[custom_mir(dialect = "built")]
fn arbitrary_let(x: i32) -> i32 {
    mir! {
        {
            let y = x;
            Goto(second)
        }
        third = {
            RET = z;
            Return()
        }
        second = {
            let z = y;
            Goto(third)
        }
    }
}

fn main() {
    assert_eq!(arbitrary_let(5), 5);
}
