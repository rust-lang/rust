// skip-filecheck
#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;

// EMIT_MIR composite_return.tuple.built.after.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn tuple() -> (i32, bool) {
    mir! {
        type RET = (i32, bool);
        {
            RET.0 = 1;
            RET.1 = true;
            Return()
        }
    }
}

fn main() {
    assert_eq!(tuple(), (1, true));
}
