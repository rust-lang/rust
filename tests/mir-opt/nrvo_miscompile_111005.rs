// This is a miscompilation, #111005 to track

//@ test-mir-pass: RenameReturnPlace

#![feature(custom_mir, core_intrinsics)]
extern crate core;
use core::intrinsics::mir::*;

// EMIT_MIR nrvo_miscompile_111005.wrong.RenameReturnPlace.diff
#[custom_mir(dialect = "runtime", phase = "initial")]
pub fn wrong(arg: char) -> char {
    // CHECK-LABEL: fn wrong(
    // CHECK: _0 = copy _1;
    // FIXME: This is wrong:
    // CHECK-NEXT: _0 = const 'b';
    // CHECK-NEXT: return;
    mir! {
        {
            let temp = arg;
            RET = temp;
            temp = 'b';
            Return()
        }
    }
}

fn main() {
    assert_eq!(wrong('a'), 'a');
}
