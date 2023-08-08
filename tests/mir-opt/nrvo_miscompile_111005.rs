// This is a miscompilation, #111005 to track

// unit-test: RenameReturnPlace

#![feature(custom_mir, core_intrinsics)]
extern crate core;
use core::intrinsics::mir::*;

// EMIT_MIR nrvo_miscompile_111005.wrong.RenameReturnPlace.diff
#[custom_mir(dialect = "runtime", phase = "initial")]
pub fn wrong(arg: char) -> char {
    mir!({
        let temp = arg;
        RET = temp;
        temp = 'b';
        Return()
    })
}

fn main() {
    assert_eq!(wrong('a'), 'a');
}
