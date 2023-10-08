// This is a miscompilation, #111005 to track
// needs-unwind

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

// EMIT_MIR nrvo_miscompile_111005.indirect.RenameReturnPlace.diff
#[custom_mir(dialect = "runtime", phase = "initial")]
pub fn indirect(arg: char) -> char {
    mir!({
        let temp = arg;
        let temp_addr = &mut temp;
        RET = temp;
        *temp_addr = 'b';
        Return()
    })
}

// EMIT_MIR nrvo_miscompile_111005.moved.RenameReturnPlace.diff
#[custom_mir(dialect = "runtime", phase = "initial")]
pub fn moved(arg: char) -> char {
    mir!({
        let temp = arg;
        RET = temp;
        let temp2 = Move(temp);
        Return()
    })
}

// EMIT_MIR nrvo_miscompile_111005.multiple.RenameReturnPlace.diff
#[custom_mir(dialect = "runtime", phase = "initial")]
pub fn multiple(arg: char) -> char {
    mir!({
        let temp = arg;
        let temp2 = arg;
        RET = temp2;
        RET = temp;
        temp = 'b';
        Return()
    })
}

// EMIT_MIR nrvo_miscompile_111005.projection.RenameReturnPlace.diff
#[custom_mir(dialect = "runtime", phase = "initial")]
pub fn projection(arg: char) -> (char, u8) {
    mir!({
        let temp = (arg, 0);
        RET = temp;
        // We write to `temp` after assigning to `RET`. Discard it.
        temp.1 = 5;
        Return()
    })
}

// EMIT_MIR nrvo_miscompile_111005.multiple_return.RenameReturnPlace.diff
#[custom_mir(dialect = "runtime", phase = "initial")]
pub fn multiple_return(arg: char, test: bool) -> char {
    mir!({
        match test { false => bb1, _ => bb2 }
    }
    // We cannot merge `RET` with both `temp` and `temp2`, so do nothing.
    // We rely on `DestinationPropagation` to handle the harder case.
    bb1 = {
        let temp = arg;
        RET = temp;
        Return()
    }
    bb2 = {
        let temp2 = 'z';
        RET = temp2;
        Return()
    })
}

// EMIT_MIR nrvo_miscompile_111005.call.RenameReturnPlace.diff
#[custom_mir(dialect = "runtime", phase = "initial")]
pub fn call(arg: char) -> char {
    mir!(
        {
            // Check that we do not ICE. #110902
            Call(RET = wrong(arg), bb1)
        }
        bb1 = {
            let temp = arg;
            RET = temp;
            // Discard from NRVO.
            Call(temp = wrong(arg), bb2)
        }
        bb2 = {
            Return()
        }
    )
}

// EMIT_MIR nrvo_miscompile_111005.call_ok.RenameReturnPlace.diff
#[custom_mir(dialect = "runtime", phase = "initial")]
pub fn call_ok(arg: char) -> char {
    mir!(
        let temp: char;
        {
            // Check that we do not ICE. #110902
            Call(RET = wrong(arg), bb1)
        }
        bb1 = {
            Call(temp = wrong(arg), bb2)
        }
        bb2 = {
            // This is ok.
            RET = temp;
            Return()
        }
    )
}

// EMIT_MIR nrvo_miscompile_111005.return_projection.RenameReturnPlace.diff
#[custom_mir(dialect = "runtime", phase = "initial")]
pub fn return_projection(arg: char) -> (char, u8) {
    mir!({
        let temp = (arg, 0);
        RET = temp;
        // FIXME: Writing to `RET` could be fine, but we skip it as a precaution.
        RET.1 = 5;
        Return()
    })
}

fn main() {
    assert_eq!(wrong('a'), 'a');
    assert_eq!(indirect('a'), 'a');
    assert_eq!(moved('a'), 'a');
    assert_eq!(multiple('a'), 'a');
    assert_eq!(call('a'), 'a');
    assert_eq!(call_ok('a'), 'a');
    assert_eq!(multiple_return('a', true), 'z');
    assert_eq!(projection('a'), ('a', 0));
    assert_eq!(return_projection('a'), ('a', 5));
}
