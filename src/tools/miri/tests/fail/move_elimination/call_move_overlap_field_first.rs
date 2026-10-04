//@revisions: normal move_elimination
//@[move_elimination]compile-flags: -Zmir-move-elimination

// A projected move retains its source place: a later whole move cannot hide the overlap.

#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;
use std::mem::MaybeUninit;

type Value = (MaybeUninit<u32>, MaybeUninit<u32>);

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let unit: ();
        {
            let value = const { (MaybeUninit::new(1u32), MaybeUninit::new(2u32)) };
            Call(unit = field_whole(Move(value.0), Move(value)), ReturnTo(done), UnwindContinue())
        }
        done = { Return() }
    }
}

// Permit uninitialized bytes so the normal revision tests protection, not value validity.
fn field_whole(_: MaybeUninit<u32>, _: Value) {
    //~[normal]^ ERROR: protected
    //~[move_elimination]| ERROR: has been freed
}
