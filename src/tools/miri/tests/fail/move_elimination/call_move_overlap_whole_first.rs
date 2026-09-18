//@revisions: normal move_elimination
//@[move_elimination]compile-flags: -Zmir-move-elimination

// Moving a whole local makes its storage inaccessible to a later field move,
// through protection normally or deallocation under move-elimination semantics.

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
            Call(unit = whole_field(Move(value), Move(value.0)), ReturnTo(done), UnwindContinue())
        }
        done = { Return() }
    }
}

fn whole_field(_: Value, _: MaybeUninit<u32>) {
    //~[normal]^ ERROR: protected
    //~[move_elimination]| ERROR: has been freed
}
