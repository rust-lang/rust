//@compile-flags: -Zmir-move-elimination

#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;
use std::mem::MaybeUninit;

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let ptr: *const MaybeUninit<[u64; 4]>;
        let unit: ();
        {
            let value = const { MaybeUninit::new([1u64; 4]) };
            ptr = &raw const value;
            Call(unit = consume(Move(value), *ptr), ReturnTo(done), UnwindContinue())
            //~^ ERROR: has been freed
        }
        done = { Return() }
    }
}

// Merely writing uninit into the source would not reject this copy. Freeing the
// source must prevent reading it, even though MaybeUninit permits uninit bytes.
fn consume(_: MaybeUninit<[u64; 4]>, _: MaybeUninit<[u64; 4]>) {}
