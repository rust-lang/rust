//@revisions: normal move_elimination
//@[normal]check-pass
//@[move_elimination]compile-flags: -Zmir-move-elimination

#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let value: [u8; 2];
        let ptr: *const [u8; 2];
        let unit: ();
        {
            value = [1, 2];
            ptr = &raw const value;
            Call(unit = consume(Move(value), ptr), ReturnTo(done), UnwindContinue())
        }
        done = { Return() }
    }
}

fn consume(_: [u8; 2], ptr: *const [u8; 2]) {
    // The call move frees the source before the callee runs, so even in-bounds
    // pointer arithmetic without a memory access is invalid.
    let next = unsafe { ptr.cast::<u8>().add(1) }; //~[move_elimination] ERROR: has been freed
    assert_eq!(next.addr(), ptr.addr() + 1);
}
