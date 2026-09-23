//@revisions: stack tree none
//@[tree]compile-flags: -Zmiri-tree-borrows
//@[none]compile-flags: -Zmiri-disable-stacked-borrows

#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;

// A moved variadic argument must be inaccessible before the next argument is copied.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let ptr: *const u64;
        let unit: ();
        {
            let value = 1u64;
            ptr = &raw const value;
            Call(unit = consume(Move(value), *ptr), ReturnTo(done), UnwindContinue())
        }
        done = { Return() }
    }
}

unsafe extern "C" fn consume(_: ...) {
    //~[stack]^ ERROR: tag does not exist in the borrow stack
    //~[tree]| ERROR: /read access .* forbidden/
    //~[none]| ERROR: uninitialized
}
