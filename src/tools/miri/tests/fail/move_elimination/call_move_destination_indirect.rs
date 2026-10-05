//@compile-flags: -Zmir-move-elimination

#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;

// A destination pointing into moved storage must fail even if the callee never returns.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let ptr: *mut u32;
        {
            let value = 42u32;
            ptr = &raw mut value;
            Call(*ptr = consume(Move(value)), ReturnTo(done), UnwindContinue())
        }
        done = { Return() }
    }
}
fn consume(_: u32) -> u32 {
    //~^ ERROR: has been freed
    std::process::exit(0)
}
