//@compile-flags: -Zmiri-disable-stacked-borrows
#![feature(custom_mir, core_intrinsics)]

use std::intrinsics::mir::*;

// Call setup clears the return destination even if it is stored as an immediate.
// Arguments must still receive its old value before it is cleared.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn call() -> u32 {
    mir! {
        {
            let value = 42u32;
            Call(value = callee(value), ReturnTo(done), UnwindCleanup(cleanup))
        }
        done = {
            RET = value;
            Return()
        }
        cleanup (cleanup) = {
            RET = value; //~ERROR: uninitialized
            UnwindResume()
        }
    }
}

fn callee(value: u32) -> u32 {
    assert_eq!(value, 42);
    panic!("unwind");
}

fn main() {
    let _ = std::panic::catch_unwind(call);
}
