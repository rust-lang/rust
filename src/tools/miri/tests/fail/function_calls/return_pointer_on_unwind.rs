// Doesn't need an aliasing model.
//@compile-flags: -Zmiri-disable-stacked-borrows
#![feature(core_intrinsics)]
#![feature(custom_mir)]

use std::intrinsics::mir::*;
use std::panic;

#[repr(C)]
struct S(i32, [u8; 128]);

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn docall(out: &mut S) {
    mir! {
        {
            Call(*out = callee(), ReturnTo(after_call), UnwindContinue())
        }

        after_call = {
            Return()
        }
    }
}

fn startpanic() -> () {
    panic!()
}

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn callee() -> S {
    mir! {
        type RET = S;
        let _unit: ();
        {
            // We test whether changes done to RET before unwinding
            // become visible to the outside. In codegen we can see them
            // but Miri should detect this as UB!
            RET.0 = 42;
            Call(_unit = startpanic(), ReturnTo(after_call), UnwindContinue())
        }

        after_call = {
            Return()
        }
    }
}

fn main() {
    let mut x = S(0, [0; 128]);
    panic::catch_unwind(panic::AssertUnwindSafe(|| docall(&mut x))).unwrap_err();
    // The return place got de-initialized before the call and assigning to RET
    // does not propagate if we do not reach the `Return`.
    dbg!(x.0); //~ERROR: uninitialized
}
