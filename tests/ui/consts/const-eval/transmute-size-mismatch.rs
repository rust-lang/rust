#![feature(core_intrinsics)]
#![feature(custom_mir)]

// These cases are statically rejected by `mem::transmute`, so we need custom
// MIR to be able to get to constant evaluation.
use std::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "initial")]
const unsafe fn mir_transmute<T, U>(x: T) -> U {
    mir!{
        {
            RET = CastTransmute(x);
            //~^ NOTE inside `mir_transmute
            //~| NOTE inside `mir_transmute
            //~| NOTE the failure occurred here
            //~| NOTE the failure occurred here
            Return()
        }
    }
}

const FROM_BIGGER: u16 = unsafe { mir_transmute(123_i32) }; //~ NOTE evaluation of constant value failed
//~^ ERROR transmuting from 4-byte type to 2-byte type: `i32` -> `u16`

const FROM_SMALLER: u32 = unsafe { mir_transmute(123_i16) }; //~ NOTE evaluation of constant value failed
//~^ ERROR transmuting from 2-byte type to 4-byte type: `i16` -> `u32`

fn main() {}
