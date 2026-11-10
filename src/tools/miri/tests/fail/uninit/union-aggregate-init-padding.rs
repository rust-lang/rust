//! Ensure that initalizing an aggregate resets padding to uninit.
//! Regression test for <https://github.com/rust-lang/miri/issues/5554>.
#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features, unused)]

use std::intrinsics::mir::*;

#[derive(Clone, Copy)]
union U {
    a: u8,
    b: u32,
}

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn f() -> u32 {
    mir! {
        let u: U;
        let x: u32;
        {
            x = 0x01020304_u32;
            u = U { b: x };
            // This creates a new union value where `a` is initialized and the remaining bytes
            // remain uninitialized. That then gets stored into `u`. As a result, this assignment
            // should reset the padding bytes of `U::a` in `u` to uninit.
            u = U { a: 9_u8 };
            RET = CastTransmute::<U, u32>(u); //~ERROR: uninitialized
            Return()
        }
    }
}

fn main() {
    f();
}
