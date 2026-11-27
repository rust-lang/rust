//@ known-bug: #119940
//@ compile-flags: -Zvalidate-mir

#![feature(custom_mir, core_intrinsics)]
extern crate core;
use core::intrinsics::mir::*;

pub enum E {
    V0 { fld0: &'static u64 },
}

#[custom_mir(dialect = "runtime", phase = "initial")]
pub fn fn0() {
    mir! {
    let e: E;
    let n: u64;
    {
        n = 0;
        place!(Field::<&u64>(Variant(e, 0), 0)) = &n;
        Return()
    }

    }
}
pub fn main() {
    fn0();
}
