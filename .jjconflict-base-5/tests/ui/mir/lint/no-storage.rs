//@ compile-flags: -Zlint-mir --crate-type=lib -Ztreat-err-as-bug
//@ failure-status: 101
//@ dont-check-compiler-stderr
//@ regex-error-pattern: use of local .*, which has no storage here
#![feature(custom_mir, core_intrinsics)]
extern crate core;
use core::intrinsics::mir::*;

#[custom_mir(dialect = "built")]
pub fn f(a: bool) {
    mir! {
        let b: ();
        {
            match a { true => bb1, _ => bb2 }
        }
        bb1 = {
            StorageLive(b);
            Goto(bb3)
        }
        bb2 = {
            Goto(bb3)
        }
        bb3 = {
            b = (); //~ ERROR broken MIR
            RET = b;
            StorageDead(b);
            Return()
        }
    }
}
