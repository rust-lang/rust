//@ revisions: a b
//@ needs-unwind

#![feature(core_intrinsics, custom_mir, rustc_attrs)]
#![crate_type = "lib"]
use std::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
#[cfg_attr(a, rustc_mir(rustc_pretty_live_locals))]
#[cfg_attr(b, rustc_mir(rustc_pretty_transitive_live_locals))]
pub fn f(mut _1: i32) {
    mir! {
        {
            Goto(bb1)
        }
        bb1 = {
            // CHECK-LABEL: fn f(

            // a:      bb1: {
            // a-NEXT: live: [_1]
            // a-NEXT: _2 = copy _1;
            // a-NEXT: live: [_2]
            // a-NEXT: _3 = copy _2;
            // a-NEXT: live: [_3]
            // a-NEXT: _1 = copy _3;

            // b:      bb1: {
            // b-NEXT: live: []
            // b-NEXT: _2 = copy _1;
            // b-NEXT: live: []
            // b-NEXT: _3 = copy _2;
            // b-NEXT: live: []
            // b-NEXT: _1 = copy _3;
            let _2 = _1;
            let _3 = _2;
            _1 = _3;
            Goto(bb1)
        }
    }
}
