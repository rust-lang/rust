//@ test-mir-pass: ReferencePropagation

#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]
#![crate_type = "lib"]

use std::intrinsics::mir::*;

#[inline(never)]
fn opaque(_: impl Sized, _: impl Sized) {}

#[custom_mir(dialect = "runtime")]
pub fn fn0() {
    // CHECK-LABEL: fn0
    // CHECK: _9 = opaque::<&u8, &u64>(copy (_2.1: &u8), copy _6) -> [return: bb1, unwind unreachable];
    mir! {
        let _1: (u8, u8);
        let _2: (u64, &u8);
        let _3: (u8, &&u64);
        let _4: u64;
        let _5: &u64;
        let _6: &u64;
        let _7: &u64;
        let _8: u64;
        let n: ();
        {
            _3.0 = 0;
            _1 = (0, _3.0);
            _4 = 0;
            _2.1 = &_1.0;
            _8 = 0;
            _5 = &_8;
            _5 = &_4;
            _6 = _5;
            _7 = _6;
            _3.1 = &_6;
            Call(n = opaque(_2.1, Move(_6)), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            _2.0 = *_7;
            Return()
        }
    }
}
