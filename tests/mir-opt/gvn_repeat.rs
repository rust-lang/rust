//@ test-mir-pass: GVN

#![feature(custom_mir, core_intrinsics)]

// Check that we do not introduce out-of-bounds access.

use std::intrinsics::mir::*;

// EMIT_MIR gvn_repeat.repeat_place.GVN.diff
#[custom_mir(dialect = "runtime")]
pub fn repeat_place(mut idx1: usize, idx2: usize, val: &i32) -> i32 {
    // CHECK-LABEL: fn repeat_place(
    // CHECK: let mut [[ELEM:.*]]: &i32;
    // CHECK: _0 = copy (*[[ELEM]])
    mir! {
        let array;
        let elem;
        {
            array = [*val; 5];
            elem = &array[idx1];
            idx1 = idx2;
            RET = *elem;
            Return()
        }
    }
}

// EMIT_MIR gvn_repeat.repeat_local.GVN.diff
#[custom_mir(dialect = "runtime")]
pub fn repeat_local(mut idx1: usize, idx2: usize, val: i32) -> i32 {
    // CHECK-LABEL: fn repeat_local(
    // CHECK: _0 = copy _3
    mir! {
        let array;
        let elem;
        {
            array = [val; 5];
            elem = &array[idx1];
            idx1 = idx2;
            RET = *elem;
            Return()
        }
    }
}

fn main() {
    assert_eq!(repeat_place(0, 5, &0), 0);
    assert_eq!(repeat_local(0, 5, 0), 0);
}
