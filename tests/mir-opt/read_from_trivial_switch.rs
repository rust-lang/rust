//@ test-mir-pass: SimplifyCfg-initial

use std::mem::MaybeUninit;

// EMIT_MIR read_from_trivial_switch.main.SimplifyCfg-initial.diff
fn main() {
    let uninit: MaybeUninit<i32> = MaybeUninit::uninit();
    let bad_ref: &i32 = unsafe { uninit.assume_init_ref() };
    // CHECK: switchInt
    let &(0 | _) = bad_ref;
}
