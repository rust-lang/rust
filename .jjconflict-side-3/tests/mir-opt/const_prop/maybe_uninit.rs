//@ compile-flags: -O

use std::mem::MaybeUninit;

// EMIT_MIR maybe_uninit.u8_array.GVN.diff
pub fn u8_array() -> [MaybeUninit<u8>; 8] {
    // CHECK: fn u8_array(
    // CHECK: _0 = const <uninit>;
    [MaybeUninit::uninit(); 8]
}
