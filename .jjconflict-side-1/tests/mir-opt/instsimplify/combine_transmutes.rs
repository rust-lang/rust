//@ test-mir-pass: InstSimplify-after-simplifycfg
//@ compile-flags: -C panic=abort
#![crate_type = "lib"]
#![feature(core_intrinsics)]
#![feature(custom_mir)]

use std::intrinsics::mir::*;
use std::mem::{ManuallyDrop, MaybeUninit, transmute};

// EMIT_MIR combine_transmutes.identity_transmutes.InstSimplify-after-simplifycfg.diff
pub unsafe fn identity_transmutes() {
    // CHECK-LABEL: fn identity_transmutes(
    // CHECK-NOT: as i32 (Transmute);
    // CHECK-NOT: as Vec<i32> (Transmute);

    // These are nops and should be removed
    let _a = transmute::<i32, i32>(1);
    let _a = transmute::<Vec<i32>, Vec<i32>>(Vec::new());
}

#[custom_mir(dialect = "runtime", phase = "initial")]
// EMIT_MIR combine_transmutes.integer_transmutes.InstSimplify-after-simplifycfg.diff
pub unsafe fn integer_transmutes() {
    // CHECK-LABEL: fn integer_transmutes(
    // CHECK-NOT: _i32 as u32 (Transmute);
    // CHECK: _i32 as u32 (IntToInt);
    // CHECK: _i32 as i64 (Transmute);
    // CHECK-NOT: _u64 as i64 (Transmute);
    // CHECK: _u64 as i64 (IntToInt);
    // CHECK: _u64 as u32 (Transmute);
    // CHECK-NOT: _isize as usize (Transmute);
    // CHECK: _isize as usize (IntToInt);

    mir! {
        {
            let A = CastTransmute::<i32, u32>(1); // Can be a cast
            let B = CastTransmute::<i32, i64>(1); // UB
            let C = CastTransmute::<u64, i64>(1); // Can be a cast
            let D = CastTransmute::<u64, u32>(1); // UB
            let E = CastTransmute::<isize, usize>(1); // Can be a cast
            Return()
        }
    }
}

// EMIT_MIR combine_transmutes.keep_transparent_transmute.InstSimplify-after-simplifycfg.diff
pub unsafe fn keep_transparent_transmute() {
    // CHECK-LABEL: fn keep_transparent_transmute(
    // CHECK-NOT: .{{[0-9]+}}: i16
    // CHECK: as i16 (Transmute);
    // CHECK-NOT: .{{[0-9]+}}: i16
    // CHECK: as i16 (Transmute);
    // CHECK-NOT: .{{[0-9]+}}: i16

    // Transmutes should not be converted to field accesses, because MCP#807
    // bans projections into `[rustc_layout_scalar_valid_range_*]` types.
    let _a: i16 = transmute(const { std::num::NonZero::new(12345_i16).unwrap() });
    let _a: i16 = transmute(std::num::Wrapping(0_i16));
}

pub union Union32 {
    u32: u32,
    i32: i32,
}
