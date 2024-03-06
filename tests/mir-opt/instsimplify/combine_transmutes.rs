//@ unit-test: InstSimplify
//@ compile-flags: -C panic=abort
#![crate_type = "lib"]
#![feature(core_intrinsics)]
#![feature(custom_mir)]
#![feature(generic_nonzero)]

use std::intrinsics::mir::*;
use std::mem::{MaybeUninit, ManuallyDrop, transmute};

// EMIT_MIR combine_transmutes.identity_transmutes.InstSimplify.diff
pub unsafe fn identity_transmutes() {
    // CHECK-LABEL: fn identity_transmutes(
    // CHECK-NOT: as i32 (Transmute);
    // CHECK-NOT: as Vec<i32> (Transmute);

    // These are nops and should be removed
    let _a = transmute::<i32, i32>(1);
    let _a = transmute::<Vec<i32>, Vec<i32>>(Vec::new());
}

#[custom_mir(dialect = "runtime", phase = "initial")]
// EMIT_MIR combine_transmutes.integer_transmutes.InstSimplify.diff
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

// EMIT_MIR combine_transmutes.adt_transmutes.InstSimplify.diff
pub unsafe fn adt_transmutes() {
    // CHECK-LABEL: fn adt_transmutes(
    // CHECK: as u8 (Transmute);
    // CHECK: ({{_.*}}.0: i16);
    // CHECK: as u16 (Transmute);
    // CHECK: as u32 (Transmute);
    // CHECK: as i32 (Transmute);
    // CHECK: ({{_.*}}.1: std::mem::ManuallyDrop<std::string::String>);

    let _a: u8 = transmute(Some(std::num::NonZero::<u8>::MAX));
    let _a: i16 = transmute(std::num::Wrapping(0_i16));
    let _a: u16 = transmute(std::num::Wrapping(0_i16));
    let _a: u32 = transmute(Union32 { i32: 0 });
    let _a: i32 = transmute(Union32 { u32: 0 });
    let _a: ManuallyDrop<String> = transmute(MaybeUninit::<String>::uninit());
}

pub union Union32 { u32: u32, i32: i32 }
