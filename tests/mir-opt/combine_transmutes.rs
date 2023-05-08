// unit-test: InstSimplify
// compile-flags: -C panic=abort

#![crate_type = "lib"]
#![feature(core_intrinsics)]
#![feature(custom_mir)]

use std::intrinsics::mir::*;
use std::mem::{MaybeUninit, ManuallyDrop, transmute};

// EMIT_MIR combine_transmutes.identity_transmutes.InstSimplify.diff
pub unsafe fn identity_transmutes() {
    // These are nops and should be removed
    let _a = transmute::<i32, i32>(1);
    let _a = transmute::<Vec<i32>, Vec<i32>>(Vec::new());
}

#[custom_mir(dialect = "runtime", phase = "initial")]
// EMIT_MIR combine_transmutes.integer_transmutes.InstSimplify.diff
pub unsafe fn integer_transmutes() {
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
    let _a: u8 = transmute(EnumNoRepr::A);
    let _a: i8 = transmute(EnumNoRepr::B);
    let _a: usize = transmute(EnumReprIsize::A);
    let _a: isize = transmute(EnumReprIsize::B);
    let _a: u8 = transmute(std::cmp::Ordering::Less);
    let _a: i8 = transmute(std::cmp::Ordering::Less);
    let _a: u8 = transmute(Some(std::num::NonZeroU8::MAX));
    let _a: i16 = transmute(std::num::Wrapping(0_i16));
    let _a: u16 = transmute(std::num::Wrapping(0_i16));
    let _a: u32 = transmute(Union32 { i32: 0 });
    let _a: i32 = transmute(Union32 { u32: 0 });
    let _a: ManuallyDrop<String> = transmute(MaybeUninit::<String>::uninit());
}

#[inline(always)]
#[custom_mir(dialect = "runtime", phase = "initial")]
const unsafe fn mir_transmute<T, U>(x: T) -> U {
    mir!{
        {
            RET = CastTransmute(x);
            Return()
        }
    }
}

pub enum EnumNoRepr { A, B, C }

#[repr(isize)]
pub enum EnumReprIsize { A, B, C }

pub union Union32 { u32: u32, i32: i32 }
