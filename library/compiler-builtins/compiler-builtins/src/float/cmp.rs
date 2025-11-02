#![allow(unreachable_code)]

use crate::float::Float;
use crate::int::MinInt;
use crate::support::cfg_if;

// Taken from LLVM config:
// https://github.com/llvm/llvm-project/blob/0cf3c437c18ed27d9663d87804a9a15ff6874af2/compiler-rt/lib/builtins/fp_compare_impl.inc#L11-L27
cfg_if! {
    if #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))] {
        // Aarch64 uses `int` rather than a pointer-sized value.
        pub type CmpResult = i32;
    } else if #[cfg(target_arch = "avr")] {
        // AVR uses a single byte.
        pub type CmpResult = i8;
    } else {
        // In compiler-rt, LLP64 ABIs use `long long` and everything else uses `long`. In effect,
        // this means the return value is always pointer-sized.
        pub type CmpResult = isize;
    }
}

#[derive(Clone, Copy)]
enum Result {
    Less,
    Equal,
    Greater,
    Unordered,
}

impl Result {
    fn to_le_abi(self) -> CmpResult {
        match self {
            Result::Less => -1,
            Result::Equal => 0,
            Result::Greater => 1,
            Result::Unordered => 1,
        }
    }

    fn to_ge_abi(self) -> CmpResult {
        match self {
            Result::Less => -1,
            Result::Equal => 0,
            Result::Greater => 1,
            Result::Unordered => -1,
        }
    }
}

fn cmp<F: Float>(a: F, b: F) -> Result {
    let one = F::Int::ONE;
    let zero = F::Int::ZERO;
    let szero = F::SignedInt::ZERO;

    let sign_bit = F::SIGN_MASK as F::Int;
    let abs_mask = sign_bit - one;
    let exponent_mask = F::EXP_MASK;
    let inf_rep = exponent_mask;

    let a_rep = a.to_bits();
    let b_rep = b.to_bits();
    let a_abs = a_rep & abs_mask;
    let b_abs = b_rep & abs_mask;

    // If either a or b is NaN, they are unordered.
    if a_abs > inf_rep || b_abs > inf_rep {
        return Result::Unordered;
    }

    // If a and b are both zeros, they are equal.
    if a_abs | b_abs == zero {
        return Result::Equal;
    }

    let a_srep = a.to_bits_signed();
    let b_srep = b.to_bits_signed();

    // If at least one of a and b is positive, we get the same result comparing
    // a and b as signed integers as we would with a fp_ting-point compare.
    if a_srep & b_srep >= szero {
        if a_srep < b_srep {
            Result::Less
        } else if a_srep == b_srep {
            Result::Equal
        } else {
            Result::Greater
        }
    // Otherwise, both are negative, so we need to flip the sense of the
    // comparison to get the correct result.  (This assumes a twos- or ones-
    // complement integer representation; if integers are represented in a
    // sign-magnitude representation, then this flip is incorrect).
    } else if a_srep > b_srep {
        Result::Less
    } else if a_srep == b_srep {
        Result::Equal
    } else {
        Result::Greater
    }
}

fn unord<F: Float>(a: F, b: F) -> bool {
    let one = F::Int::ONE;

    let sign_bit = F::SIGN_MASK as F::Int;
    let abs_mask = sign_bit - one;
    let exponent_mask = F::EXP_MASK;
    let inf_rep = exponent_mask;

    let a_rep = a.to_bits();
    let b_rep = b.to_bits();
    let a_abs = a_rep & abs_mask;
    let b_abs = b_rep & abs_mask;

    a_abs > inf_rep || b_abs > inf_rep
}

#[cfg(f16_enabled)]
intrinsics! {
    pub extern "C" fn __lehf2(a: f16, b: f16) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_le_abi()
    }

    pub extern "C" fn __gehf2(a: f16, b: f16) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_ge_abi()
    }

    pub extern "C" fn __unordhf2(a: f16, b: f16) -> crate::float::cmp::CmpResult {
        unord(a, b) as crate::float::cmp::CmpResult
    }

    pub extern "C" fn __eqhf2(a: f16, b: f16) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_le_abi()
    }

    pub extern "C" fn __lthf2(a: f16, b: f16) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_le_abi()
    }

    pub extern "C" fn __nehf2(a: f16, b: f16) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_le_abi()
    }

    pub extern "C" fn __gthf2(a: f16, b: f16) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_ge_abi()
    }
}

intrinsics! {
    pub extern "C" fn __lesf2(a: f32, b: f32) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_le_abi()
    }

    pub extern "C" fn __gesf2(a: f32, b: f32) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_ge_abi()
    }

    #[arm_aeabi_alias = __aeabi_fcmpun]
    pub extern "C" fn __unordsf2(a: f32, b: f32) -> crate::float::cmp::CmpResult {
        unord(a, b) as crate::float::cmp::CmpResult
    }

    pub extern "C" fn __eqsf2(a: f32, b: f32) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_le_abi()
    }

    pub extern "C" fn __ltsf2(a: f32, b: f32) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_le_abi()
    }

    pub extern "C" fn __nesf2(a: f32, b: f32) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_le_abi()
    }

    pub extern "C" fn __gtsf2(a: f32, b: f32) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_ge_abi()
    }

    pub extern "C" fn __ledf2(a: f64, b: f64) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_le_abi()
    }

    pub extern "C" fn __gedf2(a: f64, b: f64) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_ge_abi()
    }

    #[arm_aeabi_alias = __aeabi_dcmpun]
    pub extern "C" fn __unorddf2(a: f64, b: f64) -> crate::float::cmp::CmpResult {
        unord(a, b) as crate::float::cmp::CmpResult
    }

    pub extern "C" fn __eqdf2(a: f64, b: f64) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_le_abi()
    }

    pub extern "C" fn __ltdf2(a: f64, b: f64) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_le_abi()
    }

    pub extern "C" fn __nedf2(a: f64, b: f64) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_le_abi()
    }

    pub extern "C" fn __gtdf2(a: f64, b: f64) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_ge_abi()
    }
}

#[cfg(f128_enabled)]
intrinsics! {
    #[ppc_alias = __lekf2]
    pub extern "C" fn __letf2(a: f128, b: f128) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_le_abi()
    }

    #[ppc_alias = __gekf2]
    pub extern "C" fn __getf2(a: f128, b: f128) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_ge_abi()
    }

    #[ppc_alias = __unordkf2]
    pub extern "C" fn __unordtf2(a: f128, b: f128) -> crate::float::cmp::CmpResult {
        unord(a, b) as crate::float::cmp::CmpResult
    }

    #[ppc_alias = __eqkf2]
    pub extern "C" fn __eqtf2(a: f128, b: f128) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_le_abi()
    }

    #[ppc_alias = __ltkf2]
    pub extern "C" fn __lttf2(a: f128, b: f128) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_le_abi()
    }

    #[ppc_alias = __nekf2]
    pub extern "C" fn __netf2(a: f128, b: f128) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_le_abi()
    }

    #[ppc_alias = __gtkf2]
    pub extern "C" fn __gttf2(a: f128, b: f128) -> crate::float::cmp::CmpResult {
        cmp(a, b).to_ge_abi()
    }
}

#[cfg(target_arch = "arm")]
intrinsics! {
    pub extern "aapcs" fn __aeabi_fcmple(a: f32, b: f32) -> i32 {
        (__lesf2(a, b) <= 0) as i32
    }

    pub extern "aapcs" fn __aeabi_fcmpge(a: f32, b: f32) -> i32 {
        (__gesf2(a, b) >= 0) as i32
    }

    pub extern "aapcs" fn __aeabi_fcmpeq(a: f32, b: f32) -> i32 {
        (__eqsf2(a, b) == 0) as i32
    }

    pub extern "aapcs" fn __aeabi_fcmplt(a: f32, b: f32) -> i32 {
        (__ltsf2(a, b) < 0) as i32
    }

    pub extern "aapcs" fn __aeabi_fcmpgt(a: f32, b: f32) -> i32 {
        (__gtsf2(a, b) > 0) as i32
    }

    pub extern "aapcs" fn __aeabi_dcmple(a: f64, b: f64) -> i32 {
        (__ledf2(a, b) <= 0) as i32
    }

    pub extern "aapcs" fn __aeabi_dcmpge(a: f64, b: f64) -> i32 {
        (__gedf2(a, b) >= 0) as i32
    }

    pub extern "aapcs" fn __aeabi_dcmpeq(a: f64, b: f64) -> i32 {
        (__eqdf2(a, b) == 0) as i32
    }

    pub extern "aapcs" fn __aeabi_dcmplt(a: f64, b: f64) -> i32 {
        (__ltdf2(a, b) < 0) as i32
    }

    pub extern "aapcs" fn __aeabi_dcmpgt(a: f64, b: f64) -> i32 {
        (__gtdf2(a, b) > 0) as i32
    }
}
