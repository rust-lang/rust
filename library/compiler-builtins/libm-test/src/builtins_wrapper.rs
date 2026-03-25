//! Wrappers around compiler-builtins functions.
//!
//! Functions from compiler-builtins have a different naming scheme from libm and often a different
//! ABI (doesn't work with libm-test traits because that changes the type signature). Wrap these
//! to make them a bit more similar to the rest of the libm functions.

macro_rules! cb_op {
    // Fully generic version
    ($mod:ident, $cb_name:ident, $new_name:ident, ($($arg:ident: $ArgTy:ty),*) -> $RetTy:ty) => {
        pub fn $new_name($($arg: $ArgTy),*) -> $RetTy {
            compiler_builtins::float::$mod::$cb_name($($arg),*)
        }
    };
    (@int $mod:ident, $cb_name:ident, $new_name:ident, ($($arg:ident: $ArgTy:ty),*) -> $RetTy:ty) => {
        pub fn $new_name($($arg: $ArgTy),*) -> $RetTy {
            compiler_builtins::int::$mod::$cb_name($($arg),*)
        }
    };

    // Common signatures
    (@binop $ty:ty, $mod:ident, $cb_name:ident, $new_name:ident) => {
        cb_op!($mod, $cb_name, $new_name, (a: $ty, b: $ty) -> $ty);
    };

    // Cmp signatures. See the documentation in cmp.rs regarding the result.
    (@cmp_eq $ty:ty, $mod:ident, $cb_name:ident, $new_name:ident) => {
        pub fn $new_name(a: $ty, b: $ty) -> bool {
            compiler_builtins::float::$mod::$cb_name(a, b) == 0
        }
    };
    (@cmp_ne $ty:ty, $mod:ident, $cb_name:ident, $new_name:ident) => {
        pub fn $new_name(a: $ty, b: $ty) -> bool {
            compiler_builtins::float::$mod::$cb_name(a, b) != 0
        }
    };
    (@cmp_unord $ty:ty, $mod:ident, $cb_name:ident, $new_name:ident) => {
        pub fn $new_name(a: $ty, b: $ty) -> bool {
            compiler_builtins::float::$mod::$cb_name(a, b) != 0
        }
    };
    (@cmp_lt $ty:ty, $mod:ident, $cb_name:ident, $new_name:ident) => {
        pub fn $new_name(a: $ty, b: $ty) -> bool {
            compiler_builtins::float::$mod::$cb_name(a, b) < 0
        }
    };
    (@cmp_le $ty:ty, $mod:ident, $cb_name:ident, $new_name:ident) => {
        pub fn $new_name(a: $ty, b: $ty) -> bool {
            compiler_builtins::float::$mod::$cb_name(a, b) <= 0
        }
    };
    (@cmp_gt $ty:ty, $mod:ident, $cb_name:ident, $new_name:ident) => {
        pub fn $new_name(a: $ty, b: $ty) -> bool {
            compiler_builtins::float::$mod::$cb_name(a, b) > 0
        }
    };
    (@cmp_ge $ty:ty, $mod:ident, $cb_name:ident, $new_name:ident) => {
        pub fn $new_name(a: $ty, b: $ty) -> bool {
            compiler_builtins::float::$mod::$cb_name(a, b) >= 0
        }
    };
}

#[cfg(f16_enabled)]
cb_op!(@binop f16, add, __addhf3, addf16);
cb_op!(@binop f32, add, __addsf3, addf32);
cb_op!(@binop f64, add, __adddf3, addf64);
#[cfg(f128_enabled)]
cb_op!(@binop f128, add, __addtf3, addf128);

#[cfg(f16_enabled)]
cb_op!(@binop f16, sub, __subhf3, subf16);
cb_op!(@binop f32, sub, __subsf3, subf32);
cb_op!(@binop f64, sub, __subdf3, subf64);
#[cfg(f128_enabled)]
cb_op!(@binop f128, sub, __subtf3, subf128);

#[cfg(f16_enabled)]
cb_op!(@binop f16, mul, __mulhf3, mulf16);
cb_op!(@binop f32, mul, __mulsf3, mulf32);
cb_op!(@binop f64, mul, __muldf3, mulf64);
#[cfg(f128_enabled)]
cb_op!(@binop f128, mul, __multf3, mulf128);

cb_op!(@binop f32, div, __divsf3, divf32);
cb_op!(@binop f64, div, __divdf3, divf64);
#[cfg(f128_enabled)]
cb_op!(@binop f128, div, __divtf3, divf128);

cb_op!(pow, __powisf2, powif32, (a: f32, b: i32) -> f32);
cb_op!(pow, __powidf2, powif64, (a: f64, b: i32) -> f64);
#[cfg(f128_enabled)]
cb_op!(pow, __powitf2, powif128, (a: f128, b: i32) -> f128);

#[cfg(f16_enabled)]
cb_op!(@cmp_eq f16, cmp, __eqhf2, eqf16);
cb_op!(@cmp_eq f32, cmp, __eqsf2, eqf32);
cb_op!(@cmp_eq f64, cmp, __eqdf2, eqf64);
#[cfg(f128_enabled)]
cb_op!(@cmp_eq f128, cmp, __eqtf2, eqf128);

#[cfg(f16_enabled)]
cb_op!(@cmp_gt f16, cmp, __gthf2, gtf16);
cb_op!(@cmp_gt f32, cmp, __gtsf2, gtf32);
cb_op!(@cmp_gt f64, cmp, __gtdf2, gtf64);
#[cfg(f128_enabled)]
cb_op!(@cmp_gt f128, cmp, __gttf2, gtf128);

#[cfg(f16_enabled)]
cb_op!(@cmp_ge f16, cmp, __gehf2, gef16);
cb_op!(@cmp_ge f32, cmp, __gesf2, gef32);
cb_op!(@cmp_ge f64, cmp, __gedf2, gef64);
#[cfg(f128_enabled)]
cb_op!(@cmp_ge f128, cmp, __getf2, gef128);

#[cfg(f16_enabled)]
cb_op!(@cmp_lt f16, cmp, __lthf2, ltf16);
cb_op!(@cmp_lt f32, cmp, __ltsf2, ltf32);
cb_op!(@cmp_lt f64, cmp, __ltdf2, ltf64);
#[cfg(f128_enabled)]
cb_op!(@cmp_lt f128, cmp, __lttf2, ltf128);

#[cfg(f16_enabled)]
cb_op!(@cmp_le f16, cmp, __lehf2, lef16);
cb_op!(@cmp_le f32, cmp, __lesf2, lef32);
cb_op!(@cmp_le f64, cmp, __ledf2, lef64);
#[cfg(f128_enabled)]
cb_op!(@cmp_le f128, cmp, __letf2, lef128);

#[cfg(f16_enabled)]
cb_op!(@cmp_ne f16, cmp, __nehf2, nef16);
cb_op!(@cmp_ne f32, cmp, __nesf2, nef32);
cb_op!(@cmp_ne f64, cmp, __nedf2, nef64);
#[cfg(f128_enabled)]
cb_op!(@cmp_ne f128, cmp, __netf2, nef128);

#[cfg(f16_enabled)]
cb_op!(@cmp_unord f16, cmp, __unordhf2, unordf16);
cb_op!(@cmp_unord f32, cmp, __unordsf2, unordf32);
cb_op!(@cmp_unord f64, cmp, __unorddf2, unordf64);
#[cfg(f128_enabled)]
cb_op!(@cmp_unord f128, cmp, __unordtf2, unordf128);

#[cfg(f16_enabled)]
cb_op!(extend, __extendhfsf2, extend_f16_f32, (a: f16) -> f32);
#[cfg(f16_enabled)]
cb_op!(extend, __extendhfdf2, extend_f16_f64, (a: f16) -> f64);
#[cfg(f16_enabled)]
#[cfg(f128_enabled)]
cb_op!(extend, __extendhftf2, extend_f16_f128, (a: f16) -> f128);
cb_op!(extend, __extendsfdf2, extend_f32_f64, (a: f32) -> f64);
#[cfg(f128_enabled)]
cb_op!(extend, __extendsftf2, extend_f32_f128, (a: f32) -> f128);
#[cfg(f128_enabled)]
cb_op!(extend, __extenddftf2, extend_f64_f128, (a: f64) -> f128);

// Note that these are renamed from trunc to narrow to avoid collision with libm `trunc`.
#[cfg(f16_enabled)]
cb_op!(trunc, __truncsfhf2, narrow_f32_f16, (a: f32) -> f16);
#[cfg(f16_enabled)]
cb_op!(trunc, __truncdfhf2, narrow_f64_f16, (a: f64) -> f16);
cb_op!(trunc, __truncdfsf2, narrow_f64_f32, (a: f64) -> f32);
#[cfg(f16_enabled)]
#[cfg(f128_enabled)]
cb_op!(trunc, __trunctfhf2, narrow_f128_f16, (a: f128) -> f16);
#[cfg(f128_enabled)]
cb_op!(trunc, __trunctfsf2, narrow_f128_f32, (a: f128) -> f32);
#[cfg(f128_enabled)]
cb_op!(trunc, __trunctfdf2, narrow_f128_f64, (a: f128) -> f64);

cb_op!(conv, __fixsfsi, ftoi_f32_i32, (a: f32) -> i32);
cb_op!(conv, __fixsfdi, ftoi_f32_i64, (a: f32) -> i64);
cb_op!(conv, __fixsfti, ftoi_f32_i128, (a: f32) -> i128);
cb_op!(conv, __fixdfsi, ftoi_f64_i32, (a: f64) -> i32);
cb_op!(conv, __fixdfdi, ftoi_f64_i64, (a: f64) -> i64);
cb_op!(conv, __fixdfti, ftoi_f64_i128, (a: f64) -> i128);
#[cfg(f128_enabled)]
cb_op!(conv, __fixtfsi, ftoi_f128_i32, (a: f128) -> i32);
#[cfg(f128_enabled)]
cb_op!(conv, __fixtfdi, ftoi_f128_i64, (a: f128) -> i64);
#[cfg(f128_enabled)]
cb_op!(conv, __fixtfti, ftoi_f128_i128, (a: f128) -> i128);
cb_op!(conv, __fixunssfsi, ftoi_f32_u32, (a: f32) -> u32);
cb_op!(conv, __fixunssfdi, ftoi_f32_u64, (a: f32) -> u64);
cb_op!(conv, __fixunssfti, ftoi_f32_u128, (a: f32) -> u128);
cb_op!(conv, __fixunsdfsi, ftoi_f64_u32, (a: f64) -> u32);
cb_op!(conv, __fixunsdfdi, ftoi_f64_u64, (a: f64) -> u64);
cb_op!(conv, __fixunsdfti, ftoi_f64_u128, (a: f64) -> u128);
#[cfg(f128_enabled)]
cb_op!(conv, __fixunstfsi, ftoi_f128_u32, (a: f128) -> u32);
#[cfg(f128_enabled)]
cb_op!(conv, __fixunstfdi, ftoi_f128_u64, (a: f128) -> u64);
#[cfg(f128_enabled)]
cb_op!(conv, __fixunstfti, ftoi_f128_u128, (a: f128) -> u128);

cb_op!(conv, __floatsisf, itof_i32_f32, (a: i32) -> f32);
cb_op!(conv, __floatdisf, itof_i64_f32, (a: i64) -> f32);
cb_op!(conv, __floattisf, itof_i128_f32, (a: i128) -> f32);
cb_op!(conv, __floatsidf, itof_i32_f64, (a: i32) -> f64);
cb_op!(conv, __floatdidf, itof_i64_f64, (a: i64) -> f64);
cb_op!(conv, __floattidf, itof_i128_f64, (a: i128) -> f64);
#[cfg(f128_enabled)]
cb_op!(conv, __floatsitf, itof_i32_f128, (a: i32) -> f128);
#[cfg(f128_enabled)]
cb_op!(conv, __floatditf, itof_i64_f128, (a: i64) -> f128);
#[cfg(f128_enabled)]
cb_op!(conv, __floattitf, itof_i128_f128, (a: i128) -> f128);
cb_op!(conv, __floatunsisf, itof_u32_f32, (a: u32) -> f32);
cb_op!(conv, __floatundisf, itof_u64_f32, (a: u64) -> f32);
cb_op!(conv, __floatuntisf, itof_u128_f32, (a: u128) -> f32);
cb_op!(conv, __floatunsidf, itof_u32_f64, (a: u32) -> f64);
cb_op!(conv, __floatundidf, itof_u64_f64, (a: u64) -> f64);
cb_op!(conv, __floatuntidf, itof_u128_f64, (a: u128) -> f64);
#[cfg(f128_enabled)]
cb_op!(conv, __floatunsitf, itof_u32_f128, (a: u32) -> f128);
#[cfg(f128_enabled)]
cb_op!(conv, __floatunditf, itof_u64_f128, (a: u64) -> f128);
#[cfg(f128_enabled)]
cb_op!(conv, __floatuntitf, itof_u128_f128, (a: u128) -> f128);

/* int ops */

cb_op!(@int shift, __ashlsi3, ashl_u32, (a: u32, b: u32) -> u32);
cb_op!(@int shift, __ashldi3, ashl_u64, (a: u64, b: u32) -> u64);
cb_op!(@int shift, __ashlti3, ashl_u128, (a: u128, b: u32) -> u128);
cb_op!(@int shift, __ashrsi3, ashr_i32, (a: i32, b: u32) -> i32);
cb_op!(@int shift, __ashrdi3, ashr_i64, (a: i64, b: u32) -> i64);
cb_op!(@int shift, __ashrti3, ashr_i128, (a: i128, b: u32) -> i128);
cb_op!(@int shift, __lshrsi3, lshr_u32, (a: u32, b: u32) -> u32);
cb_op!(@int shift, __lshrdi3, lshr_u64, (a: u64, b: u32) -> u64);
cb_op!(@int shift, __lshrti3, lshr_u128, (a: u128, b: u32) -> u128);

cb_op!(@int leading_zeros, __clzsi2, leading_zeros_u32, (a: u32) -> usize);
cb_op!(@int leading_zeros, __clzdi2, leading_zeros_u64, (a: u64) -> usize);
cb_op!(@int leading_zeros, __clzti2, leading_zeros_u128, (a: u128) -> usize);
cb_op!(@int trailing_zeros, __ctzsi2, trailing_zeros_u32, (a: u32) -> usize);
cb_op!(@int trailing_zeros, __ctzdi2, trailing_zeros_u64, (a: u64) -> usize);
cb_op!(@int trailing_zeros, __ctzti2, trailing_zeros_u128, (a: u128) -> usize);
