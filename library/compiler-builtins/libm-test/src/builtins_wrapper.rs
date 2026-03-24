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
