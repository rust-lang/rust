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
