//! Wrappers around compiler-builtins functions.
//!
//! Functions from compiler-builtins have a different naming scheme from libm and often a different
//! ABI (doesn't work with libm-test traits because that changes the type signature). Wrap these
//! to make them a bit more similar to the rest of the libm functions.

macro_rules! binop {
    ($op:ident, $ty:ty, $sfx:ident) => {
        paste::paste! {
            pub fn [< $op $ty >](a: $ty, b: $ty) -> $ty {
                compiler_builtins::float::$op::[< __ $op $sfx >](a, b)
            }
        }
    };
}

#[cfg(f16_enabled)]
binop!(add, f16, hf3);
#[cfg(f16_enabled)]
binop!(sub, f16, hf3);
#[cfg(f16_enabled)]
binop!(mul, f16, hf3);
binop!(add, f32, sf3);
binop!(sub, f32, sf3);
binop!(mul, f32, sf3);
binop!(div, f32, sf3);
binop!(add, f64, df3);
binop!(sub, f64, df3);
binop!(mul, f64, df3);
binop!(div, f64, df3);
#[cfg(f128_enabled)]
binop!(add, f128, tf3);
#[cfg(f128_enabled)]
binop!(sub, f128, tf3);
#[cfg(f128_enabled)]
binop!(mul, f128, tf3);
#[cfg(f128_enabled)]
binop!(div, f128, tf3);

pub fn powif32(a: f32, b: i32) -> f32 {
    compiler_builtins::float::pow::__powisf2(a, b)
}

pub fn powif64(a: f64, b: i32) -> f64 {
    compiler_builtins::float::pow::__powidf2(a, b)
}

#[cfg(f128_enabled)]
pub fn powif128(a: f128, b: i32) -> f128 {
    compiler_builtins::float::pow::__powitf2(a, b)
}
