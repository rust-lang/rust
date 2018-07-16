// src: musl/src/fenv/fenv.c
/* Dummy functions for archs lacking fenv implementation */

pub const FE_UNDERFLOW: i32 = 0;
pub const FE_INEXACT: i32 = 0;

pub const FE_TONEAREST: i32 = 0;
pub const FE_TOWARDZERO: i32 = 0;

#[inline]
pub fn feclearexcept(_mask: i32) -> i32 {
    0
}

#[inline]
pub fn feraiseexcept(_mask: i32) -> i32 {
    0
}

#[inline]
pub fn fetestexcept(_mask: i32) -> i32 {
    0
}

#[inline]
pub fn fegetround() -> i32 {
    FE_TONEAREST
}

#[inline]
pub fn fesetround(_r: i32) -> i32 {
    0
}
