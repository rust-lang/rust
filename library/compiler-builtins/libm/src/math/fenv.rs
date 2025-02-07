// src: musl/src/fenv/fenv.c
/* Dummy functions for archs lacking fenv implementation */

pub(crate) const FE_UNDERFLOW: i32 = 0;
pub(crate) const FE_INEXACT: i32 = 0;

pub(crate) const FE_TONEAREST: i32 = 0;
pub(crate) const FE_DOWNWARD: i32 = 1;
pub(crate) const FE_UPWARD: i32 = 2;
pub(crate) const FE_TOWARDZERO: i32 = 3;

#[inline]
pub(crate) fn feclearexcept(_mask: i32) -> i32 {
    0
}

#[inline]
pub(crate) fn feraiseexcept(_mask: i32) -> i32 {
    0
}

#[inline]
pub(crate) fn fetestexcept(_mask: i32) -> i32 {
    0
}

#[inline]
pub(crate) fn fegetround() -> i32 {
    FE_TONEAREST
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Rounding {
    Nearest = FE_TONEAREST as isize,
    Downward = FE_DOWNWARD as isize,
    Upward = FE_UPWARD as isize,
    ToZero = FE_TOWARDZERO as isize,
}

impl Rounding {
    pub(crate) fn get() -> Self {
        match fegetround() {
            x if x == FE_DOWNWARD => Self::Downward,
            x if x == FE_UPWARD => Self::Upward,
            x if x == FE_TOWARDZERO => Self::ToZero,
            _ => Self::Nearest,
        }
    }
}
