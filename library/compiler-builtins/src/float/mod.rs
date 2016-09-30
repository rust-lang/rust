use core::mem;
use core::fmt;

pub mod add;
pub mod pow;

/// Trait for some basic operations on floats
pub trait Float: Sized + Copy {
    /// A uint of the same with as the float
    type Int;

    /// Returns the bitwidth of the float type
    fn bits() -> u32;

    /// Returns the bitwidth of the significand
    fn significand_bits() -> u32;

    /// Returns `self` transmuted to `Self::Int`
    fn repr(self) -> Self::Int;

    #[cfg(test)]
    /// Checks if two floats have the same bit representation. *Except* for NaNs! NaN can be
    /// represented in multiple different ways. This methods returns `true` if two NaNs are
    /// compared.
    fn eq_repr(self, rhs: Self) -> bool;

    /// Returns a `Self::Int` transmuted back to `Self`
    fn from_repr(a: Self::Int) -> Self;

    /// Returns (normalized exponent, normalized significand)
    fn normalize(significand: Self::Int) -> (i32, Self::Int);
}

impl Float for f32 {
    type Int = u32;
    fn bits() -> u32 {
        32
    }
    fn significand_bits() -> u32 {
        23
    }
    fn repr(self) -> Self::Int {
        unsafe { mem::transmute(self) }
    }
    #[cfg(test)]
    fn eq_repr(self, rhs: Self) -> bool {
        if self.is_nan() && rhs.is_nan() {
            true
        } else {
            self.repr() == rhs.repr()
        }
    }
    fn from_repr(a: Self::Int) -> Self {
        unsafe { mem::transmute(a) }
    }
    fn normalize(significand: Self::Int) -> (i32, Self::Int) {
        let shift = significand.leading_zeros()
            .wrapping_sub((1u32 << Self::significand_bits()).leading_zeros());
        (1i32.wrapping_sub(shift as i32), significand << shift as Self::Int)
    }
}
impl Float for f64 {
    type Int = u64;
    fn bits() -> u32 {
        64
    }
    fn significand_bits() -> u32 {
        52
    }
    fn repr(self) -> Self::Int {
        unsafe { mem::transmute(self) }
    }
    #[cfg(test)]
    fn eq_repr(self, rhs: Self) -> bool {
        if self.is_nan() && rhs.is_nan() {
            true
        } else {
            self.repr() == rhs.repr()
        }
    }
    fn from_repr(a: Self::Int) -> Self {
        unsafe { mem::transmute(a) }
    }
    fn normalize(significand: Self::Int) -> (i32, Self::Int) {
        let shift = significand.leading_zeros()
            .wrapping_sub((1u64 << Self::significand_bits()).leading_zeros());
        (1i32.wrapping_sub(shift as i32), significand << shift as Self::Int)
    }
}

// TODO: Move this to F32/F64 in qc.rs
#[cfg(test)]
#[derive(Copy, Clone)]
pub struct FRepr<F>(F);

#[cfg(test)]
impl<F: Float> PartialEq for FRepr<F> {
    fn eq(&self, other: &FRepr<F>) -> bool {
        // NOTE(cfg) for some reason, on hard float targets, our implementation doesn't
        // match the output of its gcc_s counterpart. Until we investigate further, we'll
        // just avoid testing against gcc_s on those targets. Do note that our
        // implementation matches the output of the FPU instruction on *hard* float targets
        // and matches its gcc_s counterpart on *soft* float targets.
        if cfg!(gnueabihf) {
            return true
        }
        self.0.eq_repr(other.0)
    }
}

#[cfg(test)]
impl<F: fmt::Debug> fmt::Debug for FRepr<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}
