use core::mem;

pub mod conv;
pub mod add;
pub mod pow;
pub mod sub;

/// Trait for some basic operations on floats
pub trait Float: Sized + Copy {
    /// A uint of the same with as the float
    type Int;

    /// Returns the bitwidth of the float type
    fn bits() -> u32;

    /// Returns the bitwidth of the significand
    fn significand_bits() -> u32;

    /// Returns the bitwidth of the exponent
    fn exponent_bits() -> u32 {
        Self::bits() - Self::significand_bits() - 1
    }
    /// Returns the maximum value of the exponent
    fn exponent_max() -> u32 {
        (1 << Self::exponent_bits()) - 1
    }

    /// Returns the exponent bias value
    fn exponent_bias() -> u32 {
        Self::exponent_max() >> 1
    }

    /// Returns a mask for the sign bit
    fn sign_mask() -> Self::Int;

    /// Returns a mask for the significand
    fn significand_mask() -> Self::Int;

    /// Returns a mask for the exponent
    fn exponent_mask() -> Self::Int;

    /// Returns `self` transmuted to `Self::Int`
    fn repr(self) -> Self::Int;

    #[cfg(test)]
    /// Checks if two floats have the same bit representation. *Except* for NaNs! NaN can be
    /// represented in multiple different ways. This method returns `true` if two NaNs are
    /// compared.
    fn eq_repr(self, rhs: Self) -> bool;

    /// Returns a `Self::Int` transmuted back to `Self`
    fn from_repr(a: Self::Int) -> Self;

    /// Constructs a `Self` from its parts. Inputs are treated as bits and shifted into position.
    fn from_parts(sign: bool, exponent: Self::Int, significand: Self::Int) -> Self;

    /// Returns (normalized exponent, normalized significand)
    fn normalize(significand: Self::Int) -> (i32, Self::Int);
}

// FIXME: Some of this can be removed if RFC Issue #1424 is resolved
//        https://github.com/rust-lang/rfcs/issues/1424
impl Float for f32 {
    type Int = u32;
    fn bits() -> u32 {
        32
    }
    fn significand_bits() -> u32 {
        23
    }
    fn sign_mask() -> Self::Int {
        1 << (Self::bits() - 1)
    }
    fn significand_mask() -> Self::Int {
        (1 << Self::significand_bits()) - 1
    }
    fn exponent_mask() -> Self::Int {
        !(Self::sign_mask() | Self::significand_mask())
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
    fn from_parts(sign: bool, exponent: Self::Int, significand: Self::Int) -> Self {
        Self::from_repr(((sign as Self::Int) << (Self::bits() - 1)) |
            ((exponent << Self::significand_bits()) & Self::exponent_mask()) |
            (significand & Self::significand_mask()))
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
    fn sign_mask() -> Self::Int {
        1 << (Self::bits() - 1)
    }
    fn significand_mask() -> Self::Int {
        (1 << Self::significand_bits()) - 1
    }
    fn exponent_mask() -> Self::Int {
        !(Self::sign_mask() | Self::significand_mask())
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
    fn from_parts(sign: bool, exponent: Self::Int, significand: Self::Int) -> Self {
        Self::from_repr(((sign as Self::Int) << (Self::bits() - 1)) |
            ((exponent << Self::significand_bits()) & Self::exponent_mask()) |
            (significand & Self::significand_mask()))
    }
    fn normalize(significand: Self::Int) -> (i32, Self::Int) {
        let shift = significand.leading_zeros()
            .wrapping_sub((1u64 << Self::significand_bits()).leading_zeros());
        (1i32.wrapping_sub(shift as i32), significand << shift as Self::Int)
    }
}
