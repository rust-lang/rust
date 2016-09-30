use core::mem;

pub mod add;

/// Trait for some basic operations on floats
pub trait Float: Sized + Copy {
    /// A uint of the same with as the float
    type Int;

    /// Returns the bitwidth of the float type
    fn bits() -> u32;

    /// Returns the bitwidth of the exponent
    fn exponent_bits() -> u32;

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

    /// Returns the sign bit of `self`
    fn sign(self) -> bool;

    /// Returns the exponent portion of `self`, shifted to the right
    fn exponent(self) -> Self::Int;

    /// Returns the significand portion of `self`
    fn significand(self) -> Self::Int;

    /// Returns (normalized exponent, normalized significand)
    fn normalize(significand: Self::Int) -> (i32, Self::Int);
}

impl Float for f32 {
    type Int = u32;
    fn bits() -> u32 {
        32
    }
    fn exponent_bits() -> u32 {
        8
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
    fn sign(self) -> bool {
        (self.repr() & 1 << Self::bits()) != 0
    }
    fn exponent(self) -> Self::Int {
        self.repr() >> Self::significand_bits()
            & ((1 << Self::exponent_bits()) - 1)
    }
    fn significand(self) -> Self::Int {
        self.repr() & ((1 << Self::significand_bits()) - 1)
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
    fn exponent_bits() -> u32 {
        11
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
    fn sign(self) -> bool {
        (self.repr() & 1 << Self::bits()) != 0
    }
    fn exponent(self) -> Self::Int {
        self.repr() >> Self::significand_bits()
            & ((1 << Self::exponent_bits()) - 1)
    }
    fn significand(self) -> Self::Int {
        self.repr() & ((1 << Self::significand_bits()) - 1)
    }
    fn normalize(significand: Self::Int) -> (i32, Self::Int) {
        let shift = significand.leading_zeros()
            .wrapping_sub((1u64 << Self::significand_bits()).leading_zeros());
        (1i32.wrapping_sub(shift as i32), significand << shift as Self::Int)
    }
}
