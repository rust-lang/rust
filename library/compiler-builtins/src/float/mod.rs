use core::mem;

pub mod add;

/// Trait for some basic operations on floats
pub trait Float: Sized {
    /// A uint of the same with as the float
    type Int;
    
    /// Returns the bitwidth of the float type
    fn bits() -> u32;

    /// Returns the bitwidth of the significand
    fn significand_bits() -> u32;

    /// Returns `self` transmuted to `Self::Int`
    fn repr(self) -> Self::Int;

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
    fn from_repr(a: Self::Int) -> Self {
        unsafe { mem::transmute(a) }
    }
    fn normalize(significand: Self::Int) -> (i32, Self::Int) {
        let shift = significand.leading_zeros()
            .wrapping_sub((1u64 << Self::significand_bits()).leading_zeros());
        (1i32.wrapping_sub(shift as i32), significand << shift as Self::Int)
    }
}
