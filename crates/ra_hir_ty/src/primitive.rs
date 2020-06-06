//! Defines primitive types, which have a couple of peculiarities:
//!
//! * during type inference, they can be uncertain (ie, `let x = 92;`)
//! * they don't belong to any particular crate.

use std::fmt;

pub use hir_def::builtin_type::{BuiltinFloat, BuiltinInt, FloatBitness, IntBitness, Signedness};

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct IntTy {
    pub signedness: Signedness,
    pub bitness: IntBitness,
}

impl fmt::Debug for IntTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for IntTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.ty_to_string())
    }
}

impl IntTy {
    pub fn isize() -> IntTy {
        IntTy { signedness: Signedness::Signed, bitness: IntBitness::Xsize }
    }

    pub fn i8() -> IntTy {
        IntTy { signedness: Signedness::Signed, bitness: IntBitness::X8 }
    }

    pub fn i16() -> IntTy {
        IntTy { signedness: Signedness::Signed, bitness: IntBitness::X16 }
    }

    pub fn i32() -> IntTy {
        IntTy { signedness: Signedness::Signed, bitness: IntBitness::X32 }
    }

    pub fn i64() -> IntTy {
        IntTy { signedness: Signedness::Signed, bitness: IntBitness::X64 }
    }

    pub fn i128() -> IntTy {
        IntTy { signedness: Signedness::Signed, bitness: IntBitness::X128 }
    }

    pub fn usize() -> IntTy {
        IntTy { signedness: Signedness::Unsigned, bitness: IntBitness::Xsize }
    }

    pub fn u8() -> IntTy {
        IntTy { signedness: Signedness::Unsigned, bitness: IntBitness::X8 }
    }

    pub fn u16() -> IntTy {
        IntTy { signedness: Signedness::Unsigned, bitness: IntBitness::X16 }
    }

    pub fn u32() -> IntTy {
        IntTy { signedness: Signedness::Unsigned, bitness: IntBitness::X32 }
    }

    pub fn u64() -> IntTy {
        IntTy { signedness: Signedness::Unsigned, bitness: IntBitness::X64 }
    }

    pub fn u128() -> IntTy {
        IntTy { signedness: Signedness::Unsigned, bitness: IntBitness::X128 }
    }

    pub fn ty_to_string(self) -> &'static str {
        match (self.signedness, self.bitness) {
            (Signedness::Signed, IntBitness::Xsize) => "isize",
            (Signedness::Signed, IntBitness::X8) => "i8",
            (Signedness::Signed, IntBitness::X16) => "i16",
            (Signedness::Signed, IntBitness::X32) => "i32",
            (Signedness::Signed, IntBitness::X64) => "i64",
            (Signedness::Signed, IntBitness::X128) => "i128",
            (Signedness::Unsigned, IntBitness::Xsize) => "usize",
            (Signedness::Unsigned, IntBitness::X8) => "u8",
            (Signedness::Unsigned, IntBitness::X16) => "u16",
            (Signedness::Unsigned, IntBitness::X32) => "u32",
            (Signedness::Unsigned, IntBitness::X64) => "u64",
            (Signedness::Unsigned, IntBitness::X128) => "u128",
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct FloatTy {
    pub bitness: FloatBitness,
}

impl fmt::Debug for FloatTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for FloatTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.ty_to_string())
    }
}

impl FloatTy {
    pub fn f32() -> FloatTy {
        FloatTy { bitness: FloatBitness::X32 }
    }

    pub fn f64() -> FloatTy {
        FloatTy { bitness: FloatBitness::X64 }
    }

    pub fn ty_to_string(self) -> &'static str {
        match self.bitness {
            FloatBitness::X32 => "f32",
            FloatBitness::X64 => "f64",
        }
    }
}

impl From<BuiltinInt> for IntTy {
    fn from(t: BuiltinInt) -> Self {
        IntTy { signedness: t.signedness, bitness: t.bitness }
    }
}

impl From<BuiltinFloat> for FloatTy {
    fn from(t: BuiltinFloat) -> Self {
        FloatTy { bitness: t.bitness }
    }
}
