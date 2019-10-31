//! FIXME: write short doc here

use std::fmt;

pub use hir_def::builtin_type::{FloatBitness, IntBitness, Signedness};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum UncertainIntTy {
    Unknown,
    Known(IntTy),
}

impl From<IntTy> for UncertainIntTy {
    fn from(ty: IntTy) -> Self {
        UncertainIntTy::Known(ty)
    }
}

impl fmt::Display for UncertainIntTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            UncertainIntTy::Unknown => write!(f, "{{integer}}"),
            UncertainIntTy::Known(ty) => write!(f, "{}", ty),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum UncertainFloatTy {
    Unknown,
    Known(FloatTy),
}

impl From<FloatTy> for UncertainFloatTy {
    fn from(ty: FloatTy) -> Self {
        UncertainFloatTy::Known(ty)
    }
}

impl fmt::Display for UncertainFloatTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            UncertainFloatTy::Unknown => write!(f, "{{float}}"),
            UncertainFloatTy::Known(ty) => write!(f, "{}", ty),
        }
    }
}

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

    pub(crate) fn ty_to_string(self) -> &'static str {
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

    pub(crate) fn from_suffix(suffix: &str) -> Option<IntTy> {
        match suffix {
            "isize" => Some(IntTy::isize()),
            "i8" => Some(IntTy::i8()),
            "i16" => Some(IntTy::i16()),
            "i32" => Some(IntTy::i32()),
            "i64" => Some(IntTy::i64()),
            "i128" => Some(IntTy::i128()),
            "usize" => Some(IntTy::usize()),
            "u8" => Some(IntTy::u8()),
            "u16" => Some(IntTy::u16()),
            "u32" => Some(IntTy::u32()),
            "u64" => Some(IntTy::u64()),
            "u128" => Some(IntTy::u128()),
            _ => None,
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

    pub(crate) fn ty_to_string(self) -> &'static str {
        match self.bitness {
            FloatBitness::X32 => "f32",
            FloatBitness::X64 => "f64",
        }
    }

    pub(crate) fn from_suffix(suffix: &str) -> Option<FloatTy> {
        match suffix {
            "f32" => Some(FloatTy::f32()),
            "f64" => Some(FloatTy::f64()),
            _ => None,
        }
    }
}
