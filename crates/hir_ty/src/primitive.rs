//! Defines primitive types, which have a couple of peculiarities:
//!
//! * during type inference, they can be uncertain (ie, `let x = 92;`)
//! * they don't belong to any particular crate.

use std::fmt;

pub use hir_def::builtin_type::{BuiltinFloat, BuiltinInt, BuiltinUint};

/// Different signed int types.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IntTy {
    Isize,
    I8,
    I16,
    I32,
    I64,
    I128,
}

/// Different unsigned int types.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UintTy {
    Usize,
    U8,
    U16,
    U32,
    U64,
    U128,
}

impl fmt::Display for IntTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.ty_to_string())
    }
}

impl IntTy {
    pub fn ty_to_string(self) -> &'static str {
        match self {
            IntTy::Isize => "isize",
            IntTy::I8 => "i8",
            IntTy::I16 => "i16",
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
            IntTy::I128 => "i128",
        }
    }
}

impl fmt::Display for UintTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.ty_to_string())
    }
}

impl UintTy {
    pub fn ty_to_string(self) -> &'static str {
        match self {
            UintTy::Usize => "usize",
            UintTy::U8 => "u8",
            UintTy::U16 => "u16",
            UintTy::U32 => "u32",
            UintTy::U64 => "u64",
            UintTy::U128 => "u128",
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FloatTy {
    F32,
    F64,
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
    pub fn ty_to_string(self) -> &'static str {
        match self {
            FloatTy::F32 => "f32",
            FloatTy::F64 => "f64",
        }
    }
}

impl From<BuiltinInt> for IntTy {
    fn from(t: BuiltinInt) -> Self {
        match t {
            BuiltinInt::Isize => Self::Isize,
            BuiltinInt::I8 => Self::I8,
            BuiltinInt::I16 => Self::I16,
            BuiltinInt::I32 => Self::I32,
            BuiltinInt::I64 => Self::I64,
            BuiltinInt::I128 => Self::I128,
        }
    }
}

impl From<BuiltinUint> for UintTy {
    fn from(t: BuiltinUint) -> Self {
        match t {
            BuiltinUint::Usize => Self::Usize,
            BuiltinUint::U8 => Self::U8,
            BuiltinUint::U16 => Self::U16,
            BuiltinUint::U32 => Self::U32,
            BuiltinUint::U64 => Self::U64,
            BuiltinUint::U128 => Self::U128,
        }
    }
}

impl From<BuiltinFloat> for FloatTy {
    fn from(t: BuiltinFloat) -> Self {
        match t {
            BuiltinFloat::F32 => Self::F32,
            BuiltinFloat::F64 => Self::F64,
        }
    }
}
