//! This module defines built-in types.
//!
//! A peculiarity of built-in types is that they are always available and are
//! not associated with any particular crate.

use std::fmt;

use hir_expand::name::{self, Name};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Signedness {
    Signed,
    Unsigned,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum IntBitness {
    Xsize,
    X8,
    X16,
    X32,
    X64,
    X128,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum FloatBitness {
    X32,
    X64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BuiltinInt {
    pub signedness: Signedness,
    pub bitness: IntBitness,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BuiltinFloat {
    pub bitness: FloatBitness,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinType {
    Char,
    Bool,
    Str,
    Int(BuiltinInt),
    Float(BuiltinFloat),
}

impl BuiltinType {
    #[rustfmt::skip]
    pub const ALL: &'static [(Name, BuiltinType)] = &[
        (name::CHAR, BuiltinType::Char),
        (name::BOOL, BuiltinType::Bool),
        (name::STR,  BuiltinType::Str ),

        (name::ISIZE, BuiltinType::Int(BuiltinInt::ISIZE)),
        (name::I8,    BuiltinType::Int(BuiltinInt::I8)),
        (name::I16,   BuiltinType::Int(BuiltinInt::I16)),
        (name::I32,   BuiltinType::Int(BuiltinInt::I32)),
        (name::I64,   BuiltinType::Int(BuiltinInt::I64)),
        (name::I128,  BuiltinType::Int(BuiltinInt::I128)),

        (name::USIZE, BuiltinType::Int(BuiltinInt::USIZE)),
        (name::U8,    BuiltinType::Int(BuiltinInt::U8)),
        (name::U16,   BuiltinType::Int(BuiltinInt::U16)),
        (name::U32,   BuiltinType::Int(BuiltinInt::U32)),
        (name::U64,   BuiltinType::Int(BuiltinInt::U64)),
        (name::U128,  BuiltinType::Int(BuiltinInt::U128)),

        (name::F32, BuiltinType::Float(BuiltinFloat::F32)),
        (name::F64, BuiltinType::Float(BuiltinFloat::F64)),
    ];
}

impl fmt::Display for BuiltinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let type_name = match self {
            BuiltinType::Char => "char",
            BuiltinType::Bool => "bool",
            BuiltinType::Str => "str",
            BuiltinType::Int(BuiltinInt { signedness, bitness }) => match (signedness, bitness) {
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
            },
            BuiltinType::Float(BuiltinFloat { bitness }) => match bitness {
                FloatBitness::X32 => "f32",
                FloatBitness::X64 => "f64",
            },
        };
        f.write_str(type_name)
    }
}

#[rustfmt::skip]
impl BuiltinInt {
    pub const ISIZE: BuiltinInt = BuiltinInt { signedness: Signedness::Signed, bitness: IntBitness::Xsize   };
    pub const I8   : BuiltinInt = BuiltinInt { signedness: Signedness::Signed, bitness: IntBitness::X8      };
    pub const I16  : BuiltinInt = BuiltinInt { signedness: Signedness::Signed, bitness: IntBitness::X16     };
    pub const I32  : BuiltinInt = BuiltinInt { signedness: Signedness::Signed, bitness: IntBitness::X32     };
    pub const I64  : BuiltinInt = BuiltinInt { signedness: Signedness::Signed, bitness: IntBitness::X64     };
    pub const I128 : BuiltinInt = BuiltinInt { signedness: Signedness::Signed, bitness: IntBitness::X128    };

    pub const USIZE: BuiltinInt = BuiltinInt { signedness: Signedness::Unsigned, bitness: IntBitness::Xsize };
    pub const U8   : BuiltinInt = BuiltinInt { signedness: Signedness::Unsigned, bitness: IntBitness::X8    };
    pub const U16  : BuiltinInt = BuiltinInt { signedness: Signedness::Unsigned, bitness: IntBitness::X16   };
    pub const U32  : BuiltinInt = BuiltinInt { signedness: Signedness::Unsigned, bitness: IntBitness::X32   };
    pub const U64  : BuiltinInt = BuiltinInt { signedness: Signedness::Unsigned, bitness: IntBitness::X64   };
    pub const U128 : BuiltinInt = BuiltinInt { signedness: Signedness::Unsigned, bitness: IntBitness::X128  };


    pub fn from_suffix(suffix: &str) -> Option<BuiltinInt> {
        let res = match suffix {
            "isize" => Self::ISIZE,
            "i8"    => Self::I8,
            "i16"   => Self::I16,
            "i32"   => Self::I32,
            "i64"   => Self::I64,
            "i128"  => Self::I128,

            "usize" => Self::USIZE,
            "u8"    => Self::U8,
            "u16"   => Self::U16,
            "u32"   => Self::U32,
            "u64"   => Self::U64,
            "u128"  => Self::U128,

            _ => return None,
        };
        Some(res)
    }
}

#[rustfmt::skip]
impl BuiltinFloat {
    pub const F32: BuiltinFloat = BuiltinFloat { bitness: FloatBitness::X32 };
    pub const F64: BuiltinFloat = BuiltinFloat { bitness: FloatBitness::X64 };

    pub fn from_suffix(suffix: &str) -> Option<BuiltinFloat> {
        let res = match suffix {
            "f32" => BuiltinFloat::F32,
            "f64" => BuiltinFloat::F64,
            _ => return None,
        };
        Some(res)
    }
}
