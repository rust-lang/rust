//! This module defines built-in types.
//!
//! A peculiarity of built-in types is that they are always available and are
//! not associated with any particular crate.

use std::fmt;

use hir_expand::name::{name, AsName, Name};

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
        (name![char], BuiltinType::Char),
        (name![bool], BuiltinType::Bool),
        (name![str],  BuiltinType::Str),

        (name![isize], BuiltinType::Int(BuiltinInt::ISIZE)),
        (name![i8],    BuiltinType::Int(BuiltinInt::I8)),
        (name![i16],   BuiltinType::Int(BuiltinInt::I16)),
        (name![i32],   BuiltinType::Int(BuiltinInt::I32)),
        (name![i64],   BuiltinType::Int(BuiltinInt::I64)),
        (name![i128],  BuiltinType::Int(BuiltinInt::I128)),

        (name![usize], BuiltinType::Int(BuiltinInt::USIZE)),
        (name![u8],    BuiltinType::Int(BuiltinInt::U8)),
        (name![u16],   BuiltinType::Int(BuiltinInt::U16)),
        (name![u32],   BuiltinType::Int(BuiltinInt::U32)),
        (name![u64],   BuiltinType::Int(BuiltinInt::U64)),
        (name![u128],  BuiltinType::Int(BuiltinInt::U128)),

        (name![f32], BuiltinType::Float(BuiltinFloat::F32)),
        (name![f64], BuiltinType::Float(BuiltinFloat::F64)),
    ];
}

impl AsName for BuiltinType {
    fn as_name(&self) -> Name {
        match self {
            BuiltinType::Char => name![char],
            BuiltinType::Bool => name![bool],
            BuiltinType::Str => name![str],
            BuiltinType::Int(BuiltinInt { signedness, bitness }) => match (signedness, bitness) {
                (Signedness::Signed, IntBitness::Xsize) => name![isize],
                (Signedness::Signed, IntBitness::X8) => name![i8],
                (Signedness::Signed, IntBitness::X16) => name![i16],
                (Signedness::Signed, IntBitness::X32) => name![i32],
                (Signedness::Signed, IntBitness::X64) => name![i64],
                (Signedness::Signed, IntBitness::X128) => name![i128],

                (Signedness::Unsigned, IntBitness::Xsize) => name![usize],
                (Signedness::Unsigned, IntBitness::X8) => name![u8],
                (Signedness::Unsigned, IntBitness::X16) => name![u16],
                (Signedness::Unsigned, IntBitness::X32) => name![u32],
                (Signedness::Unsigned, IntBitness::X64) => name![u64],
                (Signedness::Unsigned, IntBitness::X128) => name![u128],
            },
            BuiltinType::Float(BuiltinFloat { bitness }) => match bitness {
                FloatBitness::X32 => name![f32],
                FloatBitness::X64 => name![f64],
            },
        }
    }
}

impl fmt::Display for BuiltinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let type_name = self.as_name();
        type_name.fmt(f)
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
