//! This module defines built-in types.
//!
//! A peculiarity of built-in types is that they are always available and are
//! not associated with any particular crate.

use std::fmt;

use hir_expand::name::{name, AsName, Name};
/// Different signed int types.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BuiltinInt {
    Isize,
    I8,
    I16,
    I32,
    I64,
    I128,
}

/// Different unsigned int types.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BuiltinUint {
    Usize,
    U8,
    U16,
    U32,
    U64,
    U128,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BuiltinFloat {
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinType {
    Char,
    Bool,
    Str,
    Int(BuiltinInt),
    Uint(BuiltinUint),
    Float(BuiltinFloat),
}

impl BuiltinType {
    #[rustfmt::skip]
    pub const ALL: &'static [(Name, BuiltinType)] = &[
        (name![char], BuiltinType::Char),
        (name![bool], BuiltinType::Bool),
        (name![str],  BuiltinType::Str),

        (name![isize], BuiltinType::Int(BuiltinInt::Isize)),
        (name![i8],    BuiltinType::Int(BuiltinInt::I8)),
        (name![i16],   BuiltinType::Int(BuiltinInt::I16)),
        (name![i32],   BuiltinType::Int(BuiltinInt::I32)),
        (name![i64],   BuiltinType::Int(BuiltinInt::I64)),
        (name![i128],  BuiltinType::Int(BuiltinInt::I128)),

        (name![usize], BuiltinType::Uint(BuiltinUint::Usize)),
        (name![u8],    BuiltinType::Uint(BuiltinUint::U8)),
        (name![u16],   BuiltinType::Uint(BuiltinUint::U16)),
        (name![u32],   BuiltinType::Uint(BuiltinUint::U32)),
        (name![u64],   BuiltinType::Uint(BuiltinUint::U64)),
        (name![u128],  BuiltinType::Uint(BuiltinUint::U128)),

        (name![f32], BuiltinType::Float(BuiltinFloat::F32)),
        (name![f64], BuiltinType::Float(BuiltinFloat::F64)),
    ];

    pub fn by_name(name: &Name) -> Option<Self> {
        Self::ALL.iter().find_map(|(n, ty)| if n == name { Some(*ty) } else { None })
    }
}

impl AsName for BuiltinType {
    fn as_name(&self) -> Name {
        match self {
            BuiltinType::Char => name![char],
            BuiltinType::Bool => name![bool],
            BuiltinType::Str => name![str],
            BuiltinType::Int(it) => match it {
                BuiltinInt::Isize => name![isize],
                BuiltinInt::I8 => name![i8],
                BuiltinInt::I16 => name![i16],
                BuiltinInt::I32 => name![i32],
                BuiltinInt::I64 => name![i64],
                BuiltinInt::I128 => name![i128],
            },
            BuiltinType::Uint(it) => match it {
                BuiltinUint::Usize => name![usize],
                BuiltinUint::U8 => name![u8],
                BuiltinUint::U16 => name![u16],
                BuiltinUint::U32 => name![u32],
                BuiltinUint::U64 => name![u64],
                BuiltinUint::U128 => name![u128],
            },
            BuiltinType::Float(it) => match it {
                BuiltinFloat::F32 => name![f32],
                BuiltinFloat::F64 => name![f64],
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
    pub fn from_suffix(suffix: &str) -> Option<BuiltinInt> {
        let res = match suffix {
            "isize" => Self::Isize,
            "i8"    => Self::I8,
            "i16"   => Self::I16,
            "i32"   => Self::I32,
            "i64"   => Self::I64,
            "i128"  => Self::I128,

            _ => return None,
        };
        Some(res)
    }
}

#[rustfmt::skip]
impl BuiltinUint {
    pub fn from_suffix(suffix: &str) -> Option<BuiltinUint> {
        let res = match suffix {
            "usize" => Self::Usize,
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
    pub fn from_suffix(suffix: &str) -> Option<BuiltinFloat> {
        let res = match suffix {
            "f32" => BuiltinFloat::F32,
            "f64" => BuiltinFloat::F64,
            _ => return None,
        };
        Some(res)
    }
}

impl fmt::Display for BuiltinInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BuiltinInt::Isize => "isize",
            BuiltinInt::I8 => "i8",
            BuiltinInt::I16 => "i16",
            BuiltinInt::I32 => "i32",
            BuiltinInt::I64 => "i64",
            BuiltinInt::I128 => "i128",
        })
    }
}

impl fmt::Display for BuiltinUint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BuiltinUint::Usize => "usize",
            BuiltinUint::U8 => "u8",
            BuiltinUint::U16 => "u16",
            BuiltinUint::U32 => "u32",
            BuiltinUint::U64 => "u64",
            BuiltinUint::U128 => "u128",
        })
    }
}

impl fmt::Display for BuiltinFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BuiltinFloat::F32 => "f32",
            BuiltinFloat::F64 => "f64",
        })
    }
}
