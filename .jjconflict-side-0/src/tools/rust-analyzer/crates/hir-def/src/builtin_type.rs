//! This module defines built-in types.
//!
//! A peculiarity of built-in types is that they are always available and are
//! not associated with any particular crate.

use std::fmt;

use hir_expand::name::{AsName, Name};
use intern::{Symbol, sym};
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
    F16,
    F32,
    F64,
    F128,
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
    pub fn all_builtin_types() -> [(Name, BuiltinType); 19] {
        [
            (Name::new_symbol_root(sym::char), BuiltinType::Char),
            (Name::new_symbol_root(sym::bool), BuiltinType::Bool),
            (Name::new_symbol_root(sym::str),  BuiltinType::Str),

            (Name::new_symbol_root(sym::isize), BuiltinType::Int(BuiltinInt::Isize)),
            (Name::new_symbol_root(sym::i8),    BuiltinType::Int(BuiltinInt::I8)),
            (Name::new_symbol_root(sym::i16),   BuiltinType::Int(BuiltinInt::I16)),
            (Name::new_symbol_root(sym::i32),   BuiltinType::Int(BuiltinInt::I32)),
            (Name::new_symbol_root(sym::i64),   BuiltinType::Int(BuiltinInt::I64)),
            (Name::new_symbol_root(sym::i128),  BuiltinType::Int(BuiltinInt::I128)),

            (Name::new_symbol_root(sym::usize), BuiltinType::Uint(BuiltinUint::Usize)),
            (Name::new_symbol_root(sym::u8),    BuiltinType::Uint(BuiltinUint::U8)),
            (Name::new_symbol_root(sym::u16),   BuiltinType::Uint(BuiltinUint::U16)),
            (Name::new_symbol_root(sym::u32),   BuiltinType::Uint(BuiltinUint::U32)),
            (Name::new_symbol_root(sym::u64),   BuiltinType::Uint(BuiltinUint::U64)),
            (Name::new_symbol_root(sym::u128),  BuiltinType::Uint(BuiltinUint::U128)),

            (Name::new_symbol_root(sym::f16), BuiltinType::Float(BuiltinFloat::F16)),
            (Name::new_symbol_root(sym::f32), BuiltinType::Float(BuiltinFloat::F32)),
            (Name::new_symbol_root(sym::f64), BuiltinType::Float(BuiltinFloat::F64)),
            (Name::new_symbol_root(sym::f128), BuiltinType::Float(BuiltinFloat::F128)),
        ]
    }

    pub fn by_name(name: &Name) -> Option<Self> {
        Self::all_builtin_types()
            .iter()
            .find_map(|(n, ty)| if n == name { Some(*ty) } else { None })
    }
}

impl AsName for BuiltinType {
    fn as_name(&self) -> Name {
        match self {
            BuiltinType::Char => Name::new_symbol_root(sym::char),
            BuiltinType::Bool => Name::new_symbol_root(sym::bool),
            BuiltinType::Str => Name::new_symbol_root(sym::str),
            BuiltinType::Int(it) => match it {
                BuiltinInt::Isize => Name::new_symbol_root(sym::isize),
                BuiltinInt::I8 => Name::new_symbol_root(sym::i8),
                BuiltinInt::I16 => Name::new_symbol_root(sym::i16),
                BuiltinInt::I32 => Name::new_symbol_root(sym::i32),
                BuiltinInt::I64 => Name::new_symbol_root(sym::i64),
                BuiltinInt::I128 => Name::new_symbol_root(sym::i128),
            },
            BuiltinType::Uint(it) => match it {
                BuiltinUint::Usize => Name::new_symbol_root(sym::usize),
                BuiltinUint::U8 => Name::new_symbol_root(sym::u8),
                BuiltinUint::U16 => Name::new_symbol_root(sym::u16),
                BuiltinUint::U32 => Name::new_symbol_root(sym::u32),
                BuiltinUint::U64 => Name::new_symbol_root(sym::u64),
                BuiltinUint::U128 => Name::new_symbol_root(sym::u128),
            },
            BuiltinType::Float(it) => match it {
                BuiltinFloat::F16 => Name::new_symbol_root(sym::f16),
                BuiltinFloat::F32 => Name::new_symbol_root(sym::f32),
                BuiltinFloat::F64 => Name::new_symbol_root(sym::f64),
                BuiltinFloat::F128 => Name::new_symbol_root(sym::f128),
            },
        }
    }
}

impl fmt::Display for BuiltinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuiltinType::Char => f.write_str("char"),
            BuiltinType::Bool => f.write_str("bool"),
            BuiltinType::Str => f.write_str("str"),
            BuiltinType::Int(it) => it.fmt(f),
            BuiltinType::Uint(it) => it.fmt(f),
            BuiltinType::Float(it) => it.fmt(f),
        }
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
    pub fn from_suffix_sym(suffix: &Symbol) -> Option<BuiltinInt> {
        let res = match suffix {
            s if *s == sym::isize => Self::Isize,
            s if *s == sym::i8    => Self::I8,
            s if *s == sym::i16   => Self::I16,
            s if *s == sym::i32   => Self::I32,
            s if *s == sym::i64   => Self::I64,
            s if *s == sym::i128  => Self::I128,
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
    pub fn from_suffix_sym(suffix: &Symbol) -> Option<BuiltinUint> {
        let res = match suffix {
            s if *s == sym::usize => Self::Usize,
            s if *s == sym::u8    => Self::U8,
            s if *s == sym::u16   => Self::U16,
            s if *s == sym::u32   => Self::U32,
            s if *s == sym::u64   => Self::U64,
            s if *s == sym::u128  => Self::U128,

            _ => return None,
        };
        Some(res)
    }
}

#[rustfmt::skip]
impl BuiltinFloat {
    pub fn from_suffix(suffix: &str) -> Option<BuiltinFloat> {
        let res = match suffix {
            "f16" => BuiltinFloat::F16,
            "f32" => BuiltinFloat::F32,
            "f64" => BuiltinFloat::F64,
            "f128" => BuiltinFloat::F128,
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
            BuiltinFloat::F16 => "f16",
            BuiltinFloat::F32 => "f32",
            BuiltinFloat::F64 => "f64",
            BuiltinFloat::F128 => "f128",
        })
    }
}
