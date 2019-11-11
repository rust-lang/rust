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
pub enum BuiltinType {
    Char,
    Bool,
    Str,
    Int { signedness: Signedness, bitness: IntBitness },
    Float { bitness: FloatBitness },
}

impl BuiltinType {
    #[rustfmt::skip]
    pub const ALL: &'static [(Name, BuiltinType)] = &[
        (name::CHAR, BuiltinType::Char),
        (name::BOOL, BuiltinType::Bool),
        (name::STR,  BuiltinType::Str ),

        (name::ISIZE, BuiltinType::Int { signedness: Signedness::Signed,   bitness: IntBitness::Xsize }),
        (name::I8,    BuiltinType::Int { signedness: Signedness::Signed,   bitness: IntBitness::X8    }),
        (name::I16,   BuiltinType::Int { signedness: Signedness::Signed,   bitness: IntBitness::X16   }),
        (name::I32,   BuiltinType::Int { signedness: Signedness::Signed,   bitness: IntBitness::X32   }),
        (name::I64,   BuiltinType::Int { signedness: Signedness::Signed,   bitness: IntBitness::X64   }),
        (name::I128,  BuiltinType::Int { signedness: Signedness::Signed,   bitness: IntBitness::X128  }),

        (name::USIZE, BuiltinType::Int { signedness: Signedness::Unsigned, bitness: IntBitness::Xsize }),
        (name::U8,    BuiltinType::Int { signedness: Signedness::Unsigned, bitness: IntBitness::X8    }),
        (name::U16,   BuiltinType::Int { signedness: Signedness::Unsigned, bitness: IntBitness::X16   }),
        (name::U32,   BuiltinType::Int { signedness: Signedness::Unsigned, bitness: IntBitness::X32   }),
        (name::U64,   BuiltinType::Int { signedness: Signedness::Unsigned, bitness: IntBitness::X64   }),
        (name::U128,  BuiltinType::Int { signedness: Signedness::Unsigned, bitness: IntBitness::X128  }),

        (name::F32, BuiltinType::Float { bitness: FloatBitness::X32 }),
        (name::F64, BuiltinType::Float { bitness: FloatBitness::X64 }),
    ];
}

impl fmt::Display for BuiltinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let type_name = match self {
            BuiltinType::Char => "char",
            BuiltinType::Bool => "bool",
            BuiltinType::Str => "str",
            BuiltinType::Int { signedness, bitness } => match (signedness, bitness) {
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
            BuiltinType::Float { bitness } => match bitness {
                FloatBitness::X32 => "f32",
                FloatBitness::X64 => "f64",
            },
        };
        f.write_str(type_name)
    }
}
