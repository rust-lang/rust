use std::fmt;

use crate::{Name, KnownName};

#[derive(Debug, Clone, Eq, PartialEq, Hash, Copy)]
pub enum UncertainIntTy {
    Unknown,
    Unsigned(UintTy),
    Signed(IntTy),
}

impl UncertainIntTy {
    pub(crate) fn from_type_name(name: &Name) -> Option<UncertainIntTy> {
        if let Some(ty) = IntTy::from_type_name(name) {
            Some(UncertainIntTy::Signed(ty))
        } else if let Some(ty) = UintTy::from_type_name(name) {
            Some(UncertainIntTy::Unsigned(ty))
        } else {
            None
        }
    }

    pub(crate) fn from_suffix(suffix: &str) -> Option<UncertainIntTy> {
        if let Some(ty) = IntTy::from_suffix(suffix) {
            Some(UncertainIntTy::Signed(ty))
        } else if let Some(ty) = UintTy::from_suffix(suffix) {
            Some(UncertainIntTy::Unsigned(ty))
        } else {
            None
        }
    }
}

impl fmt::Display for UncertainIntTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            UncertainIntTy::Unknown => write!(f, "{{integer}}"),
            UncertainIntTy::Signed(ty) => write!(f, "{}", ty),
            UncertainIntTy::Unsigned(ty) => write!(f, "{}", ty),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Copy)]
pub enum UncertainFloatTy {
    Unknown,
    Known(FloatTy),
}

impl UncertainFloatTy {
    pub(crate) fn from_type_name(name: &Name) -> Option<UncertainFloatTy> {
        FloatTy::from_type_name(name).map(UncertainFloatTy::Known)
    }

    pub(crate) fn from_suffix(suffix: &str) -> Option<UncertainFloatTy> {
        FloatTy::from_suffix(suffix).map(UncertainFloatTy::Known)
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

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Copy)]
pub enum IntTy {
    Isize,
    I8,
    I16,
    I32,
    I64,
    I128,
}

impl fmt::Debug for IntTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for IntTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match *self {
            IntTy::Isize => "isize",
            IntTy::I8 => "i8",
            IntTy::I16 => "i16",
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
            IntTy::I128 => "i128",
        };
        write!(f, "{}", s)
    }
}

impl IntTy {
    fn from_type_name(name: &Name) -> Option<IntTy> {
        match name.as_known_name()? {
            KnownName::Isize => Some(IntTy::Isize),
            KnownName::I8 => Some(IntTy::I8),
            KnownName::I16 => Some(IntTy::I16),
            KnownName::I32 => Some(IntTy::I32),
            KnownName::I64 => Some(IntTy::I64),
            KnownName::I128 => Some(IntTy::I128),
            _ => None,
        }
    }

    fn from_suffix(suffix: &str) -> Option<IntTy> {
        match suffix {
            "isize" => Some(IntTy::Isize),
            "i8" => Some(IntTy::I8),
            "i16" => Some(IntTy::I16),
            "i32" => Some(IntTy::I32),
            "i64" => Some(IntTy::I64),
            "i128" => Some(IntTy::I128),
            _ => None,
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Copy)]
pub enum UintTy {
    Usize,
    U8,
    U16,
    U32,
    U64,
    U128,
}

impl fmt::Display for UintTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match *self {
            UintTy::Usize => "usize",
            UintTy::U8 => "u8",
            UintTy::U16 => "u16",
            UintTy::U32 => "u32",
            UintTy::U64 => "u64",
            UintTy::U128 => "u128",
        };
        write!(f, "{}", s)
    }
}

impl UintTy {
    fn from_type_name(name: &Name) -> Option<UintTy> {
        match name.as_known_name()? {
            KnownName::Usize => Some(UintTy::Usize),
            KnownName::U8 => Some(UintTy::U8),
            KnownName::U16 => Some(UintTy::U16),
            KnownName::U32 => Some(UintTy::U32),
            KnownName::U64 => Some(UintTy::U64),
            KnownName::U128 => Some(UintTy::U128),
            _ => None,
        }
    }

    fn from_suffix(suffix: &str) -> Option<UintTy> {
        match suffix {
            "usize" => Some(UintTy::Usize),
            "u8" => Some(UintTy::U8),
            "u16" => Some(UintTy::U16),
            "u32" => Some(UintTy::U32),
            "u64" => Some(UintTy::U64),
            "u128" => Some(UintTy::U128),
            _ => None,
        }
    }
}

impl fmt::Debug for UintTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Copy, PartialOrd, Ord)]
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

    fn from_type_name(name: &Name) -> Option<FloatTy> {
        match name.as_known_name()? {
            KnownName::F32 => Some(FloatTy::F32),
            KnownName::F64 => Some(FloatTy::F64),
            _ => None,
        }
    }

    fn from_suffix(suffix: &str) -> Option<FloatTy> {
        match suffix {
            "f32" => Some(FloatTy::F32),
            "f64" => Some(FloatTy::F64),
            _ => None,
        }
    }
}
