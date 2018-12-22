use std::fmt;

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
        write!(f, "{}", self.ty_to_string())
    }
}

impl IntTy {
    pub fn ty_to_string(&self) -> &'static str {
        match *self {
            IntTy::Isize => "isize",
            IntTy::I8 => "i8",
            IntTy::I16 => "i16",
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
            IntTy::I128 => "i128",
        }
    }

    pub fn from_string(s: &str) -> Option<IntTy> {
        match s {
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

impl UintTy {
    pub fn ty_to_string(&self) -> &'static str {
        match *self {
            UintTy::Usize => "usize",
            UintTy::U8 => "u8",
            UintTy::U16 => "u16",
            UintTy::U32 => "u32",
            UintTy::U64 => "u64",
            UintTy::U128 => "u128",
        }
    }

    pub fn from_string(s: &str) -> Option<UintTy> {
        match s {
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

impl fmt::Display for UintTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.ty_to_string())
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

    pub fn from_string(s: &str) -> Option<FloatTy> {
        match s {
            "f32" => Some(FloatTy::F32),
            "f64" => Some(FloatTy::F64),
            _ => None,
        }
    }
}
