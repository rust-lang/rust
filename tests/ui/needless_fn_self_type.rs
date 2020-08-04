#![warn(clippy::style, clippy::needless_fn_self_type)]

pub enum ValType {
    I32,
    I64,
    F32,
    F64,
}

impl ValType {
    pub fn bytes_bad(self: Self) -> usize {
        match self {
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    pub fn bytes_good(self) -> usize {
        match self {
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }
}

fn main() {}
