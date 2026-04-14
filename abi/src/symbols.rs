pub type SymbolId = u32;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SymbolRefWire {
    pub tag: u32,
    pub ptr_or_id: u64,
    pub len: u64,
}

pub const SYMBOL_REF_TAG_ID: u32 = 0;
pub const SYMBOL_REF_TAG_STR: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolError {
    Success = 0,
    BadUtf8 = 1,
    BadPtr = 2,
    TooLong = 3,
    TableFull = 4,
}
