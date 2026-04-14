use abi::symbols::{SymbolRefWire, SYMBOL_REF_TAG_ID, SYMBOL_REF_TAG_STR};

pub trait IntoSymbolRef {
    fn to_wire(&self) -> SymbolRefWire;
}

// Specific impl for &str instead of generic AsRef<str> to avoid conflict
impl<'a> IntoSymbolRef for &'a str {
    fn to_wire(&self) -> SymbolRefWire {
        SymbolRefWire {
            tag: SYMBOL_REF_TAG_STR,
            ptr_or_id: self.as_ptr() as u64,
            len: self.len() as u64,
        }
    }
}

// Impl for u32 (SymbolId)
impl IntoSymbolRef for u32 {
    fn to_wire(&self) -> SymbolRefWire {
        SymbolRefWire {
            tag: SYMBOL_REF_TAG_ID,
            ptr_or_id: *self as u64,
            len: 0,
        }
    }
}

// Impl for u64 (Legacy/Lazy SymbolId)
impl IntoSymbolRef for u64 {
    fn to_wire(&self) -> SymbolRefWire {
        SymbolRefWire {
            tag: SYMBOL_REF_TAG_ID,
            ptr_or_id: *self,
            len: 0,
        }
    }
}
