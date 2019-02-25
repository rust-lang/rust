pub use crate::llvm::Value;

use crate::llvm;

use std::fmt;
use std::hash::{Hash, Hasher};
use std::ptr;

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        (self as *const Self).hash(hasher);
    }
}


impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&llvm::build_string(|s| unsafe {
            llvm::LLVMRustWriteValueToString(self, s);
        }).expect("non-UTF8 value description from LLVM"))
    }
}
