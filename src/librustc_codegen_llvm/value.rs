pub use llvm::Value;

use llvm;

use std::fmt;
use std::hash::{Hash, Hasher};

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self as *const _ == other as *const _
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        (self as *const Self).hash(hasher);
    }
}


impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&llvm::build_string(|s| unsafe {
            llvm::LLVMRustWriteValueToString(self, s);
        }).expect("non-UTF8 value description from LLVM"))
    }
}
