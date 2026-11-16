use std::fmt;

use crate::llvm::{self, Value};

impl<'ll> fmt::Debug for Value<'ll> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            &llvm::build_string(|s| unsafe {
                llvm::LLVMRustWriteValueToString(*self, s);
            })
            .expect("non-UTF8 value description from LLVM"),
        )
    }
}
