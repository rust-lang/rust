use std::hash::{Hash, Hasher};
use std::string::FromUtf8Error;
use std::{fmt, ptr};

fn build_string(f: impl FnOnce(&crate::RustString)) -> Result<String, FromUtf8Error> {
    String::from_utf8(crate::RustString::build_byte_buffer(f))
}

impl PartialEq for crate::ffi::Metadata {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl Eq for crate::ffi::Metadata {}

impl Hash for crate::ffi::Metadata {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        (self as *const Self).hash(hasher);
    }
}

impl fmt::Debug for crate::ffi::Metadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self as *const Self).fmt(f)
    }
}

impl PartialEq for crate::ffi::Type {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl Eq for crate::ffi::Type {}

impl Hash for crate::ffi::Type {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ptr::hash(self, state);
    }
}

impl fmt::Debug for crate::ffi::Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            &build_string(|s| unsafe {
                crate::ffi::LLVMRustWriteTypeToString(self, s);
            })
            .expect("non-UTF8 type description from LLVM"),
        )
    }
}

impl PartialEq for crate::ffi::Value {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl Eq for crate::ffi::Value {}

impl Hash for crate::ffi::Value {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        (self as *const Self).hash(hasher);
    }
}

impl fmt::Debug for crate::ffi::Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            &build_string(|s| unsafe {
                crate::ffi::LLVMRustWriteValueToString(self, s);
            })
            .expect("non-UTF8 value description from LLVM"),
        )
    }
}
