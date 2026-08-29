use std::cell::RefCell;
use std::ffi::c_char;
use std::ptr;
use std::string::FromUtf8Error;

use libc::size_t;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Blame {
    pub file_idx: u32,
    pub line_no: u32,
}

impl Blame {
    pub const NONE: Self = Self { file_idx: 0, line_no: 0 };

    #[inline]
    pub fn is_none(self) -> bool {
        self.line_no == 0
    }

    #[inline]
    pub fn is_some(self) -> bool {
        self.line_no != 0
    }
}

unsafe extern "C" {
    pub(crate) type Opaque;
    /// Opaque type that allows C++ code to write bytes to a Rust-side buffer,
    /// in conjunction with `RawRustStringOstream`. Use this as `&RustString`
    /// (Rust) and `RustStringRef` (C++) in FFI signatures.
    pub(crate) type RustString;

    pub(crate) fn LLVMRustSpecialCaseListCreate(
        Paths: *const *const c_char,
        NumPaths: size_t,
        ErrorMsg: &RustString,
    ) -> *mut Opaque;

    pub(crate) fn LLVMRustSpecialCaseListDestroy(List: *mut Opaque);
    pub(crate) fn LLVMRustSpecialCaseListInSectionBlame(
        List: *const Opaque,
        Mask: u32,
        Section: *const c_char,
        Prefix: *const c_char,
        Query: *const c_char,
        OutNoSan: *mut Blame,
        OutSan: *mut Blame,
    );
}

/// Underlying implementation of [`RustString`].
///
/// Having two separate types makes it possible to use the opaque [`RustString`]
/// in FFI signatures without `improper_ctypes` warnings. This is a workaround
/// for the fact that there is no way to opt out of `improper_ctypes` when
/// _declaring_ a type (as opposed to using that type).
#[derive(Default)]
struct RustStringInner {
    bytes: RefCell<Vec<u8>>,
}

impl RustStringInner {
    fn as_opaque(&self) -> &RustString {
        let ptr: *const RustStringInner = ptr::from_ref(self);
        // We can't use `ptr::cast` here because extern types are `!Sized`.
        let ptr = ptr as *const RustString;
        unsafe {
            // Safety: `self` outlives returned `&RustString` and it is originated from `rustc`
            &*ptr
        }
    }

    fn into_inner(self) -> Vec<u8> {
        self.bytes.into_inner()
    }
}

impl RustString {
    pub(crate) fn build_byte_buffer(closure: impl FnOnce(&Self)) -> Vec<u8> {
        let buf = RustStringInner::default();
        closure(buf.as_opaque());
        buf.into_inner()
    }
}

pub(crate) fn build_string(f: impl FnOnce(&RustString)) -> Result<String, FromUtf8Error> {
    String::from_utf8(RustString::build_byte_buffer(f))
}
