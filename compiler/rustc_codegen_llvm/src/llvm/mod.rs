#![allow(non_snake_case)]

pub use self::AtomicRmwBinOp::*;
pub use self::CallConv::*;
pub use self::CodeGenOptSize::*;
pub use self::IntPredicate::*;
pub use self::Linkage::*;
pub use self::MetadataType::*;
pub use self::RealPredicate::*;

use libc::c_uint;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_llvm::RustString;
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::str::FromStr;
use std::string::FromUtf8Error;

pub mod archive_ro;
pub mod diagnostic;
mod ffi;

pub use self::ffi::*;

impl LLVMRustResult {
    pub fn into_result(self) -> Result<(), ()> {
        match self {
            LLVMRustResult::Success => Ok(()),
            LLVMRustResult::Failure => Err(()),
        }
    }
}

pub fn AddFunctionAttrStringValue(llfn: &'a Value, idx: AttributePlace, attr: &CStr, value: &CStr) {
    unsafe {
        LLVMRustAddFunctionAttrStringValue(llfn, idx.as_uint(), attr.as_ptr(), value.as_ptr())
    }
}

pub fn AddFunctionAttrString(llfn: &'a Value, idx: AttributePlace, attr: &CStr) {
    unsafe {
        LLVMRustAddFunctionAttrStringValue(llfn, idx.as_uint(), attr.as_ptr(), std::ptr::null())
    }
}

#[derive(Copy, Clone)]
pub enum AttributePlace {
    ReturnValue,
    Argument(u32),
    Function,
}

impl AttributePlace {
    pub fn as_uint(self) -> c_uint {
        match self {
            AttributePlace::ReturnValue => 0,
            AttributePlace::Argument(i) => 1 + i,
            AttributePlace::Function => !0,
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub enum CodeGenOptSize {
    CodeGenOptSizeNone = 0,
    CodeGenOptSizeDefault = 1,
    CodeGenOptSizeAggressive = 2,
}

impl FromStr for ArchiveKind {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gnu" => Ok(ArchiveKind::K_GNU),
            "bsd" => Ok(ArchiveKind::K_BSD),
            "darwin" => Ok(ArchiveKind::K_DARWIN),
            "coff" => Ok(ArchiveKind::K_COFF),
            _ => Err(()),
        }
    }
}

pub fn SetInstructionCallConv(instr: &'a Value, cc: CallConv) {
    unsafe {
        LLVMSetInstructionCallConv(instr, cc as c_uint);
    }
}
pub fn SetFunctionCallConv(fn_: &'a Value, cc: CallConv) {
    unsafe {
        LLVMSetFunctionCallConv(fn_, cc as c_uint);
    }
}

// Externally visible symbols that might appear in multiple codegen units need to appear in
// their own comdat section so that the duplicates can be discarded at link time. This can for
// example happen for generics when using multiple codegen units. This function simply uses the
// value's name as the comdat value to make sure that it is in a 1-to-1 relationship to the
// function.
// For more details on COMDAT sections see e.g., http://www.airs.com/blog/archives/52
pub fn SetUniqueComdat(llmod: &Module, val: &'a Value) {
    unsafe {
        let name = get_value_name(val);
        LLVMRustSetComdat(llmod, val, name.as_ptr().cast(), name.len());
    }
}

pub fn UnsetComdat(val: &'a Value) {
    unsafe {
        LLVMRustUnsetComdat(val);
    }
}

pub fn SetUnnamedAddress(global: &'a Value, unnamed: UnnamedAddr) {
    unsafe {
        LLVMSetUnnamedAddress(global, unnamed);
    }
}

pub fn set_thread_local_mode(global: &'a Value, mode: ThreadLocalMode) {
    unsafe {
        LLVMSetThreadLocalMode(global, mode);
    }
}

impl Attribute {
    pub fn apply_llfn(&self, idx: AttributePlace, llfn: &Value) {
        unsafe { LLVMRustAddFunctionAttribute(llfn, idx.as_uint(), *self) }
    }

    pub fn apply_callsite(&self, idx: AttributePlace, callsite: &Value) {
        unsafe { LLVMRustAddCallSiteAttribute(callsite, idx.as_uint(), *self) }
    }

    pub fn unapply_llfn(&self, idx: AttributePlace, llfn: &Value) {
        unsafe { LLVMRustRemoveFunctionAttributes(llfn, idx.as_uint(), *self) }
    }

    pub fn toggle_llfn(&self, idx: AttributePlace, llfn: &Value, set: bool) {
        if set {
            self.apply_llfn(idx, llfn);
        } else {
            self.unapply_llfn(idx, llfn);
        }
    }
}

// Memory-managed interface to object files.

pub struct ObjectFile {
    pub llof: &'static mut ffi::ObjectFile,
}

unsafe impl Send for ObjectFile {}

impl ObjectFile {
    // This will take ownership of llmb
    pub fn new(llmb: &'static mut MemoryBuffer) -> Option<ObjectFile> {
        unsafe {
            let llof = LLVMCreateObjectFile(llmb)?;
            Some(ObjectFile { llof })
        }
    }
}

impl Drop for ObjectFile {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeObjectFile(&mut *(self.llof as *mut _));
        }
    }
}

// Memory-managed interface to section iterators.

pub struct SectionIter<'a> {
    pub llsi: &'a mut SectionIterator<'a>,
}

impl Drop for SectionIter<'a> {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeSectionIterator(&mut *(self.llsi as *mut _));
        }
    }
}

pub fn mk_section_iter(llof: &ffi::ObjectFile) -> SectionIter<'_> {
    unsafe { SectionIter { llsi: LLVMGetSections(llof) } }
}

pub fn set_section(llglobal: &Value, section_name: &str) {
    let section_name_cstr = CString::new(section_name).expect("unexpected CString error");
    unsafe {
        LLVMSetSection(llglobal, section_name_cstr.as_ptr());
    }
}

pub fn add_global<'a>(llmod: &'a Module, ty: &'a Type, name: &str) -> &'a Value {
    let name_cstr = CString::new(name).expect("unexpected CString error");
    unsafe { LLVMAddGlobal(llmod, ty, name_cstr.as_ptr()) }
}

pub fn set_initializer(llglobal: &Value, constant_val: &Value) {
    unsafe {
        LLVMSetInitializer(llglobal, constant_val);
    }
}

pub fn set_global_constant(llglobal: &Value, is_constant: bool) {
    unsafe {
        LLVMSetGlobalConstant(llglobal, if is_constant { ffi::True } else { ffi::False });
    }
}

pub fn set_linkage(llglobal: &Value, linkage: Linkage) {
    unsafe {
        LLVMRustSetLinkage(llglobal, linkage);
    }
}

pub fn set_alignment(llglobal: &Value, bytes: usize) {
    unsafe {
        ffi::LLVMSetAlignment(llglobal, bytes as c_uint);
    }
}

/// Safe wrapper around `LLVMGetParam`, because segfaults are no fun.
pub fn get_param(llfn: &Value, index: c_uint) -> &Value {
    unsafe {
        assert!(
            index < LLVMCountParams(llfn),
            "out of bounds argument access: {} out of {} arguments",
            index,
            LLVMCountParams(llfn)
        );
        LLVMGetParam(llfn, index)
    }
}

/// Safe wrapper for `LLVMGetValueName2` into a byte slice
pub fn get_value_name(value: &Value) -> &[u8] {
    unsafe {
        let mut len = 0;
        let data = LLVMGetValueName2(value, &mut len);
        std::slice::from_raw_parts(data.cast(), len)
    }
}

/// Safe wrapper for `LLVMSetValueName2` from a byte slice
pub fn set_value_name(value: &Value, name: &[u8]) {
    unsafe {
        let data = name.as_ptr().cast();
        LLVMSetValueName2(value, data, name.len());
    }
}

pub fn build_string(f: impl FnOnce(&RustString)) -> Result<String, FromUtf8Error> {
    let sr = RustString { bytes: RefCell::new(Vec::new()) };
    f(&sr);
    String::from_utf8(sr.bytes.into_inner())
}

pub fn build_byte_buffer(f: impl FnOnce(&RustString)) -> Vec<u8> {
    let sr = RustString { bytes: RefCell::new(Vec::new()) };
    f(&sr);
    sr.bytes.into_inner()
}

pub fn twine_to_string(tr: &Twine) -> String {
    unsafe {
        build_string(|s| LLVMRustWriteTwineToString(tr, s)).expect("got a non-UTF8 Twine from LLVM")
    }
}

pub fn last_error() -> Option<String> {
    unsafe {
        let cstr = LLVMRustGetLastError();
        if cstr.is_null() {
            None
        } else {
            let err = CStr::from_ptr(cstr).to_bytes();
            let err = String::from_utf8_lossy(err).to_string();
            libc::free(cstr as *mut _);
            Some(err)
        }
    }
}

pub struct OperandBundleDef<'a> {
    pub raw: &'a mut ffi::OperandBundleDef<'a>,
}

impl OperandBundleDef<'a> {
    pub fn new(name: &str, vals: &[&'a Value]) -> Self {
        let name = SmallCStr::new(name);
        let def = unsafe {
            LLVMRustBuildOperandBundleDef(name.as_ptr(), vals.as_ptr(), vals.len() as c_uint)
        };
        OperandBundleDef { raw: def }
    }
}

impl Drop for OperandBundleDef<'a> {
    fn drop(&mut self) {
        unsafe {
            LLVMRustFreeOperandBundleDef(&mut *(self.raw as *mut _));
        }
    }
}
