#![allow(non_snake_case)]

//use crate::enzyme::LLVMValueRef;

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

// pub fn GetNamedFunction<'ll>(name: &str) -> &'ll LLVMValueRef {
//     unsafe {
//         LLVMRustGetN
//     }
// }

pub fn AddFunctionAttributes<'ll>(llfn: &'ll Value, idx: AttributePlace, attrs: &[&'ll Attribute]) {
    unsafe {
        LLVMRustAddFunctionAttributes(llfn, idx.as_uint(), attrs.as_ptr(), attrs.len());
    }
}

pub fn AddCallSiteAttributes<'ll>(
    callsite: &'ll Value,
    idx: AttributePlace,
    attrs: &[&'ll Attribute],
) {
    unsafe {
        LLVMRustAddCallSiteAttributes(callsite, idx.as_uint(), attrs.as_ptr(), attrs.len());
    }
}

pub fn CreateAttrStringValue<'ll>(llcx: &'ll Context, attr: &str, value: &str) -> &'ll Attribute {
    unsafe {
        LLVMCreateStringAttribute(
            llcx,
            attr.as_ptr().cast(),
            attr.len().try_into().unwrap(),
            value.as_ptr().cast(),
            value.len().try_into().unwrap(),
        )
    }
}

pub fn CreateAttrString<'ll>(llcx: &'ll Context, attr: &str) -> &'ll Attribute {
    unsafe {
        LLVMCreateStringAttribute(
            llcx,
            attr.as_ptr().cast(),
            attr.len().try_into().unwrap(),
            std::ptr::null(),
            0,
        )
    }
}

pub fn CreateAlignmentAttr(llcx: &Context, bytes: u64) -> &Attribute {
    unsafe { LLVMRustCreateAlignmentAttr(llcx, bytes) }
}

pub fn CreateDereferenceableAttr(llcx: &Context, bytes: u64) -> &Attribute {
    unsafe { LLVMRustCreateDereferenceableAttr(llcx, bytes) }
}

pub fn CreateDereferenceableOrNullAttr(llcx: &Context, bytes: u64) -> &Attribute {
    unsafe { LLVMRustCreateDereferenceableOrNullAttr(llcx, bytes) }
}

pub fn CreateByValAttr<'ll>(llcx: &'ll Context, ty: &'ll Type) -> &'ll Attribute {
    unsafe { LLVMRustCreateByValAttr(llcx, ty) }
}

pub fn CreateStructRetAttr<'ll>(llcx: &'ll Context, ty: &'ll Type) -> &'ll Attribute {
    unsafe { LLVMRustCreateStructRetAttr(llcx, ty) }
}

pub fn CreateUWTableAttr(llcx: &Context, async_: bool) -> &Attribute {
    unsafe { LLVMRustCreateUWTableAttr(llcx, async_) }
}

pub fn CreateAllocSizeAttr(llcx: &Context, size_arg: u32) -> &Attribute {
    unsafe { LLVMRustCreateAllocSizeAttr(llcx, size_arg) }
}

pub fn CreateAllocKindAttr(llcx: &Context, kind_arg: AllocKindFlags) -> &Attribute {
    unsafe { LLVMRustCreateAllocKindAttr(llcx, kind_arg.bits()) }
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
            "aix_big" => Ok(ArchiveKind::K_AIXBIG),
            _ => Err(()),
        }
    }
}

pub fn SetInstructionCallConv(instr: &Value, cc: CallConv) {
    unsafe {
        LLVMSetInstructionCallConv(instr, cc as c_uint);
    }
}
pub fn SetFunctionCallConv(fn_: &Value, cc: CallConv) {
    unsafe {
        LLVMSetFunctionCallConv(fn_, cc as c_uint);
    }
}

// Externally visible symbols that might appear in multiple codegen units need to appear in
// their own comdat section so that the duplicates can be discarded at link time. This can for
// example happen for generics when using multiple codegen units. This function simply uses the
// value's name as the comdat value to make sure that it is in a 1-to-1 relationship to the
// function.
// For more details on COMDAT sections see e.g., https://www.airs.com/blog/archives/52
pub fn SetUniqueComdat(llmod: &Module, val: &Value) {
    unsafe {
        let name = get_value_name(val);
        LLVMRustSetComdat(llmod, val, name.as_ptr().cast(), name.len());
    }
}

pub fn SetUnnamedAddress(global: &Value, unnamed: UnnamedAddr) {
    unsafe {
        LLVMSetUnnamedAddress(global, unnamed);
    }
}

pub fn set_thread_local_mode(global: &Value, mode: ThreadLocalMode) {
    unsafe {
        LLVMSetThreadLocalMode(global, mode);
    }
}

impl AttributeKind {
    /// Create an LLVM Attribute with no associated value.
    pub fn create_attr(self, llcx: &Context) -> &Attribute {
        unsafe { LLVMRustCreateAttrNoValue(llcx, self) }
    }
}

impl MemoryEffects {
    /// Create an LLVM Attribute with these memory effects.
    pub fn create_attr(self, llcx: &Context) -> &Attribute {
        unsafe { LLVMRustCreateMemoryEffectsAttr(llcx, self) }
    }
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

pub fn set_visibility(llglobal: &Value, visibility: Visibility) {
    unsafe {
        LLVMRustSetVisibility(llglobal, visibility);
    }
}

pub fn set_alignment(llglobal: &Value, bytes: usize) {
    unsafe {
        ffi::LLVMSetAlignment(llglobal, bytes as c_uint);
    }
}

pub fn set_comdat(llmod: &Module, llglobal: &Value, name: &str) {
    unsafe {
        LLVMRustSetComdat(llmod, llglobal, name.as_ptr().cast(), name.len());
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

impl<'a> OperandBundleDef<'a> {
    pub fn new(name: &str, vals: &[&'a Value]) -> Self {
        let name = SmallCStr::new(name);
        let def = unsafe {
            LLVMRustBuildOperandBundleDef(name.as_ptr(), vals.as_ptr(), vals.len() as c_uint)
        };
        OperandBundleDef { raw: def }
    }
}

impl Drop for OperandBundleDef<'_> {
    fn drop(&mut self) {
        unsafe {
            LLVMRustFreeOperandBundleDef(&mut *(self.raw as *mut _));
        }
    }
}
