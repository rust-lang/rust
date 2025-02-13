#![allow(non_snake_case)]

use std::ffi::{CStr, CString};
use std::ops::Deref;
use std::ptr;
use std::str::FromStr;
use std::string::FromUtf8Error;

use libc::c_uint;
use rustc_abi::{Align, Size, WrappingRange};
use rustc_llvm::RustString;

pub use self::CallConv::*;
pub use self::CodeGenOptSize::*;
pub use self::MetadataType::*;
pub use self::ffi::*;
use crate::common::AsCCharPtr;

pub mod archive_ro;
pub mod diagnostic;
pub mod enzyme_ffi;
mod ffi;

pub use self::enzyme_ffi::*;

impl LLVMRustResult {
    pub fn into_result(self) -> Result<(), ()> {
        match self {
            LLVMRustResult::Success => Ok(()),
            LLVMRustResult::Failure => Err(()),
        }
    }
}

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
            attr.as_c_char_ptr(),
            attr.len().try_into().unwrap(),
            value.as_c_char_ptr(),
            value.len().try_into().unwrap(),
        )
    }
}

pub fn CreateAttrString<'ll>(llcx: &'ll Context, attr: &str) -> &'ll Attribute {
    unsafe {
        LLVMCreateStringAttribute(
            llcx,
            attr.as_c_char_ptr(),
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

pub fn CreateRangeAttr(llcx: &Context, size: Size, range: WrappingRange) -> &Attribute {
    let lower = range.start;
    let upper = range.end.wrapping_add(1);
    let lower_words = [lower as u64, (lower >> 64) as u64];
    let upper_words = [upper as u64, (upper >> 64) as u64];
    unsafe {
        LLVMRustCreateRangeAttribute(
            llcx,
            size.bits().try_into().unwrap(),
            lower_words.as_ptr(),
            upper_words.as_ptr(),
        )
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
    let name_buf = get_value_name(val).to_vec();
    let name =
        CString::from_vec_with_nul(name_buf).or_else(|buf| CString::new(buf.into_bytes())).unwrap();
    set_comdat(llmod, val, &name);
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

pub fn set_section(llglobal: &Value, section_name: &CStr) {
    unsafe {
        LLVMSetSection(llglobal, section_name.as_ptr());
    }
}

pub fn add_global<'a>(llmod: &'a Module, ty: &'a Type, name_cstr: &CStr) -> &'a Value {
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

pub fn get_linkage(llglobal: &Value) -> Linkage {
    unsafe { LLVMGetLinkage(llglobal) }.to_rust()
}

pub fn set_linkage(llglobal: &Value, linkage: Linkage) {
    unsafe {
        LLVMSetLinkage(llglobal, linkage);
    }
}

pub fn is_declaration(llglobal: &Value) -> bool {
    unsafe { LLVMIsDeclaration(llglobal) == ffi::True }
}

pub fn get_visibility(llglobal: &Value) -> Visibility {
    unsafe { LLVMGetVisibility(llglobal) }.to_rust()
}

pub fn set_visibility(llglobal: &Value, visibility: Visibility) {
    unsafe {
        LLVMSetVisibility(llglobal, visibility);
    }
}

pub fn set_alignment(llglobal: &Value, align: Align) {
    unsafe {
        ffi::LLVMSetAlignment(llglobal, align.bytes() as c_uint);
    }
}

/// Get the `name`d comdat from `llmod` and assign it to `llglobal`.
///
/// Inserts the comdat into `llmod` if it does not exist.
/// It is an error to call this if the target does not support comdat.
pub fn set_comdat(llmod: &Module, llglobal: &Value, name: &CStr) {
    unsafe {
        let comdat = LLVMGetOrInsertComdat(llmod, name.as_ptr());
        LLVMSetComdat(llglobal, comdat);
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
        let data = name.as_c_char_ptr();
        LLVMSetValueName2(value, data, name.len());
    }
}

pub fn build_string(f: impl FnOnce(&RustString)) -> Result<String, FromUtf8Error> {
    String::from_utf8(RustString::build_byte_buffer(f))
}

pub fn build_byte_buffer(f: impl FnOnce(&RustString)) -> Vec<u8> {
    RustString::build_byte_buffer(f)
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

/// Owns an [`OperandBundle`], and will dispose of it when dropped.
pub(crate) struct OperandBundleOwned<'a> {
    raw: ptr::NonNull<OperandBundle<'a>>,
}

impl<'a> OperandBundleOwned<'a> {
    pub(crate) fn new(name: &str, vals: &[&'a Value]) -> Self {
        let raw = unsafe {
            LLVMCreateOperandBundle(
                name.as_c_char_ptr(),
                name.len(),
                vals.as_ptr(),
                vals.len() as c_uint,
            )
        };
        OperandBundleOwned { raw: ptr::NonNull::new(raw).unwrap() }
    }
}

impl Drop for OperandBundleOwned<'_> {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeOperandBundle(self.raw);
        }
    }
}

impl<'a> Deref for OperandBundleOwned<'a> {
    type Target = OperandBundle<'a>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: The returned reference is opaque and can only used for FFI.
        // It is valid for as long as `&self` is.
        unsafe { self.raw.as_ref() }
    }
}

pub(crate) fn add_module_flag_u32(
    module: &Module,
    merge_behavior: ModuleFlagMergeBehavior,
    key: &str,
    value: u32,
) {
    unsafe {
        LLVMRustAddModuleFlagU32(module, merge_behavior, key.as_c_char_ptr(), key.len(), value);
    }
}

pub(crate) fn add_module_flag_str(
    module: &Module,
    merge_behavior: ModuleFlagMergeBehavior,
    key: &str,
    value: &str,
) {
    unsafe {
        LLVMRustAddModuleFlagString(
            module,
            merge_behavior,
            key.as_c_char_ptr(),
            key.len(),
            value.as_c_char_ptr(),
            value.len(),
        );
    }
}
