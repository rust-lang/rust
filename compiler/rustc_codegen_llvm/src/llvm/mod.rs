#![allow(non_snake_case)]

use std::ffi::{CStr, CString};
use std::num::NonZero;
use std::ptr;
use std::string::FromUtf8Error;

use libc::c_uint;
use rustc_abi::{AddressSpace, Align, Size, WrappingRange};
use rustc_llvm::RustString;

pub(crate) use self::CallConv::*;
pub(crate) use self::conversions::*;
pub(crate) use self::ffi::*;
pub(crate) use self::metadata_kind::*;

mod conversions;
pub(crate) mod diagnostic;
pub(crate) mod enzyme_ffi;
mod ffi;
mod metadata_kind;
pub(crate) mod offload_ffi;
pub(crate) mod tpde_ffi;

pub(crate) use self::enzyme_ffi::*;
pub(crate) use self::offload_ffi::*;
pub(crate) use self::tpde_ffi::*;

impl LLVMRustResult {
    pub(crate) fn into_result(self) -> Result<(), ()> {
        match self {
            LLVMRustResult::Success => Ok(()),
            LLVMRustResult::Failure => Err(()),
        }
    }
}

pub(crate) fn AddFunctionAttributes<'ll>(
    llfn: Value<'ll>,
    idx: AttributePlace,
    attrs: &[&'ll Attribute],
) {
    unsafe {
        LLVMRustAddFunctionAttributes(llfn, idx.as_uint(), attrs.as_ptr(), attrs.len());
    }
}

pub(crate) fn HasStringAttribute<'ll>(llfn: Value<'ll>, name: &str) -> bool {
    unsafe { LLVMRustHasFnAttribute(llfn, name.as_ptr(), name.len()) }
}

pub(crate) fn RemoveStringAttrFromFn<'ll>(llfn: Value<'ll>, name: &str) {
    unsafe { LLVMRustRemoveFnAttribute(llfn, name.as_ptr(), name.len()) }
}

pub(crate) fn AddCallSiteAttributes<'ll>(
    callsite: Value<'ll>,
    idx: AttributePlace,
    attrs: &[&'ll Attribute],
) {
    unsafe {
        LLVMRustAddCallSiteAttributes(callsite, idx.as_uint(), attrs.as_ptr(), attrs.len());
    }
}

pub(crate) fn CreateAttrStringValue<'ll>(
    llcx: Context<'ll>,
    attr: &str,
    value: &str,
) -> &'ll Attribute {
    unsafe {
        LLVMCreateStringAttribute(
            llcx,
            attr.as_ptr(),
            attr.len().try_into().unwrap(),
            value.as_ptr(),
            value.len().try_into().unwrap(),
        )
    }
}
pub(crate) fn CreateAttrStringValueFromCStr<'ll>(
    llcx: Context<'ll>,
    attr: &std::ffi::CStr,
    value: &std::ffi::CStr,
) -> &'ll Attribute {
    let attr = attr.to_bytes();
    let value = value.to_bytes();
    unsafe {
        LLVMCreateStringAttribute(
            llcx,
            attr.as_ptr(),
            attr.len() as c_uint,
            value.as_ptr(),
            value.len() as c_uint,
        )
    }
}

pub(crate) fn CreateAttrString<'ll>(llcx: Context<'ll>, attr: &str) -> &'ll Attribute {
    unsafe {
        LLVMCreateStringAttribute(
            llcx,
            attr.as_ptr(),
            attr.len().try_into().unwrap(),
            std::ptr::null(),
            0,
        )
    }
}

pub(crate) fn CreateAlignmentAttr<'ll>(llcx: Context<'ll>, bytes: u64) -> &'ll Attribute {
    unsafe { LLVMRustCreateAlignmentAttr(llcx, bytes) }
}

pub(crate) fn CreateDereferenceableAttr<'ll>(llcx: Context<'ll>, bytes: u64) -> &'ll Attribute {
    unsafe { LLVMRustCreateDereferenceableAttr(llcx, bytes) }
}

pub(crate) fn CreateDereferenceableOrNullAttr<'ll>(
    llcx: Context<'ll>,
    bytes: u64,
) -> &'ll Attribute {
    unsafe { LLVMRustCreateDereferenceableOrNullAttr(llcx, bytes) }
}

pub(crate) fn CreateByValAttr<'ll>(llcx: Context<'ll>, ty: Type<'ll>) -> &'ll Attribute {
    unsafe { LLVMRustCreateByValAttr(llcx, ty) }
}

pub(crate) fn CreateByRefAttr<'ll>(llcx: Context<'ll>, ty: Type<'ll>) -> &'ll Attribute {
    unsafe { LLVMRustCreateByRefAttr(llcx, ty) }
}

pub(crate) fn CreateStructRetAttr<'ll>(llcx: Context<'ll>, ty: Type<'ll>) -> &'ll Attribute {
    unsafe { LLVMRustCreateStructRetAttr(llcx, ty) }
}

pub(crate) fn CreateUWTableAttr<'ll>(llcx: Context<'ll>, async_: bool) -> &'ll Attribute {
    unsafe { LLVMRustCreateUWTableAttr(llcx, async_) }
}

pub(crate) fn CreateAllocSizeAttr<'ll>(llcx: Context<'ll>, size_arg: u32) -> &'ll Attribute {
    unsafe { LLVMRustCreateAllocSizeAttr(llcx, size_arg) }
}

pub(crate) fn CreateAllocKindAttr<'ll>(
    llcx: Context<'ll>,
    kind_arg: AllocKindFlags,
) -> &'ll Attribute {
    unsafe { LLVMRustCreateAllocKindAttr(llcx, kind_arg.bits()) }
}

pub(crate) fn CreateRangeAttr<'ll>(
    llcx: Context<'ll>,
    size: Size,
    range: WrappingRange,
) -> &'ll Attribute {
    let lower = range.start;
    // LLVM treats the upper bound as exclusive, but allows wrapping.
    let upper = range.end.wrapping_add(1);

    // Pass each `u128` endpoint value as a `[u64; 2]` array, least-significant part first.
    let as_u64_array = |x: u128| [x as u64, (x >> 64) as u64];
    let lower_words: [u64; 2] = as_u64_array(lower);
    let upper_words: [u64; 2] = as_u64_array(upper);

    // To ensure that LLVM doesn't try to read beyond the `[u64; 2]` arrays,
    // we must explicitly check that `size_bits` does not exceed 128.
    let size_bits = size.bits();
    assert!(size_bits <= 128);
    // More robust assertions that are redundant with `size_bits <= 128` and
    // should be optimized away.
    assert!(size_bits.div_ceil(64) <= u64::try_from(lower_words.len()).unwrap());
    assert!(size_bits.div_ceil(64) <= u64::try_from(upper_words.len()).unwrap());
    let size_bits = c_uint::try_from(size_bits).unwrap();

    unsafe {
        LLVMRustCreateRangeAttribute(llcx, size_bits, lower_words.as_ptr(), upper_words.as_ptr())
    }
}

#[derive(Copy, Clone)]
pub(crate) enum AttributePlace {
    ReturnValue,
    Argument(u32),
    Function,
}

impl AttributePlace {
    pub(crate) fn as_uint(self) -> c_uint {
        match self {
            AttributePlace::ReturnValue => 0,
            AttributePlace::Argument(i) => 1 + i,
            AttributePlace::Function => !0,
        }
    }
}

pub(crate) fn SetInstructionCallConv<'ll>(instr: Value<'ll>, cc: CallConv) {
    unsafe {
        LLVMSetInstructionCallConv(instr, cc as c_uint);
    }
}
pub(crate) fn SetFunctionCallConv<'ll>(fn_: Value<'ll>, cc: CallConv) {
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
pub(crate) fn SetUniqueComdat<'ll>(llmod: Module<'ll>, val: Value<'ll>) {
    let name_buf = get_value_name(val);
    let name =
        CString::from_vec_with_nul(name_buf).or_else(|buf| CString::new(buf.into_bytes())).unwrap();
    set_comdat(llmod, val, &name);
}

pub(crate) fn set_unnamed_address<'ll>(global: Value<'ll>, unnamed: UnnamedAddr) {
    LLVMSetUnnamedAddress(global, unnamed);
}

pub(crate) fn set_thread_local_mode<'ll>(global: Value<'ll>, mode: ThreadLocalMode) {
    unsafe {
        LLVMSetThreadLocalMode(global, mode);
    }
}

impl AttributeKind {
    /// Create an LLVM Attribute with no associated value.
    pub(crate) fn create_attr<'ll>(self, llcx: Context<'ll>) -> &'ll Attribute {
        unsafe { LLVMRustCreateAttrNoValue(llcx, self) }
    }
}

impl MemoryEffects {
    /// Create an LLVM Attribute with these memory effects.
    pub(crate) fn create_attr<'ll>(self, llcx: Context<'ll>) -> &'ll Attribute {
        unsafe { LLVMRustCreateMemoryEffectsAttr(llcx, self) }
    }
}

pub(crate) fn set_section<'ll>(llglobal: Value<'ll>, section_name: &CStr) {
    unsafe {
        LLVMSetSection(llglobal, section_name.as_ptr());
    }
}

pub(crate) fn add_global<'ll>(llmod: Module<'ll>, ty: Type<'ll>, name_cstr: &CStr) -> Value<'ll> {
    unsafe { LLVMAddGlobal(llmod, ty, name_cstr.as_ptr()) }
}

pub(crate) fn set_initializer<'ll>(llglobal: Value<'ll>, constant_val: Value<'ll>) {
    unsafe {
        assert_eq!(
            crate::llvm::LLVMGlobalGetValueType(llglobal),
            crate::llvm::LLVMTypeOf(constant_val)
        );
    }
    unsafe {
        LLVMSetInitializer(llglobal, constant_val);
    }
}

pub(crate) fn set_global_constant<'ll>(llglobal: Value<'ll>, is_constant: bool) {
    LLVMSetGlobalConstant(llglobal, is_constant.to_llvm_bool());
}

pub(crate) fn get_linkage<'ll>(llglobal: Value<'ll>) -> Linkage {
    unsafe { LLVMGetLinkage(llglobal) }.to_rust()
}

pub(crate) fn set_linkage<'ll>(llglobal: Value<'ll>, linkage: Linkage) {
    unsafe {
        LLVMSetLinkage(llglobal, linkage);
    }
}

pub(crate) fn is_declaration<'ll>(llglobal: Value<'ll>) -> bool {
    unsafe { LLVMIsDeclaration(llglobal) }.is_true()
}

pub(crate) fn get_visibility<'ll>(llglobal: Value<'ll>) -> Visibility {
    unsafe { LLVMGetVisibility(llglobal) }.to_rust()
}

pub(crate) fn set_visibility<'ll>(llglobal: Value<'ll>, visibility: Visibility) {
    unsafe {
        LLVMSetVisibility(llglobal, visibility);
    }
}

pub(crate) fn set_alignment<'ll>(llglobal: Value<'ll>, align: Align) {
    unsafe {
        ffi::LLVMSetAlignment(llglobal, align.bytes() as c_uint);
    }
}

pub(crate) fn set_externally_initialized<'ll>(llglobal: Value<'ll>, is_ext_init: bool) {
    LLVMSetExternallyInitialized(llglobal, is_ext_init.to_llvm_bool());
}

/// Get the `name`d comdat from `llmod` and assign it to `llglobal`.
///
/// Inserts the comdat into `llmod` if it does not exist.
/// It is an error to call this if the target does not support comdat.
pub(crate) fn set_comdat<'ll>(llmod: Module<'ll>, llglobal: Value<'ll>, name: &CStr) {
    unsafe {
        let comdat = LLVMGetOrInsertComdat(llmod, name.as_ptr());
        LLVMSetComdat(llglobal, comdat);
    }
}

pub(crate) fn count_params<'ll>(llfn: Value<'ll>) -> c_uint {
    LLVMCountParams(llfn)
}

/// Safe wrapper around `LLVMGetParam`, because segfaults are no fun.
pub(crate) fn get_param<'ll>(llfn: Value<'ll>, index: c_uint) -> Value<'ll> {
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

/// Safe wrapper for `LLVMGetValueName2`
/// Needs to allocate the value, because `set_value_name` will invalidate
/// the pointer.
pub(crate) fn get_value_name<'ll>(value: Value<'ll>) -> Vec<u8> {
    unsafe {
        let mut len = 0;
        let data = LLVMGetValueName2(value, &mut len);
        std::slice::from_raw_parts(data.cast(), len).to_vec()
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct Intrinsic {
    id: NonZero<c_uint>,
}

impl Intrinsic {
    pub(crate) fn lookup(name: &[u8]) -> Option<Self> {
        let id = unsafe { LLVMLookupIntrinsicID(name.as_ptr(), name.len()) };
        NonZero::new(id).map(|id| Self { id })
    }

    pub(crate) fn is_overloaded(self) -> bool {
        unsafe { LLVMIntrinsicIsOverloaded(self.id).is_true() }
    }

    pub(crate) fn is_target_specific(self) -> bool {
        unsafe { LLVMRustIsTargetIntrinsic(self.id) }
    }

    pub(crate) fn get_declaration<'ll>(
        self,
        llmod: Module<'ll>,
        type_params: &[Type<'ll>],
    ) -> Value<'ll> {
        unsafe {
            LLVMGetIntrinsicDeclaration(llmod, self.id, type_params.as_ptr(), type_params.len())
        }
    }
}

/// Safe wrapper for `LLVMSetValueName2` from a byte slice
pub(crate) fn set_value_name<'ll>(value: Value<'ll>, name: &[u8]) {
    assert!(!name.contains(&0));
    unsafe { LLVMSetValueName2(value, name.as_ptr(), name.len()) };
}

pub(crate) fn build_string(f: impl FnOnce(&RustString)) -> Result<String, FromUtf8Error> {
    String::from_utf8(RustString::build_byte_buffer(f))
}

pub(crate) fn build_byte_buffer(f: impl FnOnce(&RustString)) -> Vec<u8> {
    RustString::build_byte_buffer(f)
}

pub(crate) fn twine_to_string(tr: &Twine) -> String {
    unsafe {
        build_string(|s| LLVMRustWriteTwineToString(tr, s)).expect("got a non-UTF8 Twine from LLVM")
    }
}

pub(crate) fn last_error() -> Option<String> {
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

/// Owning pointer to an [`OperandBundle`] that will dispose of the bundle
/// when dropped.
pub(crate) struct OperandBundleBox<'ll> {
    raw: ptr::NonNull<OperandBundle<'ll>>,
}

impl<'ll> OperandBundleBox<'ll> {
    pub(crate) fn new(name: &str, vals: &[Value<'ll>]) -> Self {
        let raw = unsafe {
            LLVMCreateOperandBundle(name.as_ptr(), name.len(), vals.as_ptr(), vals.len() as c_uint)
        };
        Self { raw: ptr::NonNull::new(raw).unwrap() }
    }

    /// Dereferences to the underlying `&OperandBundle`.
    ///
    /// This can't be a `Deref` implementation because `OperandBundle` transitively
    /// contains an extern type, which is incompatible with `Deref::Target: ?Sized`.
    pub(crate) fn as_ref(&self) -> &OperandBundle<'ll> {
        // SAFETY: The returned reference is opaque and can only used for FFI.
        // It is valid for as long as `&self` is.
        unsafe { self.raw.as_ref() }
    }
}

impl Drop for OperandBundleBox<'_> {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeOperandBundle(self.raw);
        }
    }
}

pub(crate) fn add_module_flag_u32<'ll>(
    module: Module<'ll>,
    merge_behavior: ModuleFlagMergeBehavior,
    key: &str,
    value: u32,
) {
    unsafe {
        LLVMRustAddModuleFlagU32(module, merge_behavior, key.as_ptr(), key.len(), value);
    }
}

pub(crate) fn add_module_flag_str<'ll>(
    module: Module<'ll>,
    merge_behavior: ModuleFlagMergeBehavior,
    key: &str,
    value: &str,
) {
    unsafe {
        LLVMRustAddModuleFlagString(
            module,
            merge_behavior,
            key.as_ptr(),
            key.len(),
            value.as_ptr(),
            value.len(),
        );
    }
}

pub(crate) fn add_module_flag_metadata<'ll>(
    module: Module<'ll>,
    merge_behavior: ModuleFlagMergeBehavior,
    key: &str,
    value: &'ll Metadata,
) {
    unsafe {
        LLVMRustAddModuleFlagMetadata(module, merge_behavior, key.as_ptr(), key.len(), value);
    }
}

pub(crate) fn set_dllimport_storage_class<'ll>(v: Value<'ll>) {
    unsafe {
        LLVMSetDLLStorageClass(v, DLLStorageClass::DllImport);
    }
}

pub(crate) fn set_dso_local<'ll>(v: Value<'ll>) {
    unsafe {
        LLVMRustSetDSOLocal(v, true);
    }
}

/// Safe wrapper for `LLVMRustAppendModuleInlineAsm`, which delegates to
/// `Module::appendModuleInlineAsm`.
pub(crate) fn append_module_inline_asm<'ll>(
    llmod: Module<'ll>,
    asm: &[u8],
    target_features: &str,
    target_cpu: &str,
) {
    unsafe {
        LLVMRustAppendModuleInlineAsm(
            llmod,
            asm.as_ptr(),
            asm.len(),
            target_features.as_ptr(),
            target_features.len(),
            target_cpu.as_ptr(),
            target_cpu.len(),
        );
    }
}

/// Safe wrapper for `LLVMAddAlias2`
pub(crate) fn add_alias<'ll>(
    module: Module<'ll>,
    ty: Type<'ll>,
    address_space: AddressSpace,
    aliasee: Value<'ll>,
    name: &CStr,
) -> Value<'ll> {
    unsafe { LLVMAddAlias2(module, ty, address_space.0, aliasee, name.as_ptr()) }
}

/// Safe wrapper for `LLVMRustConstPtrAuth`.
pub(crate) fn const_ptr_auth<'ll>(
    ptr: Value<'ll>,
    key: u32,
    disc: u64,
    addr_diversity: Option<Value<'ll>>,
) -> Value<'ll> {
    unsafe { LLVMRustConstPtrAuth(ptr, key, disc, addr_diversity, None) }
}
