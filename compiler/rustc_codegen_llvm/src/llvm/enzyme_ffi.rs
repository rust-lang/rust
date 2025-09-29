#![expect(dead_code)]

use libc::{c_char, c_uint};

use super::MetadataKindId;
use super::ffi::{AttributeKind, BasicBlock, Context, Metadata, Module, Type, Value};
use crate::llvm::{Bool, Builder};

// TypeTree types
pub(crate) type CTypeTreeRef = *mut EnzymeTypeTree;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub(crate) struct EnzymeTypeTree {
    _unused: [u8; 0],
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub(crate) enum CConcreteType {
    DT_Anything = 0,
    DT_Integer = 1,
    DT_Pointer = 2,
    DT_Half = 3,
    DT_Float = 4,
    DT_Double = 5,
    DT_Unknown = 6,
    DT_FP128 = 9,
}

pub(crate) struct TypeTree {
    pub(crate) inner: CTypeTreeRef,
}

#[link(name = "llvm-wrapper", kind = "static")]
unsafe extern "C" {
    // Enzyme
    pub(crate) safe fn LLVMRustHasMetadata(I: &Value, KindID: MetadataKindId) -> bool;
    pub(crate) fn LLVMRustEraseInstUntilInclusive(BB: &BasicBlock, I: &Value);
    pub(crate) fn LLVMRustGetLastInstruction<'a>(BB: &BasicBlock) -> Option<&'a Value>;
    pub(crate) fn LLVMRustDIGetInstMetadata(I: &Value) -> Option<&Metadata>;
    pub(crate) fn LLVMRustEraseInstFromParent(V: &Value);
    pub(crate) fn LLVMRustGetTerminator<'a>(B: &BasicBlock) -> &'a Value;
    pub(crate) fn LLVMRustVerifyFunction(V: &Value, action: LLVMRustVerifierFailureAction) -> Bool;
    pub(crate) fn LLVMRustHasAttributeAtIndex(V: &Value, i: c_uint, Kind: AttributeKind) -> bool;
    pub(crate) fn LLVMRustGetArrayNumElements(Ty: &Type) -> u64;
    pub(crate) fn LLVMRustHasFnAttribute(
        F: &Value,
        Name: *const c_char,
        NameLen: libc::size_t,
    ) -> bool;
    pub(crate) fn LLVMRustRemoveFnAttribute(F: &Value, Name: *const c_char, NameLen: libc::size_t);
    pub(crate) fn LLVMGetFirstFunction(M: &Module) -> Option<&Value>;
    pub(crate) fn LLVMGetNextFunction(Fn: &Value) -> Option<&Value>;
    pub(crate) fn LLVMRustRemoveEnumAttributeAtIndex(
        Fn: &Value,
        index: c_uint,
        kind: AttributeKind,
    );
    pub(crate) fn LLVMRustPositionBefore<'a>(B: &'a Builder<'_>, I: &'a Value);
    pub(crate) fn LLVMRustPositionAfter<'a>(B: &'a Builder<'_>, I: &'a Value);
    pub(crate) fn LLVMRustGetFunctionCall(
        F: &Value,
        name: *const c_char,
        NameLen: libc::size_t,
    ) -> Option<&Value>;

}

unsafe extern "C" {
    // Enzyme
    pub(crate) fn LLVMDumpModule(M: &Module);
    pub(crate) fn LLVMDumpValue(V: &Value);
    pub(crate) fn LLVMGetFunctionCallConv(F: &Value) -> c_uint;
    pub(crate) fn LLVMGetReturnType(T: &Type) -> &Type;
    pub(crate) fn LLVMGetParams(Fnc: &Value, params: *mut &Value);
    pub(crate) fn LLVMGetNamedFunction(M: &Module, Name: *const c_char) -> Option<&Value>;
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq)]
pub(crate) enum LLVMRustVerifierFailureAction {
    LLVMAbortProcessAction = 0,
    LLVMPrintMessageAction = 1,
    LLVMReturnStatusAction = 2,
}

#[cfg(feature = "llvm_enzyme")]
pub(crate) use self::Enzyme_AD::*;

#[cfg(feature = "llvm_enzyme")]
pub(crate) mod Enzyme_AD {
    use std::ffi::{CString, c_char};

    use libc::c_void;

    use super::{CConcreteType, CTypeTreeRef, Context};

    unsafe extern "C" {
        pub(crate) fn EnzymeSetCLBool(arg1: *mut ::std::os::raw::c_void, arg2: u8);
        pub(crate) fn EnzymeSetCLString(arg1: *mut ::std::os::raw::c_void, arg2: *const c_char);
    }

    // TypeTree functions
    unsafe extern "C" {
        pub(crate) fn EnzymeNewTypeTree() -> CTypeTreeRef;
        pub(crate) fn EnzymeNewTypeTreeCT(arg1: CConcreteType, ctx: &Context) -> CTypeTreeRef;
        pub(crate) fn EnzymeNewTypeTreeTR(arg1: CTypeTreeRef) -> CTypeTreeRef;
        pub(crate) fn EnzymeFreeTypeTree(CTT: CTypeTreeRef);
        pub(crate) fn EnzymeMergeTypeTree(arg1: CTypeTreeRef, arg2: CTypeTreeRef) -> bool;
        pub(crate) fn EnzymeTypeTreeOnlyEq(arg1: CTypeTreeRef, pos: i64);
        pub(crate) fn EnzymeTypeTreeData0Eq(arg1: CTypeTreeRef);
        pub(crate) fn EnzymeTypeTreeShiftIndiciesEq(
            arg1: CTypeTreeRef,
            data_layout: *const c_char,
            offset: i64,
            max_size: i64,
            add_offset: u64,
        );
        pub(crate) fn EnzymeTypeTreeInsertEq(
            CTT: CTypeTreeRef,
            indices: *const i64,
            len: usize,
            ct: CConcreteType,
            ctx: &Context,
        );
        pub(crate) fn EnzymeTypeTreeToString(arg1: CTypeTreeRef) -> *const c_char;
        pub(crate) fn EnzymeTypeTreeToStringFree(arg1: *const c_char);
    }

    unsafe extern "C" {
        static mut EnzymePrintPerf: c_void;
        static mut EnzymePrintActivity: c_void;
        static mut EnzymePrintType: c_void;
        static mut EnzymeFunctionToAnalyze: c_void;
        static mut EnzymePrint: c_void;
        static mut EnzymeStrictAliasing: c_void;
        static mut looseTypeAnalysis: c_void;
        static mut EnzymeInline: c_void;
        static mut RustTypeRules: c_void;
    }
    pub(crate) fn set_print_perf(print: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymePrintPerf), print as u8);
        }
    }
    pub(crate) fn set_print_activity(print: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymePrintActivity), print as u8);
        }
    }
    pub(crate) fn set_print_type(print: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymePrintType), print as u8);
        }
    }
    pub(crate) fn set_print_type_fun(fun_name: &str) {
        let c_fun_name = CString::new(fun_name).unwrap();
        unsafe {
            EnzymeSetCLString(
                std::ptr::addr_of_mut!(EnzymeFunctionToAnalyze),
                c_fun_name.as_ptr() as *const c_char,
            );
        }
    }
    pub(crate) fn set_print(print: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymePrint), print as u8);
        }
    }
    pub(crate) fn set_strict_aliasing(strict: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymeStrictAliasing), strict as u8);
        }
    }
    pub(crate) fn set_loose_types(loose: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(looseTypeAnalysis), loose as u8);
        }
    }
    pub(crate) fn set_inline(val: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymeInline), val as u8);
        }
    }
    pub(crate) fn set_rust_rules(val: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(RustTypeRules), val as u8);
        }
    }
}

#[cfg(not(feature = "llvm_enzyme"))]
pub(crate) use self::Fallback_AD::*;

#[cfg(not(feature = "llvm_enzyme"))]
pub(crate) mod Fallback_AD {
    #![allow(unused_variables)]

    use libc::c_char;

    use super::{CConcreteType, CTypeTreeRef, Context};

    // TypeTree function fallbacks
    pub(crate) unsafe fn EnzymeNewTypeTree() -> CTypeTreeRef {
        unimplemented!()
    }

    pub(crate) unsafe fn EnzymeNewTypeTreeCT(arg1: CConcreteType, ctx: &Context) -> CTypeTreeRef {
        unimplemented!()
    }

    pub(crate) unsafe fn EnzymeNewTypeTreeTR(arg1: CTypeTreeRef) -> CTypeTreeRef {
        unimplemented!()
    }

    pub(crate) unsafe fn EnzymeFreeTypeTree(CTT: CTypeTreeRef) {
        unimplemented!()
    }

    pub(crate) unsafe fn EnzymeMergeTypeTree(arg1: CTypeTreeRef, arg2: CTypeTreeRef) -> bool {
        unimplemented!()
    }

    pub(crate) unsafe fn EnzymeTypeTreeOnlyEq(arg1: CTypeTreeRef, pos: i64) {
        unimplemented!()
    }

    pub(crate) unsafe fn EnzymeTypeTreeData0Eq(arg1: CTypeTreeRef) {
        unimplemented!()
    }

    pub(crate) unsafe fn EnzymeTypeTreeShiftIndiciesEq(
        arg1: CTypeTreeRef,
        data_layout: *const c_char,
        offset: i64,
        max_size: i64,
        add_offset: u64,
    ) {
        unimplemented!()
    }

    pub(crate) unsafe fn EnzymeTypeTreeInsertEq(
        CTT: CTypeTreeRef,
        indices: *const i64,
        len: usize,
        ct: CConcreteType,
        ctx: &Context,
    ) {
        unimplemented!()
    }

    pub(crate) unsafe fn EnzymeTypeTreeToString(arg1: CTypeTreeRef) -> *const c_char {
        unimplemented!()
    }

    pub(crate) unsafe fn EnzymeTypeTreeToStringFree(arg1: *const c_char) {
        unimplemented!()
    }

    pub(crate) fn set_inline(val: bool) {
        unimplemented!()
    }
    pub(crate) fn set_print_perf(print: bool) {
        unimplemented!()
    }
    pub(crate) fn set_print_activity(print: bool) {
        unimplemented!()
    }
    pub(crate) fn set_print_type(print: bool) {
        unimplemented!()
    }
    pub(crate) fn set_print_type_fun(fun_name: &str) {
        unimplemented!()
    }
    pub(crate) fn set_print(print: bool) {
        unimplemented!()
    }
    pub(crate) fn set_strict_aliasing(strict: bool) {
        unimplemented!()
    }
    pub(crate) fn set_loose_types(loose: bool) {
        unimplemented!()
    }
    pub(crate) fn set_rust_rules(val: bool) {
        unimplemented!()
    }
}

impl TypeTree {
    pub(crate) fn new() -> TypeTree {
        let inner = unsafe { EnzymeNewTypeTree() };
        TypeTree { inner }
    }

    pub(crate) fn from_type(t: CConcreteType, ctx: &Context) -> TypeTree {
        let inner = unsafe { EnzymeNewTypeTreeCT(t, ctx) };
        TypeTree { inner }
    }

    pub(crate) fn merge(self, other: Self) -> Self {
        unsafe {
            EnzymeMergeTypeTree(self.inner, other.inner);
        }
        drop(other);
        self
    }

    #[must_use]
    pub(crate) fn shift(
        self,
        layout: &str,
        offset: isize,
        max_size: isize,
        add_offset: usize,
    ) -> Self {
        let layout = std::ffi::CString::new(layout).unwrap();

        unsafe {
            EnzymeTypeTreeShiftIndiciesEq(
                self.inner,
                layout.as_ptr(),
                offset as i64,
                max_size as i64,
                add_offset as u64,
            );
        }

        self
    }

    pub(crate) fn insert(&mut self, indices: &[i64], ct: CConcreteType, ctx: &Context) {
        unsafe {
            EnzymeTypeTreeInsertEq(self.inner, indices.as_ptr(), indices.len(), ct, ctx);
        }
    }
}

impl Clone for TypeTree {
    fn clone(&self) -> Self {
        let inner = unsafe { EnzymeNewTypeTreeTR(self.inner) };
        TypeTree { inner }
    }
}

impl std::fmt::Display for TypeTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ptr = unsafe { EnzymeTypeTreeToString(self.inner) };
        let cstr = unsafe { std::ffi::CStr::from_ptr(ptr) };
        match cstr.to_str() {
            Ok(x) => write!(f, "{}", x)?,
            Err(err) => write!(f, "could not parse: {}", err)?,
        }

        // delete C string pointer
        unsafe {
            EnzymeTypeTreeToStringFree(ptr);
        }

        Ok(())
    }
}

impl std::fmt::Debug for TypeTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as std::fmt::Display>::fmt(self, f)
    }
}

impl Drop for TypeTree {
    fn drop(&mut self) {
        unsafe { EnzymeFreeTypeTree(self.inner) }
    }
}
