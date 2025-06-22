#![allow(non_camel_case_types)]

use libc::{c_char, c_uint};

use super::{AttributeKind, BasicBlock, Bool, Metadata, MetadataKindId, Module, Type, Value};

#[link(name = "llvm-wrapper", kind = "static")]
unsafe extern "C" {
    // Enzyme
    pub safe fn LLVMRustHasMetadata(I: &Value, KindID: MetadataKindId) -> bool;
    pub fn LLVMRustEraseInstUntilInclusive(BB: &BasicBlock, I: &Value);
    pub fn LLVMRustGetLastInstruction<'a>(BB: &BasicBlock) -> Option<&'a Value>;
    pub fn LLVMRustDIGetInstMetadata(I: &Value) -> Option<&Metadata>;
    pub fn LLVMRustEraseInstFromParent(V: &Value);
    pub fn LLVMRustGetTerminator<'a>(B: &BasicBlock) -> &'a Value;
    pub fn LLVMRustVerifyFunction(V: &Value, action: LLVMRustVerifierFailureAction) -> Bool;
    pub fn LLVMRustHasAttributeAtIndex(V: &Value, i: c_uint, Kind: AttributeKind) -> bool;
    pub fn LLVMRustGetArrayNumElements(Ty: &Type) -> u64;
    pub fn LLVMRustHasFnAttribute(F: &Value, Name: *const c_char, NameLen: libc::size_t) -> bool;
    pub fn LLVMRustRemoveFnAttribute(F: &Value, Name: *const c_char, NameLen: libc::size_t);
    pub fn LLVMRustRemoveEnumAttributeAtIndex(Fn: &Value, index: c_uint, kind: AttributeKind);
}

unsafe extern "C" {
    // Enzyme
    pub fn LLVMDumpModule(M: &Module);
    pub fn LLVMDumpValue(V: &Value);
    pub fn LLVMGetFunctionCallConv(F: &Value) -> c_uint;
    pub fn LLVMGetReturnType(T: &Type) -> &Type;
    pub fn LLVMGetParams(Fnc: &Value, params: *mut &Value);
    pub fn LLVMGetNamedFunction(M: &Module, Name: *const c_char) -> Option<&Value>;
    pub fn LLVMGetFirstFunction(M: &Module) -> Option<&Value>;
    pub fn LLVMGetNextFunction(Fn: &Value) -> Option<&Value>;
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq)]
pub enum LLVMRustVerifierFailureAction {
    LLVMAbortProcessAction = 0,
    LLVMPrintMessageAction = 1,
    LLVMReturnStatusAction = 2,
}

#[cfg(llvm_enzyme)]
pub use self::enzyme_ad::*;

#[cfg(llvm_enzyme)]
pub mod enzyme_ad {
    use std::ffi::{CString, c_char};

    use libc::c_void;
    unsafe extern "C" {
        pub fn EnzymeSetCLBool(arg1: *mut ::std::os::raw::c_void, arg2: u8);
        pub fn EnzymeSetCLString(arg1: *mut ::std::os::raw::c_void, arg2: *const c_char);
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
    pub fn set_print_perf(print: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymePrintPerf), print as u8);
        }
    }
    pub fn set_print_activity(print: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymePrintActivity), print as u8);
        }
    }
    pub fn set_print_type(print: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymePrintType), print as u8);
        }
    }
    pub fn set_print_type_fun(fun_name: &str) {
        let c_fun_name = CString::new(fun_name).unwrap();
        unsafe {
            EnzymeSetCLString(
                std::ptr::addr_of_mut!(EnzymeFunctionToAnalyze),
                c_fun_name.as_ptr() as *const c_char,
            );
        }
    }
    pub fn set_print(print: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymePrint), print as u8);
        }
    }
    pub fn set_strict_aliasing(strict: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymeStrictAliasing), strict as u8);
        }
    }
    pub fn set_loose_types(loose: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(looseTypeAnalysis), loose as u8);
        }
    }
    pub fn set_inline(val: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymeInline), val as u8);
        }
    }
    pub fn set_rust_rules(val: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(RustTypeRules), val as u8);
        }
    }
}

#[cfg(not(llvm_enzyme))]
pub use self::fallback_ad::*;

#[cfg(not(llvm_enzyme))]
pub mod fallback_ad {
    #![allow(unused_variables)]

    pub fn set_inline(val: bool) {
        unimplemented!()
    }
    pub fn set_print_perf(print: bool) {
        unimplemented!()
    }
    pub fn set_print_activity(print: bool) {
        unimplemented!()
    }
    pub fn set_print_type(print: bool) {
        unimplemented!()
    }
    pub fn set_print_type_fun(fun_name: &str) {
        unimplemented!()
    }
    pub fn set_print(print: bool) {
        unimplemented!()
    }
    pub fn set_strict_aliasing(strict: bool) {
        unimplemented!()
    }
    pub fn set_loose_types(loose: bool) {
        unimplemented!()
    }
    pub fn set_rust_rules(val: bool) {
        unimplemented!()
    }
}
