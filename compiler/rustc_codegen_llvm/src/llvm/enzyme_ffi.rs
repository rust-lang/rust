#![allow(non_camel_case_types)]
#![expect(dead_code)]

use libc::{c_char, c_uint};

use super::MetadataKindId;
use super::ffi::{AttributeKind, BasicBlock, Metadata, Module, Type, Value};
use crate::llvm::Bool;

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
}

unsafe extern "C" {
    // Enzyme
    pub(crate) fn LLVMDumpModule(M: &Module);
    pub(crate) fn LLVMDumpValue(V: &Value);
    pub(crate) fn LLVMGetFunctionCallConv(F: &Value) -> c_uint;
    pub(crate) fn LLVMGetReturnType(T: &Type) -> &Type;
    pub(crate) fn LLVMGetParams(Fnc: &Value, parms: *mut &Value);
    pub(crate) fn LLVMGetNamedFunction(M: &Module, Name: *const c_char) -> Option<&Value>;
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq)]
pub(crate) enum LLVMRustVerifierFailureAction {
    LLVMAbortProcessAction = 0,
    LLVMPrintMessageAction = 1,
    LLVMReturnStatusAction = 2,
}

#[cfg(llvm_enzyme)]
pub(crate) use self::Enzyme_AD::*;

#[cfg(llvm_enzyme)]
pub(crate) mod Enzyme_AD {
    use libc::c_void;
    unsafe extern "C" {
        pub(crate) fn EnzymeSetCLBool(arg1: *mut ::std::os::raw::c_void, arg2: u8);
    }
    unsafe extern "C" {
        static mut EnzymePrintPerf: c_void;
        static mut EnzymePrintActivity: c_void;
        static mut EnzymePrintType: c_void;
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

#[cfg(not(llvm_enzyme))]
pub(crate) use self::Fallback_AD::*;

#[cfg(not(llvm_enzyme))]
pub(crate) mod Fallback_AD {
    #![allow(unused_variables)]

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
