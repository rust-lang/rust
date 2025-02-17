#![allow(non_camel_case_types)]
#![expect(dead_code)]

use libc::{c_char, c_uint};

use super::ffi::{BasicBlock, Metadata, Module, Type, Value};
use crate::llvm::Bool;

#[link(name = "llvm-wrapper", kind = "static")]
extern "C" {
    // Enzyme
    pub(crate) fn LLVMRustHasMetadata(I: &Value, KindID: c_uint) -> bool;
    pub(crate) fn LLVMRustEraseInstUntilInclusive(BB: &BasicBlock, I: &Value);
    pub(crate) fn LLVMRustGetLastInstruction<'a>(BB: &BasicBlock) -> Option<&'a Value>;
    pub(crate) fn LLVMRustDIGetInstMetadata(I: &Value) -> Option<&Metadata>;
    pub(crate) fn LLVMRustEraseInstFromParent(V: &Value);
    pub(crate) fn LLVMRustGetTerminator<'a>(B: &BasicBlock) -> &'a Value;
    pub(crate) fn LLVMRustVerifyFunction(V: &Value, action: LLVMRustVerifierFailureAction) -> Bool;
}

extern "C" {
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
pub enum LLVMRustVerifierFailureAction {
    LLVMAbortProcessAction = 0,
    LLVMPrintMessageAction = 1,
    LLVMReturnStatusAction = 2,
}
