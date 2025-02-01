#![allow(non_camel_case_types)]

use libc::{c_char, c_uint};

use super::ffi::{BasicBlock, Metadata, Module, Type, Value};
use crate::llvm::Bool;

#[link(name = "llvm-wrapper", kind = "static")]
extern "C" {
    // Enzyme
    pub fn LLVMRustHasMetadata(I: &Value, KindID: c_uint) -> bool;
    pub fn LLVMRustEraseInstUntilInclusive(BB: &BasicBlock, I: &Value);
    pub fn LLVMRustGetLastInstruction<'a>(BB: &BasicBlock) -> Option<&'a Value>;
    pub fn LLVMRustDIGetInstMetadata(I: &Value) -> Option<&Metadata>;
    pub fn LLVMRustEraseInstFromParent(V: &Value);
    pub fn LLVMRustGetTerminator<'a>(B: &BasicBlock) -> &'a Value;
    pub fn LLVMRustVerifyFunction(V: &Value, action: LLVMRustVerifierFailureAction) -> Bool;
}

extern "C" {
    // Enzyme
    pub fn LLVMDumpModule(M: &Module);
    pub fn LLVMDumpValue(V: &Value);
    pub fn LLVMGetFunctionCallConv(F: &Value) -> c_uint;
    pub fn LLVMGetReturnType(T: &Type) -> &Type;
    pub fn LLVMGetParams(Fnc: &Value, parms: *mut &Value);
    pub fn LLVMGetNamedFunction(M: &Module, Name: *const c_char) -> Option<&Value>;
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq)]
pub enum LLVMRustVerifierFailureAction {
    LLVMAbortProcessAction = 0,
    LLVMPrintMessageAction = 1,
    LLVMReturnStatusAction = 2,
}
