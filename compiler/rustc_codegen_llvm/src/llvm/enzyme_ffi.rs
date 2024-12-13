#![allow(non_camel_case_types)]

use libc::{c_char, c_uint, size_t};

use super::ffi::{Attribute, BasicBlock, Builder, Metadata, Module, Type, Value};
extern "C" {
    // Enzyme
    pub fn LLVMRustHasMetadata(I: &Value, KindID: c_uint) -> bool;
    pub fn LLVMRustEraseInstBefore(BB: &BasicBlock, I: &Value);
    pub fn LLVMRustGetLastInstruction<'a>(BB: &BasicBlock) -> Option<&'a Value>;
    pub fn LLVMRustDIGetInstMetadata(I: &Value) -> &Metadata;
    pub fn LLVMRustEraseInstFromParent(V: &Value);
    pub fn LLVMRustGetTerminator<'a>(B: &BasicBlock) -> &'a Value;

    pub fn LLVMGetReturnType(T: &Type) -> &Type;
    pub fn LLVMDumpModule(M: &Module);
    pub fn LLVMCountStructElementTypes(T: &Type) -> c_uint;
    pub fn LLVMVerifyFunction(V: &Value, action: LLVMVerifierFailureAction) -> bool;
    pub fn LLVMGetParams(Fnc: &Value, parms: *mut &Value);
    pub fn LLVMBuildCall2<'a>(
        arg1: &Builder<'a>,
        ty: &Type,
        func: &Value,
        args: *mut &Value,
        num_args: size_t,
        name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMGetNamedFunction(M: &Module, Name: *const c_char) -> Option<&Value>;
    pub fn LLVMIsEnumAttribute(A: &Attribute) -> bool;
    pub fn LLVMIsStringAttribute(A: &Attribute) -> bool;
}

#[repr(C)]
pub enum LLVMVerifierFailureAction {
    LLVMAbortProcessAction,
    LLVMPrintMessageAction,
    LLVMReturnStatusAction,
}
