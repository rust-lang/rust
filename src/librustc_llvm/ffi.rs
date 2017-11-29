// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use LLVMAtomicOrdering as AtomicOrdering;
pub use LLVMAtomicRMWBinOp as AtomicRmwBinOp;
pub use LLVMBasicBlockRef as BasicBlockRef;
pub use LLVMBool as Bool;
pub use LLVMBuilderRef as BuilderRef;
pub use LLVMContextRef as ContextRef;
pub use LLVMDLLStorageClass as DLLStorageClass;
pub use LLVMDebugLocRef as DebugLocRef;
pub use LLVMDiagnosticInfoRef as DiagnosticInfoRef;
pub use LLVMIntPredicate as IntPredicate;
pub use LLVMIntPredicate::LLVMIntEQ as IntEQ;
pub use LLVMIntPredicate::LLVMIntNE as IntNE;
pub use LLVMIntPredicate::LLVMIntSGE as IntSGE;
pub use LLVMIntPredicate::LLVMIntSGT as IntSGT;
pub use LLVMIntPredicate::LLVMIntSLE as IntSLE;
pub use LLVMIntPredicate::LLVMIntSLT as IntSLT;
pub use LLVMIntPredicate::LLVMIntUGE as IntUGE;
pub use LLVMIntPredicate::LLVMIntUGT as IntUGT;
pub use LLVMIntPredicate::LLVMIntULE as IntULE;
pub use LLVMIntPredicate::LLVMIntULT as IntULT;
pub use LLVMMemoryBufferRef as MemoryBufferRef;
pub use LLVMModuleRef as ModuleRef;
pub use LLVMObjectFileRef as ObjectFileRef;
pub use LLVMOpcode as Opcode;
pub use LLVMPassManagerBuilderRef as PassManagerBuilderRef;
pub use LLVMPassManagerRef as PassManagerRef;
pub use LLVMRealPredicate as RealPredicate;
pub use LLVMRealPredicate::LLVMRealOEQ as RealOEQ;
pub use LLVMRealPredicate::LLVMRealOGE as RealOGE;
pub use LLVMRealPredicate::LLVMRealOGT as RealOGT;
pub use LLVMRealPredicate::LLVMRealOLE as RealOLE;
pub use LLVMRealPredicate::LLVMRealOLT as RealOLT;
pub use LLVMRealPredicate::LLVMRealULT as RealULT;
pub use LLVMRealPredicate::LLVMRealUNE as RealUNE;
pub use LLVMRustArchiveChildRef as ArchiveChildRef;
pub use LLVMRustArchiveIteratorRef as ArchiveIteratorRef;
pub use LLVMRustArchiveKind as ArchiveKind;
pub use LLVMRustArchiveRef as ArchiveRef;
pub use LLVMRustAsmDialect as AsmDialect;
pub use LLVMRustAttribute as Attribute;
pub use LLVMRustCodeGenOptLevel as CodeGenOptLevel;
pub use LLVMRustCodeModel as CodeModel;
pub use LLVMRustDiagnosticKind as DiagnosticKind;
pub use LLVMRustFileType as FileType;
pub use LLVMRustLinkage as Linkage;
pub use LLVMRustModuleBuffer as ModuleBuffer;
pub use LLVMRustOperandBundleDefRef as OperandBundleDefRef;
pub use LLVMRustPassKind as PassKind;
pub use LLVMRustRelocMode as RelocMode;
pub use LLVMRustSynchronizationScope as SynchronizationScope;
pub use LLVMRustThinLTOBuffer as ThinLTOBuffer;
pub use LLVMRustThinLTOData as ThinLTOData;
pub use LLVMRustThinLTOModule as ThinLTOModule;
pub use LLVMRustVisibility as Visibility;
pub use LLVMSMDiagnosticRef as SMDiagnosticRef;
pub use LLVMSectionIteratorRef as SectionIteratorRef;
pub use LLVMTargetDataRef as TargetDataRef;
pub use LLVMTargetMachineRef as TargetMachineRef;
pub use LLVMThreadLocalMode as ThreadLocalMode;
pub use LLVMTwineRef as TwineRef;
pub use LLVMTypeKind as TypeKind;
pub use LLVMTypeKind::LLVMDoubleTypeKind as Double;
pub use LLVMTypeKind::LLVMFP128TypeKind as FP128;
pub use LLVMTypeKind::LLVMFloatTypeKind as Float;
pub use LLVMTypeKind::LLVMPPC_FP128TypeKind as PPC_FP128;
pub use LLVMTypeKind::LLVMVectorTypeKind as Vector;
pub use LLVMTypeKind::LLVMX86_FP80TypeKind as X86_FP80;
pub use LLVMTypeRef as TypeRef;
pub use LLVMValueRef as ValueRef;
pub use llvm_CallingConv__bindgen_ty_1 as CallConv;
pub use llvm_LLVMContext__bindgen_ty_1 as MetadataType;

pub type LLVMSMDiagnosticRef = *const llvm_SMDiagnostic;

pub const True: Bool = 1 as Bool;
pub const False: Bool = 0 as Bool;

//#[allow(dead_code)]
//#[allow(non_camel_case_types)]
//#[allow(non_snake_case)]
//#[allow(non_upper_case_globals)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub mod debuginfo {
    use super::LLVMMetadataRef as MetadataRef;

    pub use super::LLVMRustDIBuilderRef as DIBuilderRef;

    pub type DIDescriptor = MetadataRef;
    pub type DIScope = DIDescriptor;
    pub type DILocation = DIDescriptor;
    pub type DIFile = DIScope;
    pub type DILexicalBlock = DIScope;
    pub type DISubprogram = DIScope;
    pub type DINameSpace = DIScope;
    pub type DIType = DIDescriptor;
    pub type DIBasicType = DIType;
    pub type DIDerivedType = DIType;
    pub type DICompositeType = DIDerivedType;
    pub type DIVariable = DIDescriptor;
    // FIXME: Rename 'DIGlobalVariable' to 'DIGlobalVariableExpression'
    // once support for LLVM 3.9 is dropped.
    //
    // This method was changed in this LLVM patch:
    // https://reviews.llvm.org/D26769
    pub type DIGlobalVariable = DIDescriptor;
    pub type DIArray = DIDescriptor;
    pub type DISubrange = DIDescriptor;
    pub type DIEnumerator = DIDescriptor;
    pub type DITemplateTypeParameter = DIDescriptor;
}
