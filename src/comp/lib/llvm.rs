import core::{vec, str, option};
import str::sbuf;

import ctypes::{c_int, c_uint, unsigned, longlong, ulonglong};

type Opcode = u32;
type Bool = unsigned;
const True: Bool = 1u32;
const False: Bool = 0u32;

// Consts for the LLVM CallConv type, pre-cast to uint.

enum CallConv {
    CCallConv = 0,
    FastCallConv = 8,
    ColdCallConv = 9,
    X86StdcallCallConv = 64,
    X86FastcallCallConv = 65,
}

enum Visibility {
    LLVMDefaultVisibility = 0,
    HiddenVisibility = 1,
    ProtectedVisibility = 2,
}

enum Linkage {
    ExternalLinkage = 0,
    AvailableExternallyLinkage = 1,
    LinkOnceAnyLinkage = 2,
    LinkOnceODRLinkage = 3,
    WeakAnyLinkage = 4,
    WeakODRLinkage = 5,
    AppendingLinkage = 6,
    InternalLinkage = 7,
    PrivateLinkage = 8,
    DLLImportLinkage = 9,
    DLLExportLinkage = 10,
    ExternalWeakLinkage = 11,
    GhostLinkage = 12,
    CommonLinkage = 13,
    LinkerPrivateLinkage = 14,
    LinkerPrivateWeakLinkage = 15,
    LinkerPrivateWeakDefAutoLinkage = 16,
}

enum Attribute {
    ZExtAttribute = 1,
    SExtAttribute = 2,
    NoReturnAttribute = 4,
    InRegAttribute = 8,
    StructRetAttribute = 16,
    NoUnwindAttribute = 32,
    NoAliasAttribute = 64,
    ByValAttribute = 128,
    NestAttribute = 256,
    ReadNoneAttribute = 512,
    ReadOnlyAttribute = 1024,
    NoInlineAttribute = 2048,
    AlwaysInlineAttribute = 4096,
    OptimizeForSizeAttribute = 8192,
    StackProtectAttribute = 16384,
    StackProtectReqAttribute = 32768,
    // 31 << 16
    AlignmentAttribute = 2031616,
    NoCaptureAttribute = 2097152,
    NoRedZoneAttribute = 4194304,
    NoImplicitFloatAttribute = 8388608,
    NakedAttribute = 16777216,
    InlineHintAttribute = 33554432,
    // 7 << 26
    StackAttribute = 469762048,
    ReturnsTwiceAttribute = 536870912,
    // 1 << 30
    UWTableAttribute = 1073741824,
    NonLazyBindAttribute = 2147483648,
}

// Consts for the LLVM IntPredicate type, pre-cast to uint.
// FIXME: as above.

enum IntPredicate {
    IntEQ = 32,
    IntNE = 33,
    IntUGT = 34,
    IntUGE = 35,
    IntULT = 36,
    IntULE = 37,
    IntSGT = 38,
    IntSGE = 39,
    IntSLT = 40,
    IntSLE = 41,
}

// Consts for the LLVM RealPredicate type, pre-case to uint.
// FIXME: as above.

enum RealPredicate {
    RealOEQ = 1,
    RealOGT = 2,
    RealOGE = 3,
    RealOLT = 4,
    RealOLE = 5,
    RealONE = 6,
    RealORD = 7,
    RealUNO = 8,
    RealUEQ = 9,
    RealUGT = 10,
    RealUGE = 11,
    RealULT = 12,
    RealULE = 13,
    RealUNE = 14,
}

// Opaque pointer types
enum Module_opaque {}
type ModuleRef = *Module_opaque;
enum Context_opaque {}
type ContextRef = *Context_opaque;
enum Type_opaque {}
type TypeRef = *Type_opaque;
enum Value_opaque {}
type ValueRef = *Value_opaque;
enum BasicBlock_opaque {}
type BasicBlockRef = *BasicBlock_opaque;
enum Builder_opaque {}
type BuilderRef = *Builder_opaque;
enum MemoryBuffer_opaque {}
type MemoryBufferRef = *MemoryBuffer_opaque;
enum PassManager_opaque {}
type PassManagerRef = *PassManager_opaque;
enum PassManagerBuilder_opaque {}
type PassManagerBuilderRef = *PassManagerBuilder_opaque;
enum Use_opaque {}
type UseRef = *Use_opaque;
enum TargetData_opaque {}
type TargetDataRef = *TargetData_opaque;
enum ObjectFile_opaque {}
type ObjectFileRef = *ObjectFile_opaque;
enum SectionIterator_opaque {}
type SectionIteratorRef = *SectionIterator_opaque;

#[link_args = "-Lrustllvm"]
#[link_name = "rustllvm"]
#[abi = "cdecl"]
native mod llvm {
    /* Create and destroy contexts. */
    fn LLVMContextCreate() -> ContextRef;
    fn LLVMGetGlobalContext() -> ContextRef;
    fn LLVMContextDispose(C: ContextRef);
    fn LLVMGetMDKindIDInContext(C: ContextRef, Name: sbuf, SLen: unsigned) ->
       unsigned;
    fn LLVMGetMDKindID(Name: sbuf, SLen: unsigned) -> unsigned;

    /* Create and destroy modules. */
    fn LLVMModuleCreateWithNameInContext(ModuleID: sbuf, C: ContextRef) ->
       ModuleRef;
    fn LLVMDisposeModule(M: ModuleRef);

    /** Data layout. See Module::getDataLayout. */
    fn LLVMGetDataLayout(M: ModuleRef) -> sbuf;
    fn LLVMSetDataLayout(M: ModuleRef, Triple: sbuf);

    /** Target triple. See Module::getTargetTriple. */
    fn LLVMGetTarget(M: ModuleRef) -> sbuf;
    fn LLVMSetTarget(M: ModuleRef, Triple: sbuf);

    /** See Module::dump. */
    fn LLVMDumpModule(M: ModuleRef);

    /** See Module::setModuleInlineAsm. */
    fn LLVMSetModuleInlineAsm(M: ModuleRef, Asm: sbuf);

    /** See llvm::LLVMTypeKind::getTypeID. */

    // FIXME: returning int rather than TypeKind because
    // we directly inspect the values, and casting from
    // a native doesn't work yet (only *to* a native).

    fn LLVMGetTypeKind(Ty: TypeRef) -> c_int;

    /** See llvm::LLVMType::getContext. */
    fn LLVMGetTypeContext(Ty: TypeRef) -> ContextRef;

    /* Operations on integer types */
    fn LLVMInt1TypeInContext(C: ContextRef) -> TypeRef;
    fn LLVMInt8TypeInContext(C: ContextRef) -> TypeRef;
    fn LLVMInt16TypeInContext(C: ContextRef) -> TypeRef;
    fn LLVMInt32TypeInContext(C: ContextRef) -> TypeRef;
    fn LLVMInt64TypeInContext(C: ContextRef) -> TypeRef;
    fn LLVMIntTypeInContext(C: ContextRef, NumBits: unsigned) -> TypeRef;

    fn LLVMInt1Type() -> TypeRef;
    fn LLVMInt8Type() -> TypeRef;
    fn LLVMInt16Type() -> TypeRef;
    fn LLVMInt32Type() -> TypeRef;
    fn LLVMInt64Type() -> TypeRef;
    fn LLVMIntType(NumBits: unsigned) -> TypeRef;
    fn LLVMGetIntTypeWidth(IntegerTy: TypeRef) -> unsigned;

    /* Operations on real types */
    fn LLVMFloatTypeInContext(C: ContextRef) -> TypeRef;
    fn LLVMDoubleTypeInContext(C: ContextRef) -> TypeRef;
    fn LLVMX86FP80TypeInContext(C: ContextRef) -> TypeRef;
    fn LLVMFP128TypeInContext(C: ContextRef) -> TypeRef;
    fn LLVMPPCFP128TypeInContext(C: ContextRef) -> TypeRef;

    fn LLVMFloatType() -> TypeRef;
    fn LLVMDoubleType() -> TypeRef;
    fn LLVMX86FP80Type() -> TypeRef;
    fn LLVMFP128Type() -> TypeRef;
    fn LLVMPPCFP128Type() -> TypeRef;

    /* Operations on function types */
    fn LLVMFunctionType(ReturnType: TypeRef, ParamTypes: *TypeRef,
                        ParamCount: unsigned, IsVarArg: Bool) -> TypeRef;
    fn LLVMIsFunctionVarArg(FunctionTy: TypeRef) -> Bool;
    fn LLVMGetReturnType(FunctionTy: TypeRef) -> TypeRef;
    fn LLVMCountParamTypes(FunctionTy: TypeRef) -> unsigned;
    fn LLVMGetParamTypes(FunctionTy: TypeRef, Dest: *TypeRef);

    /* Operations on struct types */
    fn LLVMStructTypeInContext(C: ContextRef, ElementTypes: *TypeRef,
                               ElementCount: unsigned,
                               Packed: Bool) -> TypeRef;
    fn LLVMStructType(ElementTypes: *TypeRef, ElementCount: unsigned,
                      Packed: Bool) -> TypeRef;
    fn LLVMCountStructElementTypes(StructTy: TypeRef) -> unsigned;
    fn LLVMGetStructElementTypes(StructTy: TypeRef, Dest: *TypeRef);
    fn LLVMIsPackedStruct(StructTy: TypeRef) -> Bool;

    /* Operations on array, pointer, and vector types (sequence types) */
    fn LLVMArrayType(ElementType: TypeRef,
                     ElementCount: unsigned) -> TypeRef;
    fn LLVMPointerType(ElementType: TypeRef,
                       AddressSpace: unsigned) -> TypeRef;
    fn LLVMVectorType(ElementType: TypeRef,
                      ElementCount: unsigned) -> TypeRef;

    fn LLVMGetElementType(Ty: TypeRef) -> TypeRef;
    fn LLVMGetArrayLength(ArrayTy: TypeRef) -> unsigned;
    fn LLVMGetPointerAddressSpace(PointerTy: TypeRef) -> unsigned;
    fn LLVMGetVectorSize(VectorTy: TypeRef) -> unsigned;

    /* Operations on other types */
    fn LLVMVoidTypeInContext(C: ContextRef) -> TypeRef;
    fn LLVMLabelTypeInContext(C: ContextRef) -> TypeRef;
    fn LLVMMetadataTypeInContext(C: ContextRef) -> TypeRef;

    fn LLVMVoidType() -> TypeRef;
    fn LLVMLabelType() -> TypeRef;
    fn LLVMMetadataType() -> TypeRef;

    /* Operations on all values */
    fn LLVMTypeOf(Val: ValueRef) -> TypeRef;
    fn LLVMGetValueName(Val: ValueRef) -> sbuf;
    fn LLVMSetValueName(Val: ValueRef, Name: sbuf);
    fn LLVMDumpValue(Val: ValueRef);
    fn LLVMReplaceAllUsesWith(OldVal: ValueRef, NewVal: ValueRef);
    fn LLVMHasMetadata(Val: ValueRef) -> c_int;
    fn LLVMGetMetadata(Val: ValueRef, KindID: unsigned) -> ValueRef;
    fn LLVMSetMetadata(Val: ValueRef, KindID: unsigned, Node: ValueRef);

    /* Operations on Uses */
    fn LLVMGetFirstUse(Val: ValueRef) -> UseRef;
    fn LLVMGetNextUse(U: UseRef) -> UseRef;
    fn LLVMGetUser(U: UseRef) -> ValueRef;
    fn LLVMGetUsedValue(U: UseRef) -> ValueRef;

    /* Operations on Users */
    fn LLVMGetOperand(Val: ValueRef, Index: unsigned) -> ValueRef;
    fn LLVMSetOperand(Val: ValueRef, Index: unsigned, Op: ValueRef);

    /* Operations on constants of any type */
    fn LLVMConstNull(Ty: TypeRef) -> ValueRef;
    /* all zeroes */
    fn LLVMConstAllOnes(Ty: TypeRef) -> ValueRef;
    /* only for int/vector */
    fn LLVMGetUndef(Ty: TypeRef) -> ValueRef;
    fn LLVMIsConstant(Val: ValueRef) -> Bool;
    fn LLVMIsNull(Val: ValueRef) -> Bool;
    fn LLVMIsUndef(Val: ValueRef) -> Bool;
    fn LLVMConstPointerNull(Ty: TypeRef) -> ValueRef;

    /* Operations on metadata */
    fn LLVMMDStringInContext(C: ContextRef, Str: sbuf, SLen: unsigned) ->
       ValueRef;
    fn LLVMMDString(Str: sbuf, SLen: unsigned) -> ValueRef;
    fn LLVMMDNodeInContext(C: ContextRef, Vals: *ValueRef, Count: unsigned) ->
       ValueRef;
    fn LLVMMDNode(Vals: *ValueRef, Count: unsigned) -> ValueRef;
    fn LLVMAddNamedMetadataOperand(M: ModuleRef, Str: sbuf,
                                   Val: ValueRef);

    /* Operations on scalar constants */
    fn LLVMConstInt(IntTy: TypeRef, N: ulonglong, SignExtend: Bool) ->
       ValueRef;
    // FIXME: radix is actually u8, but our native layer can't handle this
    // yet.  lucky for us we're little-endian. Small miracles.
    fn LLVMConstIntOfString(IntTy: TypeRef, Text: sbuf, Radix: c_int) ->
       ValueRef;
    fn LLVMConstIntOfStringAndSize(IntTy: TypeRef, Text: sbuf, SLen: unsigned,
                                   Radix: u8) -> ValueRef;
    fn LLVMConstReal(RealTy: TypeRef, N: f64) -> ValueRef;
    fn LLVMConstRealOfString(RealTy: TypeRef, Text: sbuf) -> ValueRef;
    fn LLVMConstRealOfStringAndSize(RealTy: TypeRef, Text: sbuf,
                                    SLen: unsigned) -> ValueRef;
    fn LLVMConstIntGetZExtValue(ConstantVal: ValueRef) -> ulonglong;
    fn LLVMConstIntGetSExtValue(ConstantVal: ValueRef) -> longlong;


    /* Operations on composite constants */
    fn LLVMConstStringInContext(C: ContextRef, Str: sbuf, Length: unsigned,
                                DontNullTerminate: Bool) -> ValueRef;
    fn LLVMConstStructInContext(C: ContextRef, ConstantVals: *ValueRef,
                                Count: unsigned, Packed: Bool) -> ValueRef;

    fn LLVMConstString(Str: sbuf, Length: unsigned,
                       DontNullTerminate: Bool) -> ValueRef;
    fn LLVMConstArray(ElementTy: TypeRef, ConstantVals: *ValueRef,
                      Length: unsigned) -> ValueRef;
    fn LLVMConstStruct(ConstantVals: *ValueRef,
                       Count: unsigned, Packed: Bool) -> ValueRef;
    fn LLVMConstVector(ScalarConstantVals: *ValueRef,
                       Size: unsigned) -> ValueRef;

    /* Constant expressions */
    fn LLVMAlignOf(Ty: TypeRef) -> ValueRef;
    fn LLVMSizeOf(Ty: TypeRef) -> ValueRef;
    fn LLVMConstNeg(ConstantVal: ValueRef) -> ValueRef;
    fn LLVMConstNSWNeg(ConstantVal: ValueRef) -> ValueRef;
    fn LLVMConstNUWNeg(ConstantVal: ValueRef) -> ValueRef;
    fn LLVMConstFNeg(ConstantVal: ValueRef) -> ValueRef;
    fn LLVMConstNot(ConstantVal: ValueRef) -> ValueRef;
    fn LLVMConstAdd(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    fn LLVMConstNSWAdd(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstNUWAdd(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstFAdd(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstSub(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    fn LLVMConstNSWSub(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstNUWSub(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstFSub(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstMul(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    fn LLVMConstNSWMul(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstNUWMul(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstFMul(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstUDiv(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstSDiv(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstExactSDiv(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstFDiv(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstURem(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstSRem(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstFRem(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstAnd(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    fn LLVMConstOr(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    fn LLVMConstXor(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    fn LLVMConstShl(LHSConstant: ValueRef, RHSConstant: ValueRef) -> ValueRef;
    fn LLVMConstLShr(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstAShr(LHSConstant: ValueRef, RHSConstant: ValueRef) ->
       ValueRef;
    fn LLVMConstGEP(ConstantVal: ValueRef, ConstantIndices: *uint,
                    NumIndices: unsigned) -> ValueRef;
    fn LLVMConstInBoundsGEP(ConstantVal: ValueRef, ConstantIndices: *uint,
                            NumIndices: unsigned) -> ValueRef;
    fn LLVMConstTrunc(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    fn LLVMConstSExt(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    fn LLVMConstZExt(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    fn LLVMConstFPTrunc(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    fn LLVMConstFPExt(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    fn LLVMConstUIToFP(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    fn LLVMConstSIToFP(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    fn LLVMConstFPToUI(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    fn LLVMConstFPToSI(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    fn LLVMConstPtrToInt(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    fn LLVMConstIntToPtr(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    fn LLVMConstBitCast(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    fn LLVMConstZExtOrBitCast(ConstantVal: ValueRef, ToType: TypeRef) ->
       ValueRef;
    fn LLVMConstSExtOrBitCast(ConstantVal: ValueRef, ToType: TypeRef) ->
       ValueRef;
    fn LLVMConstTruncOrBitCast(ConstantVal: ValueRef, ToType: TypeRef) ->
       ValueRef;
    fn LLVMConstPointerCast(ConstantVal: ValueRef, ToType: TypeRef) ->
       ValueRef;
    fn LLVMConstIntCast(ConstantVal: ValueRef, ToType: TypeRef,
                        isSigned: Bool) -> ValueRef;
    fn LLVMConstFPCast(ConstantVal: ValueRef, ToType: TypeRef) -> ValueRef;
    fn LLVMConstSelect(ConstantCondition: ValueRef, ConstantIfTrue: ValueRef,
                       ConstantIfFalse: ValueRef) -> ValueRef;
    fn LLVMConstExtractElement(VectorConstant: ValueRef,
                               IndexConstant: ValueRef) -> ValueRef;
    fn LLVMConstInsertElement(VectorConstant: ValueRef,
                              ElementValueConstant: ValueRef,
                              IndexConstant: ValueRef) -> ValueRef;
    fn LLVMConstShuffleVector(VectorAConstant: ValueRef,
                              VectorBConstant: ValueRef,
                              MaskConstant: ValueRef) -> ValueRef;
    fn LLVMConstExtractValue(AggConstant: ValueRef, IdxList: *uint,
                             NumIdx: unsigned) -> ValueRef;
    fn LLVMConstInsertValue(AggConstant: ValueRef,
                            ElementValueConstant: ValueRef, IdxList: *uint,
                            NumIdx: unsigned) -> ValueRef;
    fn LLVMConstInlineAsm(Ty: TypeRef, AsmString: sbuf, Constraints: sbuf,
                          HasSideEffects: Bool, IsAlignStack: Bool) ->
       ValueRef;
    fn LLVMBlockAddress(F: ValueRef, BB: BasicBlockRef) -> ValueRef;



    /* Operations on global variables, functions, and aliases (globals) */
    fn LLVMGetGlobalParent(Global: ValueRef) -> ModuleRef;
    fn LLVMIsDeclaration(Global: ValueRef) -> Bool;
    fn LLVMGetLinkage(Global: ValueRef) -> unsigned;
    fn LLVMSetLinkage(Global: ValueRef, Link: unsigned);
    fn LLVMGetSection(Global: ValueRef) -> sbuf;
    fn LLVMSetSection(Global: ValueRef, Section: sbuf);
    fn LLVMGetVisibility(Global: ValueRef) -> unsigned;
    fn LLVMSetVisibility(Global: ValueRef, Viz: unsigned);
    fn LLVMGetAlignment(Global: ValueRef) -> unsigned;
    fn LLVMSetAlignment(Global: ValueRef, Bytes: unsigned);


    /* Operations on global variables */
    fn LLVMAddGlobal(M: ModuleRef, Ty: TypeRef, Name: sbuf) -> ValueRef;
    fn LLVMAddGlobalInAddressSpace(M: ModuleRef, Ty: TypeRef, Name: sbuf,
                                   AddressSpace: unsigned) -> ValueRef;
    fn LLVMGetNamedGlobal(M: ModuleRef, Name: sbuf) -> ValueRef;
    fn LLVMGetFirstGlobal(M: ModuleRef) -> ValueRef;
    fn LLVMGetLastGlobal(M: ModuleRef) -> ValueRef;
    fn LLVMGetNextGlobal(GlobalVar: ValueRef) -> ValueRef;
    fn LLVMGetPreviousGlobal(GlobalVar: ValueRef) -> ValueRef;
    fn LLVMDeleteGlobal(GlobalVar: ValueRef);
    fn LLVMGetInitializer(GlobalVar: ValueRef) -> ValueRef;
    fn LLVMSetInitializer(GlobalVar: ValueRef, ConstantVal: ValueRef);
    fn LLVMIsThreadLocal(GlobalVar: ValueRef) -> Bool;
    fn LLVMSetThreadLocal(GlobalVar: ValueRef, IsThreadLocal: Bool);
    fn LLVMIsGlobalConstant(GlobalVar: ValueRef) -> Bool;
    fn LLVMSetGlobalConstant(GlobalVar: ValueRef, IsConstant: Bool);

    /* Operations on aliases */
    fn LLVMAddAlias(M: ModuleRef, Ty: TypeRef, Aliasee: ValueRef, Name: sbuf)
       -> ValueRef;

    /* Operations on functions */
    fn LLVMAddFunction(M: ModuleRef, Name: sbuf, FunctionTy: TypeRef) ->
       ValueRef;
    fn LLVMGetNamedFunction(M: ModuleRef, Name: sbuf) -> ValueRef;
    fn LLVMGetFirstFunction(M: ModuleRef) -> ValueRef;
    fn LLVMGetLastFunction(M: ModuleRef) -> ValueRef;
    fn LLVMGetNextFunction(Fn: ValueRef) -> ValueRef;
    fn LLVMGetPreviousFunction(Fn: ValueRef) -> ValueRef;
    fn LLVMDeleteFunction(Fn: ValueRef);
    fn LLVMGetOrInsertFunction(M: ModuleRef, Name: sbuf, FunctionTy: TypeRef)
       -> ValueRef;
    fn LLVMGetIntrinsicID(Fn: ValueRef) -> unsigned;
    fn LLVMGetFunctionCallConv(Fn: ValueRef) -> unsigned;
    fn LLVMSetFunctionCallConv(Fn: ValueRef, CC: unsigned);
    fn LLVMGetGC(Fn: ValueRef) -> sbuf;
    fn LLVMSetGC(Fn: ValueRef, Name: sbuf);
    fn LLVMAddFunctionAttr(Fn: ValueRef, PA: unsigned, HighPA: unsigned);
    fn LLVMGetFunctionAttr(Fn: ValueRef) -> unsigned;
    fn LLVMRemoveFunctionAttr(Fn: ValueRef, PA: unsigned, HighPA: unsigned);

    /* Operations on parameters */
    fn LLVMCountParams(Fn: ValueRef) -> unsigned;
    fn LLVMGetParams(Fn: ValueRef, Params: *ValueRef);
    fn LLVMGetParam(Fn: ValueRef, Index: unsigned) -> ValueRef;
    fn LLVMGetParamParent(Inst: ValueRef) -> ValueRef;
    fn LLVMGetFirstParam(Fn: ValueRef) -> ValueRef;
    fn LLVMGetLastParam(Fn: ValueRef) -> ValueRef;
    fn LLVMGetNextParam(Arg: ValueRef) -> ValueRef;
    fn LLVMGetPreviousParam(Arg: ValueRef) -> ValueRef;
    fn LLVMAddAttribute(Arg: ValueRef, PA: unsigned);
    fn LLVMRemoveAttribute(Arg: ValueRef, PA: unsigned);
    fn LLVMGetAttribute(Arg: ValueRef) -> unsigned;
    fn LLVMSetParamAlignment(Arg: ValueRef, align: unsigned);

    /* Operations on basic blocks */
    fn LLVMBasicBlockAsValue(BB: BasicBlockRef) -> ValueRef;
    fn LLVMValueIsBasicBlock(Val: ValueRef) -> Bool;
    fn LLVMValueAsBasicBlock(Val: ValueRef) -> BasicBlockRef;
    fn LLVMGetBasicBlockParent(BB: BasicBlockRef) -> ValueRef;
    fn LLVMCountBasicBlocks(Fn: ValueRef) -> unsigned;
    fn LLVMGetBasicBlocks(Fn: ValueRef, BasicBlocks: *ValueRef);
    fn LLVMGetFirstBasicBlock(Fn: ValueRef) -> BasicBlockRef;
    fn LLVMGetLastBasicBlock(Fn: ValueRef) -> BasicBlockRef;
    fn LLVMGetNextBasicBlock(BB: BasicBlockRef) -> BasicBlockRef;
    fn LLVMGetPreviousBasicBlock(BB: BasicBlockRef) -> BasicBlockRef;
    fn LLVMGetEntryBasicBlock(Fn: ValueRef) -> BasicBlockRef;

    fn LLVMAppendBasicBlockInContext(C: ContextRef, Fn: ValueRef, Name: sbuf)
       -> BasicBlockRef;
    fn LLVMInsertBasicBlockInContext(C: ContextRef, BB: BasicBlockRef,
                                     Name: sbuf) -> BasicBlockRef;

    fn LLVMAppendBasicBlock(Fn: ValueRef, Name: sbuf) -> BasicBlockRef;
    fn LLVMInsertBasicBlock(InsertBeforeBB: BasicBlockRef, Name: sbuf) ->
       BasicBlockRef;
    fn LLVMDeleteBasicBlock(BB: BasicBlockRef);

    /* Operations on instructions */
    fn LLVMGetInstructionParent(Inst: ValueRef) -> BasicBlockRef;
    fn LLVMGetFirstInstruction(BB: BasicBlockRef) -> ValueRef;
    fn LLVMGetLastInstruction(BB: BasicBlockRef) -> ValueRef;
    fn LLVMGetNextInstruction(Inst: ValueRef) -> ValueRef;
    fn LLVMGetPreviousInstruction(Inst: ValueRef) -> ValueRef;

    /* Operations on call sites */
    fn LLVMSetInstructionCallConv(Instr: ValueRef, CC: unsigned);
    fn LLVMGetInstructionCallConv(Instr: ValueRef) -> unsigned;
    fn LLVMAddInstrAttribute(Instr: ValueRef, index: unsigned, IA: unsigned);
    fn LLVMRemoveInstrAttribute(Instr: ValueRef, index: unsigned,
                                IA: unsigned);
    fn LLVMSetInstrParamAlignment(Instr: ValueRef, index: unsigned,
                                  align: unsigned);

    /* Operations on call instructions (only) */
    fn LLVMIsTailCall(CallInst: ValueRef) -> Bool;
    fn LLVMSetTailCall(CallInst: ValueRef, IsTailCall: Bool);

    /* Operations on phi nodes */
    fn LLVMAddIncoming(PhiNode: ValueRef, IncomingValues: *ValueRef,
                       IncomingBlocks: *BasicBlockRef, Count: unsigned);
    fn LLVMCountIncoming(PhiNode: ValueRef) -> unsigned;
    fn LLVMGetIncomingValue(PhiNode: ValueRef, Index: unsigned) -> ValueRef;
    fn LLVMGetIncomingBlock(PhiNode: ValueRef,
                            Index: unsigned) -> BasicBlockRef;

    /* Instruction builders */
    fn LLVMCreateBuilderInContext(C: ContextRef) -> BuilderRef;
    fn LLVMCreateBuilder() -> BuilderRef;
    fn LLVMPositionBuilder(Builder: BuilderRef, Block: BasicBlockRef,
                           Instr: ValueRef);
    fn LLVMPositionBuilderBefore(Builder: BuilderRef, Instr: ValueRef);
    fn LLVMPositionBuilderAtEnd(Builder: BuilderRef, Block: BasicBlockRef);
    fn LLVMGetInsertBlock(Builder: BuilderRef) -> BasicBlockRef;
    fn LLVMClearInsertionPosition(Builder: BuilderRef);
    fn LLVMInsertIntoBuilder(Builder: BuilderRef, Instr: ValueRef);
    fn LLVMInsertIntoBuilderWithName(Builder: BuilderRef, Instr: ValueRef,
                                     Name: sbuf);
    fn LLVMDisposeBuilder(Builder: BuilderRef);

    /* Metadata */
    fn LLVMSetCurrentDebugLocation(Builder: BuilderRef, L: ValueRef);
    fn LLVMGetCurrentDebugLocation(Builder: BuilderRef) -> ValueRef;
    fn LLVMSetInstDebugLocation(Builder: BuilderRef, Inst: ValueRef);

    /* Terminators */
    fn LLVMBuildRetVoid(B: BuilderRef) -> ValueRef;
    fn LLVMBuildRet(B: BuilderRef, V: ValueRef) -> ValueRef;
    fn LLVMBuildAggregateRet(B: BuilderRef, RetVals: *ValueRef,
                             N: unsigned) -> ValueRef;
    fn LLVMBuildBr(B: BuilderRef, Dest: BasicBlockRef) -> ValueRef;
    fn LLVMBuildCondBr(B: BuilderRef, If: ValueRef, Then: BasicBlockRef,
                       Else: BasicBlockRef) -> ValueRef;
    fn LLVMBuildSwitch(B: BuilderRef, V: ValueRef, Else: BasicBlockRef,
                       NumCases: unsigned) -> ValueRef;
    fn LLVMBuildIndirectBr(B: BuilderRef, Addr: ValueRef,
                           NumDests: unsigned) -> ValueRef;
    fn LLVMBuildInvoke(B: BuilderRef, Fn: ValueRef, Args: *ValueRef,
                       NumArgs: unsigned, Then: BasicBlockRef,
                       Catch: BasicBlockRef, Name: sbuf) -> ValueRef;
    fn LLVMBuildLandingPad(B: BuilderRef, Ty: TypeRef, PersFn: ValueRef,
                           NumClauses: unsigned, Name: sbuf) -> ValueRef;
    fn LLVMBuildResume(B: BuilderRef, Exn: ValueRef) -> ValueRef;
    fn LLVMBuildUnreachable(B: BuilderRef) -> ValueRef;

    /* Add a case to the switch instruction */
    fn LLVMAddCase(Switch: ValueRef, OnVal: ValueRef, Dest: BasicBlockRef);

    /* Add a destination to the indirectbr instruction */
    fn LLVMAddDestination(IndirectBr: ValueRef, Dest: BasicBlockRef);

    /* Add a clause to the landing pad instruction */
    fn LLVMAddClause(LandingPad: ValueRef, ClauseVal: ValueRef);

    /* Set the cleanup on a landing pad instruction */
    fn LLVMSetCleanup(LandingPad: ValueRef, Val: Bool);

    /* Arithmetic */
    fn LLVMBuildAdd(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildNSWAdd(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                       Name: sbuf) -> ValueRef;
    fn LLVMBuildNUWAdd(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                       Name: sbuf) -> ValueRef;
    fn LLVMBuildFAdd(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildSub(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildNSWSub(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                       Name: sbuf) -> ValueRef;
    fn LLVMBuildNUWSub(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                       Name: sbuf) -> ValueRef;
    fn LLVMBuildFSub(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildMul(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildNSWMul(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                       Name: sbuf) -> ValueRef;
    fn LLVMBuildNUWMul(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                       Name: sbuf) -> ValueRef;
    fn LLVMBuildFMul(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildUDiv(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildSDiv(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildExactSDiv(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                          Name: sbuf) -> ValueRef;
    fn LLVMBuildFDiv(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildURem(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildSRem(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildFRem(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildShl(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildLShr(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildAShr(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildAnd(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildOr(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf) ->
       ValueRef;
    fn LLVMBuildXor(B: BuilderRef, LHS: ValueRef, RHS: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildBinOp(B: BuilderRef, Op: Opcode, LHS: ValueRef, RHS: ValueRef,
                      Name: sbuf) -> ValueRef;
    fn LLVMBuildNeg(B: BuilderRef, V: ValueRef, Name: sbuf) -> ValueRef;
    fn LLVMBuildNSWNeg(B: BuilderRef, V: ValueRef, Name: sbuf) -> ValueRef;
    fn LLVMBuildNUWNeg(B: BuilderRef, V: ValueRef, Name: sbuf) -> ValueRef;
    fn LLVMBuildFNeg(B: BuilderRef, V: ValueRef, Name: sbuf) -> ValueRef;
    fn LLVMBuildNot(B: BuilderRef, V: ValueRef, Name: sbuf) -> ValueRef;

    /* Memory */
    fn LLVMBuildMalloc(B: BuilderRef, Ty: TypeRef, Name: sbuf) -> ValueRef;
    fn LLVMBuildArrayMalloc(B: BuilderRef, Ty: TypeRef, Val: ValueRef,
                            Name: sbuf) -> ValueRef;
    fn LLVMBuildAlloca(B: BuilderRef, Ty: TypeRef, Name: sbuf) -> ValueRef;
    fn LLVMBuildArrayAlloca(B: BuilderRef, Ty: TypeRef, Val: ValueRef,
                            Name: sbuf) -> ValueRef;
    fn LLVMBuildFree(B: BuilderRef, PointerVal: ValueRef) -> ValueRef;
    fn LLVMBuildLoad(B: BuilderRef, PointerVal: ValueRef, Name: sbuf) ->
       ValueRef;
    fn LLVMBuildStore(B: BuilderRef, Val: ValueRef, Ptr: ValueRef) ->
       ValueRef;
    fn LLVMBuildGEP(B: BuilderRef, Pointer: ValueRef, Indices: *ValueRef,
                    NumIndices: unsigned, Name: sbuf) -> ValueRef;
    fn LLVMBuildInBoundsGEP(B: BuilderRef, Pointer: ValueRef,
                            Indices: *ValueRef, NumIndices: unsigned,
                            Name: sbuf)
       -> ValueRef;
    fn LLVMBuildStructGEP(B: BuilderRef, Pointer: ValueRef, Idx: unsigned,
                          Name: sbuf) -> ValueRef;
    fn LLVMBuildGlobalString(B: BuilderRef, Str: sbuf, Name: sbuf) ->
       ValueRef;
    fn LLVMBuildGlobalStringPtr(B: BuilderRef, Str: sbuf, Name: sbuf) ->
       ValueRef;

    /* Casts */
    fn LLVMBuildTrunc(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                      Name: sbuf) -> ValueRef;
    fn LLVMBuildZExt(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                     Name: sbuf) -> ValueRef;
    fn LLVMBuildSExt(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                     Name: sbuf) -> ValueRef;
    fn LLVMBuildFPToUI(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                       Name: sbuf) -> ValueRef;
    fn LLVMBuildFPToSI(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                       Name: sbuf) -> ValueRef;
    fn LLVMBuildUIToFP(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                       Name: sbuf) -> ValueRef;
    fn LLVMBuildSIToFP(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                       Name: sbuf) -> ValueRef;
    fn LLVMBuildFPTrunc(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                        Name: sbuf) -> ValueRef;
    fn LLVMBuildFPExt(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                      Name: sbuf) -> ValueRef;
    fn LLVMBuildPtrToInt(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                         Name: sbuf) -> ValueRef;
    fn LLVMBuildIntToPtr(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                         Name: sbuf) -> ValueRef;
    fn LLVMBuildBitCast(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                        Name: sbuf) -> ValueRef;
    fn LLVMBuildZExtOrBitCast(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                              Name: sbuf) -> ValueRef;
    fn LLVMBuildSExtOrBitCast(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                              Name: sbuf) -> ValueRef;
    fn LLVMBuildTruncOrBitCast(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                               Name: sbuf) -> ValueRef;
    fn LLVMBuildCast(B: BuilderRef, Op: Opcode, Val: ValueRef,
                     DestTy: TypeRef, Name: sbuf) -> ValueRef;
    fn LLVMBuildPointerCast(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                            Name: sbuf) -> ValueRef;
    fn LLVMBuildIntCast(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                        Name: sbuf) -> ValueRef;
    fn LLVMBuildFPCast(B: BuilderRef, Val: ValueRef, DestTy: TypeRef,
                       Name: sbuf) -> ValueRef;

    /* Comparisons */
    fn LLVMBuildICmp(B: BuilderRef, Op: unsigned, LHS: ValueRef,
                     RHS: ValueRef, Name: sbuf) -> ValueRef;
    fn LLVMBuildFCmp(B: BuilderRef, Op: unsigned, LHS: ValueRef,
                     RHS: ValueRef, Name: sbuf) -> ValueRef;

    /* Miscellaneous instructions */
    fn LLVMBuildPhi(B: BuilderRef, Ty: TypeRef, Name: sbuf) -> ValueRef;
    fn LLVMBuildCall(B: BuilderRef, Fn: ValueRef, Args: *ValueRef,
                     NumArgs: unsigned, Name: sbuf) -> ValueRef;
    fn LLVMBuildSelect(B: BuilderRef, If: ValueRef, Then: ValueRef,
                       Else: ValueRef, Name: sbuf) -> ValueRef;
    fn LLVMBuildVAArg(B: BuilderRef, list: ValueRef, Ty: TypeRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildExtractElement(B: BuilderRef, VecVal: ValueRef,
                               Index: ValueRef, Name: sbuf) -> ValueRef;
    fn LLVMBuildInsertElement(B: BuilderRef, VecVal: ValueRef,
                              EltVal: ValueRef, Index: ValueRef, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildShuffleVector(B: BuilderRef, V1: ValueRef, V2: ValueRef,
                              Mask: ValueRef, Name: sbuf) -> ValueRef;
    fn LLVMBuildExtractValue(B: BuilderRef, AggVal: ValueRef, Index: unsigned,
                             Name: sbuf) -> ValueRef;
    fn LLVMBuildInsertValue(B: BuilderRef, AggVal: ValueRef, EltVal: ValueRef,
                            Index: unsigned, Name: sbuf) -> ValueRef;

    fn LLVMBuildIsNull(B: BuilderRef, Val: ValueRef, Name: sbuf) -> ValueRef;
    fn LLVMBuildIsNotNull(B: BuilderRef, Val: ValueRef, Name: sbuf) ->
       ValueRef;
    fn LLVMBuildPtrDiff(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                        Name: sbuf) -> ValueRef;

    /* Selected entries from the downcasts. */
    fn LLVMIsATerminatorInst(Inst: ValueRef) -> ValueRef;

    /** Writes a module to the specified path. Returns 0 on success. */
    fn LLVMWriteBitcodeToFile(M: ModuleRef, Path: sbuf) -> c_int;

    /** Creates target data from a target layout string. */
    fn LLVMCreateTargetData(StringRep: sbuf) -> TargetDataRef;
    /** Adds the target data to the given pass manager. The pass manager
        references the target data only weakly. */
    fn LLVMAddTargetData(TD: TargetDataRef, PM: PassManagerRef);
    /** Returns the size of a type. FIXME: rv is actually a ULongLong! */
    fn LLVMStoreSizeOfType(TD: TargetDataRef, Ty: TypeRef) -> unsigned;
    /** Returns the alignment of a type. */
    fn LLVMPreferredAlignmentOfType(TD: TargetDataRef,
                                    Ty: TypeRef) -> unsigned;
    /** Disposes target data. */
    fn LLVMDisposeTargetData(TD: TargetDataRef);

    /** Creates a pass manager. */
    fn LLVMCreatePassManager() -> PassManagerRef;
    /** Disposes a pass manager. */
    fn LLVMDisposePassManager(PM: PassManagerRef);
    /** Runs a pass manager on a module. */
    fn LLVMRunPassManager(PM: PassManagerRef, M: ModuleRef) -> Bool;

    /** Adds a verification pass. */
    fn LLVMAddVerifierPass(PM: PassManagerRef);

    fn LLVMAddGlobalOptimizerPass(PM: PassManagerRef);
    fn LLVMAddIPSCCPPass(PM: PassManagerRef);
    fn LLVMAddDeadArgEliminationPass(PM: PassManagerRef);
    fn LLVMAddInstructionCombiningPass(PM: PassManagerRef);
    fn LLVMAddCFGSimplificationPass(PM: PassManagerRef);
    fn LLVMAddFunctionInliningPass(PM: PassManagerRef);
    fn LLVMAddFunctionAttrsPass(PM: PassManagerRef);
    fn LLVMAddScalarReplAggregatesPass(PM: PassManagerRef);
    fn LLVMAddScalarReplAggregatesPassSSA(PM: PassManagerRef);
    fn LLVMAddJumpThreadingPass(PM: PassManagerRef);
    fn LLVMAddConstantPropagationPass(PM: PassManagerRef);
    fn LLVMAddReassociatePass(PM: PassManagerRef);
    fn LLVMAddLoopRotatePass(PM: PassManagerRef);
    fn LLVMAddLICMPass(PM: PassManagerRef);
    fn LLVMAddLoopUnswitchPass(PM: PassManagerRef);
    fn LLVMAddLoopDeletionPass(PM: PassManagerRef);
    fn LLVMAddLoopUnrollPass(PM: PassManagerRef);
    fn LLVMAddGVNPass(PM: PassManagerRef);
    fn LLVMAddMemCpyOptPass(PM: PassManagerRef);
    fn LLVMAddSCCPPass(PM: PassManagerRef);
    fn LLVMAddDeadStoreEliminationPass(PM: PassManagerRef);
    fn LLVMAddStripDeadPrototypesPass(PM: PassManagerRef);
    fn LLVMAddConstantMergePass(PM: PassManagerRef);
    fn LLVMAddArgumentPromotionPass(PM: PassManagerRef);
    fn LLVMAddTailCallEliminationPass(PM: PassManagerRef);
    fn LLVMAddIndVarSimplifyPass(PM: PassManagerRef);
    fn LLVMAddAggressiveDCEPass(PM: PassManagerRef);
    fn LLVMAddGlobalDCEPass(PM: PassManagerRef);
    fn LLVMAddCorrelatedValuePropagationPass(PM: PassManagerRef);
    fn LLVMAddPruneEHPass(PM: PassManagerRef);
    fn LLVMAddSimplifyLibCallsPass(PM: PassManagerRef);
    fn LLVMAddLoopIdiomPass(PM: PassManagerRef);
    fn LLVMAddEarlyCSEPass(PM: PassManagerRef);
    fn LLVMAddTypeBasedAliasAnalysisPass(PM: PassManagerRef);
    fn LLVMAddBasicAliasAnalysisPass(PM: PassManagerRef);

    fn LLVMPassManagerBuilderCreate() -> PassManagerBuilderRef;
    fn LLVMPassManagerBuilderDispose(PMB: PassManagerBuilderRef);
    fn LLVMPassManagerBuilderSetOptLevel(PMB: PassManagerBuilderRef,
                                         OptimizationLevel: unsigned);
    fn LLVMPassManagerBuilderSetSizeLevel(PMB: PassManagerBuilderRef,
                                          Value: Bool);
    fn LLVMPassManagerBuilderSetDisableUnitAtATime(PMB: PassManagerBuilderRef,
                                                   Value: Bool);
    fn LLVMPassManagerBuilderSetDisableUnrollLoops(PMB: PassManagerBuilderRef,
                                                   Value: Bool);
    fn LLVMPassManagerBuilderSetDisableSimplifyLibCalls
        (PMB: PassManagerBuilderRef, Value: Bool);
    fn LLVMPassManagerBuilderUseInlinerWithThreshold
        (PMB: PassManagerBuilderRef, threshold: unsigned);
    fn LLVMPassManagerBuilderPopulateModulePassManager
        (PMB: PassManagerBuilderRef, PM: PassManagerRef);

    fn LLVMPassManagerBuilderPopulateFunctionPassManager
        (PMB: PassManagerBuilderRef, PM: PassManagerRef);

    /** Destroys a memory buffer. */
    fn LLVMDisposeMemoryBuffer(MemBuf: MemoryBufferRef);


    /* Stuff that's in rustllvm/ because it's not upstream yet. */

    /** Opens an object file. */
    fn LLVMCreateObjectFile(MemBuf: MemoryBufferRef) -> ObjectFileRef;
    /** Closes an object file. */
    fn LLVMDisposeObjectFile(ObjectFile: ObjectFileRef);

    /** Enumerates the sections in an object file. */
    fn LLVMGetSections(ObjectFile: ObjectFileRef) -> SectionIteratorRef;
    /** Destroys a section iterator. */
    fn LLVMDisposeSectionIterator(SI: SectionIteratorRef);
    /** Returns true if the section iterator is at the end of the section
        list: */
    fn LLVMIsSectionIteratorAtEnd(ObjectFile: ObjectFileRef,
                                  SI: SectionIteratorRef) -> Bool;
    /** Moves the section iterator to point to the next section. */
    fn LLVMMoveToNextSection(SI: SectionIteratorRef);
    /** Returns the current section name. */
    fn LLVMGetSectionName(SI: SectionIteratorRef) -> sbuf;
    /** Returns the current section size. */
    fn LLVMGetSectionSize(SI: SectionIteratorRef) -> ulonglong;
    /** Returns the current section contents as a string buffer. */
    fn LLVMGetSectionContents(SI: SectionIteratorRef) -> sbuf;

    /** Reads the given file and returns it as a memory buffer. Use
        LLVMDisposeMemoryBuffer() to get rid of it. */
    fn LLVMRustCreateMemoryBufferWithContentsOfFile(Path: sbuf) ->
       MemoryBufferRef;

    /* FIXME: The FileType is an enum.*/
    fn LLVMRustWriteOutputFile(PM: PassManagerRef, M: ModuleRef, Triple: sbuf,
                               Output: sbuf, FileType: c_int, OptLevel: c_int,
                               EnableSegmentedStacks: bool);

    /** Returns a string describing the last error caused by an LLVMRust*
        call. */
    fn LLVMRustGetLastError() -> sbuf;

    /** Parses the bitcode in the given memory buffer. */
    fn LLVMRustParseBitcode(MemBuf: MemoryBufferRef) -> ModuleRef;

    /** Parses LLVM asm in the given file */
    fn LLVMRustParseAssemblyFile(Filename: sbuf) -> ModuleRef;

    /** FiXME: Hacky adaptor for lack of ULongLong in FFI: */
    fn LLVMRustConstInt(IntTy: TypeRef, N_hi: unsigned, N_lo: unsigned,
                        SignExtend: Bool) -> ValueRef;

    fn LLVMRustAddPrintModulePass(PM: PassManagerRef, M: ModuleRef,
                                  Output: sbuf);

    /** Turn on LLVM pass-timing. */
    fn LLVMRustEnableTimePasses();

    /** Print the pass timings since static dtors aren't picking them up. */
    fn LLVMRustPrintPassTimings();

    fn LLVMStructCreateNamed(C: ContextRef, Name: sbuf) -> TypeRef;

    fn LLVMStructSetBody(StructTy: TypeRef, ElementTypes: *TypeRef,
                         ElementCount: unsigned, Packed: Bool);

    fn LLVMConstNamedStruct(S: TypeRef, ConstantVals: *ValueRef,
                            Count: unsigned) -> ValueRef;

    /** Links LLVM modules together. `Src` is destroyed by this call and
        must never be referenced again. */
    fn LLVMLinkModules(Dest: ModuleRef, Src: ModuleRef) -> Bool;
}

fn SetInstructionCallConv(Instr: ValueRef, CC: CallConv) {
    llvm::LLVMSetInstructionCallConv(Instr, CC as unsigned);
}
fn SetFunctionCallConv(Fn: ValueRef, CC: CallConv) {
    llvm::LLVMSetFunctionCallConv(Fn, CC as unsigned);
}
fn SetLinkage(Global: ValueRef, Link: Linkage) {
    llvm::LLVMSetLinkage(Global, Link as unsigned);
}

/* Memory-managed object interface to type handles. */

type type_names = @{type_names: std::map::hashmap<TypeRef, str>,
                    named_types: std::map::hashmap<str, TypeRef>};

fn associate_type(tn: type_names, s: str, t: TypeRef) {
    assert tn.type_names.insert(t, s);
    assert tn.named_types.insert(s, t);
}

fn type_has_name(tn: type_names, t: TypeRef) -> option<str> {
    ret tn.type_names.find(t);
}

fn name_has_type(tn: type_names, s: str) -> option<TypeRef> {
    ret tn.named_types.find(s);
}

fn mk_type_names() -> type_names {
    fn hash(&&t: TypeRef) -> uint { ret t as uint; }
    fn eq(&&a: TypeRef, &&b: TypeRef) -> bool { ret a as uint == b as uint; }
    @{type_names: std::map::mk_hashmap(hash, eq),
      named_types: std::map::new_str_hash()}
}

fn type_to_str(names: type_names, ty: TypeRef) -> str {
    ret type_to_str_inner(names, [], ty);
}

fn type_to_str_inner(names: type_names, outer0: [TypeRef], ty: TypeRef) ->
   str {
    alt type_has_name(names, ty) {
      option::some(n) { ret n; }
      _ {}
    }

    let outer = outer0 + [ty];

    let kind: int = llvm::LLVMGetTypeKind(ty) as int;

    fn tys_str(names: type_names, outer: [TypeRef], tys: [TypeRef]) -> str {
        let s: str = "";
        let first: bool = true;
        for t: TypeRef in tys {
            if first { first = false; } else { s += ", "; }
            s += type_to_str_inner(names, outer, t);
        }
        ret s;
    }

    alt kind {
      // FIXME: more enum-as-int constants determined from Core::h;
      // horrible, horrible. Complete as needed.
      0 { ret "Void"; }
      1 { ret "Half"; }
      2 { ret "Float"; }
      3 { ret "Double"; }
      4 { ret "X86_FP80"; }
      5 { ret "FP128"; }
      6 { ret "PPC_FP128"; }
      7 { ret "Label"; }
      8 {
        ret "i" + int::str(llvm::LLVMGetIntTypeWidth(ty) as int);
      }
      9 {
        let s = "fn(";
        let out_ty: TypeRef = llvm::LLVMGetReturnType(ty);
        let n_args = llvm::LLVMCountParamTypes(ty) as uint;
        let args: [TypeRef] = vec::init_elt::<TypeRef>(n_args, 0 as TypeRef);
        unsafe {
            llvm::LLVMGetParamTypes(ty, vec::to_ptr(args));
        }
        s += tys_str(names, outer, args);
        s += ") -> ";
        s += type_to_str_inner(names, outer, out_ty);
        ret s;
      }
      10 {
        let s: str = "{";
        let n_elts = llvm::LLVMCountStructElementTypes(ty) as uint;
        let elts: [TypeRef] = vec::init_elt::<TypeRef>(n_elts, 0 as TypeRef);
        unsafe {
            llvm::LLVMGetStructElementTypes(ty, vec::to_ptr(elts));
        }
        s += tys_str(names, outer, elts);
        s += "}";
        ret s;
      }
      11 {
        let el_ty = llvm::LLVMGetElementType(ty);
        ret "[" + type_to_str_inner(names, outer, el_ty) + " x " +
            uint::str(llvm::LLVMGetArrayLength(ty) as uint) + "]";
      }
      12 {
        let i: uint = 0u;
        for tout: TypeRef in outer0 {
            i += 1u;
            if tout as int == ty as int {
                let n: uint = vec::len::<TypeRef>(outer0) - i;
                ret "*\\" + int::str(n as int);
            }
        }
        ret "*" +
                type_to_str_inner(names, outer, llvm::LLVMGetElementType(ty));
      }
      13 { ret "Vector"; }
      14 { ret "Metadata"; }
      15 { ret "X86_MMAX"; }
      _ { #error("unknown TypeKind %d", kind as int); fail; }
    }
}

fn float_width(llt: TypeRef) -> uint {
    ret alt llvm::LLVMGetTypeKind(llt) as int {
          1 { 32u }
          2 { 64u }
          3 { 80u }
          4 | 5 { 128u }
          _ { fail "llvm_float_width called on a non-float type" }
        };
}

fn fn_ty_param_tys(fn_ty: TypeRef) -> [TypeRef] unsafe {
    let args = vec::init_elt(llvm::LLVMCountParamTypes(fn_ty) as uint,
                             0 as TypeRef);
    llvm::LLVMGetParamTypes(fn_ty, vec::to_ptr(args));
    ret args;
}


/* Memory-managed interface to target data. */

resource target_data_res(TD: TargetDataRef) {
    llvm::LLVMDisposeTargetData(TD);
}

type target_data = {lltd: TargetDataRef, dtor: @target_data_res};

fn mk_target_data(string_rep: str) -> target_data {
    let lltd =
        str::as_buf(string_rep, {|buf| llvm::LLVMCreateTargetData(buf) });
    ret {lltd: lltd, dtor: @target_data_res(lltd)};
}

/* Memory-managed interface to pass managers. */

resource pass_manager_res(PM: PassManagerRef) {
    llvm::LLVMDisposePassManager(PM);
}

type pass_manager = {llpm: PassManagerRef, dtor: @pass_manager_res};

fn mk_pass_manager() -> pass_manager {
    let llpm = llvm::LLVMCreatePassManager();
    ret {llpm: llpm, dtor: @pass_manager_res(llpm)};
}

/* Memory-managed interface to object files. */

resource object_file_res(ObjectFile: ObjectFileRef) {
    llvm::LLVMDisposeObjectFile(ObjectFile);
}

type object_file = {llof: ObjectFileRef, dtor: @object_file_res};

fn mk_object_file(llmb: MemoryBufferRef) -> option<object_file> {
    let llof = llvm::LLVMCreateObjectFile(llmb);
    if llof as int == 0 { ret option::none::<object_file>; }
    ret option::some({llof: llof, dtor: @object_file_res(llof)});
}

/* Memory-managed interface to section iterators. */

resource section_iter_res(SI: SectionIteratorRef) {
    llvm::LLVMDisposeSectionIterator(SI);
}

type section_iter = {llsi: SectionIteratorRef, dtor: @section_iter_res};

fn mk_section_iter(llof: ObjectFileRef) -> section_iter {
    let llsi = llvm::LLVMGetSections(llof);
    ret {llsi: llsi, dtor: @section_iter_res(llsi)};
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
