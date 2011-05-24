import std::vec;
import std::str;
import std::str::rustrt::sbuf;
import std::vec::rustrt::vbuf;

import llvm::ModuleRef;
import llvm::ContextRef;
import llvm::TypeRef;
import llvm::TypeHandleRef;
import llvm::ValueRef;
import llvm::BasicBlockRef;
import llvm::BuilderRef;
import llvm::ModuleProviderRef;
import llvm::MemoryBufferRef;
import llvm::PassManagerRef;
import llvm::UseRef;
import llvm::TargetDataRef;
import llvm::Linkage;
import llvm::Attribute;
import llvm::Visibility;
import llvm::CallConv;
import llvm::IntPredicate;
import llvm::RealPredicate;
import llvm::Opcode;
import llvm::ObjectFileRef;
import llvm::SectionIteratorRef;

type ULongLong = u64;
type LongLong = i64;
type Long = i32;
type Bool = int;


const Bool True = 1;
const Bool False = 0;

// Consts for the LLVM CallConv type, pre-cast to uint.
// FIXME: figure out a way to merge these with the native
// typedef and/or a tag type in the native module below.

const uint LLVMCCallConv = 0u;
const uint LLVMFastCallConv = 8u;
const uint LLVMColdCallConv = 9u;
const uint LLVMX86StdcallCallConv = 64u;
const uint LLVMX86FastcallCallConv = 65u;

const uint LLVMDefaultVisibility = 0u;
const uint LLVMHiddenVisibility = 1u;
const uint LLVMProtectedVisibility = 2u;

const uint LLVMExternalLinkage = 0u;
const uint LLVMAvailableExternallyLinkage = 1u;
const uint LLVMLinkOnceAnyLinkage = 2u;
const uint LLVMLinkOnceODRLinkage = 3u;
const uint LLVMWeakAnyLinkage = 4u;
const uint LLVMWeakODRLinkage = 5u;
const uint LLVMAppendingLinkage = 6u;
const uint LLVMInternalLinkage = 7u;
const uint LLVMPrivateLinkage = 8u;
const uint LLVMDLLImportLinkage = 9u;
const uint LLVMDLLExportLinkage = 10u;
const uint LLVMExternalWeakLinkage = 11u;
const uint LLVMGhostLinkage = 12u;
const uint LLVMCommonLinkage = 13u;
const uint LLVMLinkerPrivateLinkage = 14u;
const uint LLVMLinkerPrivateWeakLinkage = 15u;

const uint LLVMZExtAttribute = 1u;
const uint LLVMSExtAttribute = 2u;
const uint LLVMNoReturnAttribute = 4u;
const uint LLVMInRegAttribute = 8u;
const uint LLVMStructRetAttribute = 16u;
const uint LLVMNoUnwindAttribute = 32u;
const uint LLVMNoAliasAttribute = 64u;
const uint LLVMByValAttribute = 128u;
const uint LLVMNestAttribute = 256u;
const uint LLVMReadNoneAttribute = 512u;
const uint LLVMReadOnlyAttribute = 1024u;
const uint LLVMNoInlineAttribute = 2048u;
const uint LLVMAlwaysInlineAttribute = 4096u;
const uint LLVMOptimizeForSizeAttribute = 8192u;
const uint LLVMStackProtectAttribute = 16384u;
const uint LLVMStackProtectReqAttribute = 32768u;
const uint LLVMAlignmentAttribute = 2031616u;   // 31 << 16
const uint LLVMNoCaptureAttribute = 2097152u;
const uint LLVMNoRedZoneAttribute = 4194304u;
const uint LLVMNoImplicitFloatAttribute = 8388608u;
const uint LLVMNakedAttribute = 16777216u;
const uint LLVMInlineHintAttribute = 33554432u;
const uint LLVMStackAttribute = 469762048u;     // 7 << 26
const uint LLVMUWTableAttribute = 1073741824u; // 1 << 30


// Consts for the LLVM IntPredicate type, pre-cast to uint.
// FIXME: as above.

const uint LLVMIntEQ = 32u;
const uint LLVMIntNE = 33u;
const uint LLVMIntUGT = 34u;
const uint LLVMIntUGE = 35u;
const uint LLVMIntULT = 36u;
const uint LLVMIntULE = 37u;
const uint LLVMIntSGT = 38u;
const uint LLVMIntSGE = 39u;
const uint LLVMIntSLT = 40u;
const uint LLVMIntSLE = 41u;


// Consts for the LLVM RealPredicate type, pre-case to uint.
// FIXME: as above.

const uint LLVMRealOEQ = 1u;
const uint LLVMRealOGT = 2u;
const uint LLVMRealOGE = 3u;
const uint LLVMRealOLT = 4u;
const uint LLVMRealOLE = 5u;
const uint LLVMRealONE = 6u;

const uint LLVMRealORD = 7u;
const uint LLVMRealUNO = 8u;
const uint LLVMRealUEQ = 9u;
const uint LLVMRealUGT = 10u;
const uint LLVMRealUGE = 11u;
const uint LLVMRealULT = 12u;
const uint LLVMRealULE = 13u;
const uint LLVMRealUNE = 14u;

native mod llvm = llvm_lib {

    type ModuleRef;
    type ContextRef;
    type TypeRef;
    type TypeHandleRef;
    type ValueRef;
    type BasicBlockRef;
    type BuilderRef;
    type ModuleProviderRef;
    type MemoryBufferRef;
    type PassManagerRef;
    type UseRef;
    type TargetDataRef;

    /* FIXME: These are enums in the C header. Represent them how, in rust? */
    type Linkage;
    type Attribute;
    type Visibility;
    type CallConv;
    type IntPredicate;
    type RealPredicate;
    type Opcode;

    /* Create and destroy contexts. */
    fn LLVMContextCreate() -> ContextRef;
    fn LLVMGetGlobalContext() -> ContextRef;
    fn LLVMContextDispose(ContextRef C);
    fn LLVMGetMDKindIDInContext(ContextRef C, sbuf Name, uint SLen) -> uint;
    fn LLVMGetMDKindID(sbuf Name, uint SLen) -> uint;

    /* Create and destroy modules. */
    fn LLVMModuleCreateWithNameInContext(sbuf ModuleID,
                                         ContextRef C)-> ModuleRef;
    fn LLVMDisposeModule(ModuleRef M);

    /** Data layout. See Module::getDataLayout. */
    fn LLVMGetDataLayout(ModuleRef M) -> sbuf;
    fn LLVMSetDataLayout(ModuleRef M, sbuf Triple);

    /** Target triple. See Module::getTargetTriple. */
    fn LLVMGetTarget(ModuleRef M) -> sbuf;
    fn LLVMSetTarget(ModuleRef M, sbuf Triple);

    /** See Module::addTypeName. */
    fn LLVMAddTypeName(ModuleRef M, sbuf Name, TypeRef Ty) -> Bool;
    fn LLVMDeleteTypeName(ModuleRef M, sbuf Name);
    fn LLVMGetTypeByName(ModuleRef M, sbuf Name) -> TypeRef;

    /** See Module::dump. */
    fn LLVMDumpModule(ModuleRef M);

    /** See Module::setModuleInlineAsm. */
    fn LLVMSetModuleInlineAsm(ModuleRef M, sbuf Asm);

    /** See llvm::LLVMTypeKind::getTypeID. */

    // FIXME: returning int rather than TypeKind because
    // we directly inspect the values, and casting from
    // a native doesn't work yet (only *to* a native).

    fn LLVMGetTypeKind(TypeRef Ty) -> int;

    /** See llvm::LLVMType::getContext. */
    fn LLVMGetTypeContext(TypeRef Ty) -> ContextRef;

    /* Operations on integer types */
    fn LLVMInt1TypeInContext(ContextRef C) -> TypeRef;
    fn LLVMInt8TypeInContext(ContextRef C) -> TypeRef;
    fn LLVMInt16TypeInContext(ContextRef C) -> TypeRef;
    fn LLVMInt32TypeInContext(ContextRef C) -> TypeRef;
    fn LLVMInt64TypeInContext(ContextRef C) -> TypeRef;
    fn LLVMIntTypeInContext(ContextRef C, uint NumBits) -> TypeRef;

    fn LLVMInt1Type() -> TypeRef;
    fn LLVMInt8Type() -> TypeRef;
    fn LLVMInt16Type() -> TypeRef;
    fn LLVMInt32Type() -> TypeRef;
    fn LLVMInt64Type() -> TypeRef;
    fn LLVMIntType(uint NumBits) -> TypeRef;
    fn LLVMGetIntTypeWidth(TypeRef IntegerTy) -> uint;

    /* Operations on real types */
    fn LLVMFloatTypeInContext(ContextRef C) -> TypeRef;
    fn LLVMDoubleTypeInContext(ContextRef C) -> TypeRef;
    fn LLVMX86FP80TypeInContext(ContextRef C) -> TypeRef;
    fn LLVMFP128TypeInContext(ContextRef C) -> TypeRef;
    fn LLVMPPCFP128TypeInContext(ContextRef C) -> TypeRef;

    fn LLVMFloatType() -> TypeRef;
    fn LLVMDoubleType() -> TypeRef;
    fn LLVMX86FP80Type() -> TypeRef;
    fn LLVMFP128Type() -> TypeRef;
    fn LLVMPPCFP128Type() -> TypeRef;

    /* Operations on function types */
    fn LLVMFunctionType(TypeRef ReturnType, vbuf ParamTypes,
                        uint ParamCount, Bool IsVarArg) -> TypeRef;
    fn LLVMIsFunctionVarArg(TypeRef FunctionTy) -> Bool;
    fn LLVMGetReturnType(TypeRef FunctionTy) -> TypeRef;
    fn LLVMCountParamTypes(TypeRef FunctionTy) -> uint;
    fn LLVMGetParamTypes(TypeRef FunctionTy, vbuf Dest);

    /* Operations on struct types */
    fn LLVMStructTypeInContext(ContextRef C, vbuf ElementTypes,
                               uint ElementCount, Bool Packed) -> TypeRef;
    fn LLVMStructType(vbuf ElementTypes, uint ElementCount,
                      Bool Packed) -> TypeRef;
    fn LLVMCountStructElementTypes(TypeRef StructTy) -> uint;
    fn LLVMGetStructElementTypes(TypeRef StructTy, vbuf Dest);
    fn LLVMIsPackedStruct(TypeRef StructTy) -> Bool;

    /* Operations on array, pointer, and vector types (sequence types) */
    fn LLVMArrayType(TypeRef ElementType, uint ElementCount) -> TypeRef;
    fn LLVMPointerType(TypeRef ElementType, uint AddressSpace) -> TypeRef;
    fn LLVMVectorType(TypeRef ElementType, uint ElementCount) -> TypeRef;

    fn LLVMGetElementType(TypeRef Ty) -> TypeRef;
    fn LLVMGetArrayLength(TypeRef ArrayTy) -> uint;
    fn LLVMGetPointerAddressSpace(TypeRef PointerTy) -> uint;
    fn LLVMGetVectorSize(TypeRef VectorTy) -> uint;

    /* Operations on other types */
    fn LLVMVoidTypeInContext(ContextRef C) -> TypeRef;
    fn LLVMLabelTypeInContext(ContextRef C) -> TypeRef;
    fn LLVMOpaqueTypeInContext(ContextRef C) -> TypeRef;

    fn LLVMVoidType() -> TypeRef;
    fn LLVMLabelType() -> TypeRef;
    fn LLVMOpaqueType() -> TypeRef;

    /* Operations on type handles */
    fn LLVMCreateTypeHandle(TypeRef PotentiallyAbstractTy) -> TypeHandleRef;
    fn LLVMRefineType(TypeRef AbstractTy, TypeRef ConcreteTy);
    fn LLVMResolveTypeHandle(TypeHandleRef TypeHandle) -> TypeRef;
    fn LLVMDisposeTypeHandle(TypeHandleRef TypeHandle);

    /* Operations on all values */
    fn LLVMTypeOf(ValueRef Val) -> TypeRef;
    fn LLVMGetValueName(ValueRef Val) -> sbuf;
    fn LLVMSetValueName(ValueRef Val, sbuf Name);
    fn LLVMDumpValue(ValueRef Val);
    fn LLVMReplaceAllUsesWith(ValueRef OldVal, ValueRef NewVal);
    fn LLVMHasMetadata(ValueRef Val) -> int;
    fn LLVMGetMetadata(ValueRef Val, uint KindID) -> ValueRef;
    fn LLVMSetMetadata(ValueRef Val, uint KindID, ValueRef Node);

    /* Operations on Uses */
    fn LLVMGetFirstUse(ValueRef Val) -> UseRef;
    fn LLVMGetNextUse(UseRef U) -> UseRef;
    fn LLVMGetUser(UseRef U) -> ValueRef;
    fn LLVMGetUsedValue(UseRef U) -> ValueRef;

    /* Operations on Users */
    fn LLVMGetOperand(ValueRef Val, uint Index) -> ValueRef;

    /* Operations on constants of any type */
    fn LLVMConstNull(TypeRef Ty) -> ValueRef; /* all zeroes */
    fn LLVMConstAllOnes(TypeRef Ty) -> ValueRef; /* only for int/vector */
    fn LLVMGetUndef(TypeRef Ty) -> ValueRef;
    fn LLVMIsConstant(ValueRef Val) -> Bool;
    fn LLVMIsNull(ValueRef Val) -> Bool;
    fn LLVMIsUndef(ValueRef Val) -> Bool;
    fn LLVMConstPointerNull(TypeRef Ty) -> ValueRef;

    /* Operations on metadata */
    fn LLVMMDStringInContext(ContextRef C, sbuf Str, uint SLen) -> ValueRef;
    fn LLVMMDString(sbuf Str, uint SLen) -> ValueRef;
    fn LLVMMDNodeInContext(ContextRef C, vbuf Vals, uint Count) -> ValueRef;
    fn LLVMMDNode(vbuf Vals, uint Count) -> ValueRef;

    /* Operations on scalar constants */
    fn LLVMConstInt(TypeRef IntTy, ULongLong N, Bool SignExtend) -> ValueRef;
    // FIXME: radix is actually u8, but our native layer can't handle this
    // yet.  lucky for us we're little-endian. Small miracles.
    fn LLVMConstIntOfString(TypeRef IntTy, sbuf Text, int Radix) -> ValueRef;
    fn LLVMConstIntOfStringAndSize(TypeRef IntTy, sbuf Text,
                                   uint SLen, u8 Radix) -> ValueRef;
    fn LLVMConstReal(TypeRef RealTy, f64 N) -> ValueRef;
    fn LLVMConstRealOfString(TypeRef RealTy, sbuf Text) -> ValueRef;
    fn LLVMConstRealOfStringAndSize(TypeRef RealTy, sbuf Text,
                                    uint SLen) -> ValueRef;
    fn LLVMConstIntGetZExtValue(ValueRef ConstantVal) -> ULongLong;
    fn LLVMConstIntGetSExtValue(ValueRef ConstantVal) -> LongLong;


    /* Operations on composite constants */
    fn LLVMConstStringInContext(ContextRef C, sbuf Str, uint Length,
                                Bool DontNullTerminate) -> ValueRef;
    fn LLVMConstStructInContext(ContextRef C, vbuf ConstantVals,
                                uint Count, Bool Packed) -> ValueRef;

    fn LLVMConstString(sbuf Str, uint Length,
                       Bool DontNullTerminate) -> ValueRef;
    fn LLVMConstArray(TypeRef ElementTy,
                      vbuf ConstantVals, uint Length) -> ValueRef;
    fn LLVMConstStruct(vbuf ConstantVals, uint Count,
                       Bool Packed) -> ValueRef;
    fn LLVMConstVector(vbuf ScalarConstantVals, uint Size) -> ValueRef;

    /* Constant expressions */
    fn LLVMAlignOf(TypeRef Ty) -> ValueRef;
    fn LLVMSizeOf(TypeRef Ty) -> ValueRef;
    fn LLVMConstNeg(ValueRef ConstantVal) -> ValueRef;
    fn LLVMConstNSWNeg(ValueRef ConstantVal) -> ValueRef;
    fn LLVMConstNUWNeg(ValueRef ConstantVal) -> ValueRef;
    fn LLVMConstFNeg(ValueRef ConstantVal) -> ValueRef;
    fn LLVMConstNot(ValueRef ConstantVal) -> ValueRef;
    fn LLVMConstAdd(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstNSWAdd(ValueRef LHSConstant,
                       ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstNUWAdd(ValueRef LHSConstant,
                       ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstFAdd(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstSub(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstNSWSub(ValueRef LHSConstant,
                       ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstNUWSub(ValueRef LHSConstant,
                       ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstFSub(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstMul(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstNSWMul(ValueRef LHSConstant,
                       ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstNUWMul(ValueRef LHSConstant,
                       ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstFMul(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstUDiv(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstSDiv(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstExactSDiv(ValueRef LHSConstant,
                          ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstFDiv(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstURem(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstSRem(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstFRem(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstAnd(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstOr(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstXor(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstShl(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstLShr(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstAShr(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
    fn LLVMConstGEP(ValueRef ConstantVal,
                    vbuf ConstantIndices, uint NumIndices) -> ValueRef;
    fn LLVMConstInBoundsGEP(ValueRef ConstantVal,
                            vbuf ConstantIndices,
                            uint NumIndices) -> ValueRef;
    fn LLVMConstTrunc(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
    fn LLVMConstSExt(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
    fn LLVMConstZExt(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
    fn LLVMConstFPTrunc(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
    fn LLVMConstFPExt(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
    fn LLVMConstUIToFP(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
    fn LLVMConstSIToFP(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
    fn LLVMConstFPToUI(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
    fn LLVMConstFPToSI(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
    fn LLVMConstPtrToInt(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
    fn LLVMConstIntToPtr(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
    fn LLVMConstBitCast(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
    fn LLVMConstZExtOrBitCast(ValueRef ConstantVal,
                              TypeRef ToType) -> ValueRef;
    fn LLVMConstSExtOrBitCast(ValueRef ConstantVal,
                              TypeRef ToType) -> ValueRef;
    fn LLVMConstTruncOrBitCast(ValueRef ConstantVal,
                               TypeRef ToType) -> ValueRef;
    fn LLVMConstPointerCast(ValueRef ConstantVal,
                            TypeRef ToType) -> ValueRef;
    fn LLVMConstIntCast(ValueRef ConstantVal, TypeRef ToType,
                        Bool isSigned) -> ValueRef;
    fn LLVMConstFPCast(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
    fn LLVMConstSelect(ValueRef ConstantCondition,
                       ValueRef ConstantIfTrue,
                       ValueRef ConstantIfFalse) -> ValueRef;
    fn LLVMConstExtractElement(ValueRef VectorConstant,
                               ValueRef IndexConstant) -> ValueRef;
    fn LLVMConstInsertElement(ValueRef VectorConstant,
                              ValueRef ElementValueConstant,
                              ValueRef IndexConstant) -> ValueRef;
    fn LLVMConstShuffleVector(ValueRef VectorAConstant,
                              ValueRef VectorBConstant,
                              ValueRef MaskConstant) -> ValueRef;
    fn LLVMConstExtractValue(ValueRef AggConstant, vbuf IdxList,
                             uint NumIdx) -> ValueRef;
    fn LLVMConstInsertValue(ValueRef AggConstant,
                            ValueRef ElementValueConstant,
                            vbuf IdxList, uint NumIdx) -> ValueRef;
    fn LLVMConstInlineAsm(TypeRef Ty,
                          sbuf AsmString, sbuf Constraints,
                          Bool HasSideEffects, Bool IsAlignStack) -> ValueRef;
    fn LLVMBlockAddress(ValueRef F, BasicBlockRef BB) -> ValueRef;



    /* Operations on global variables, functions, and aliases (globals) */
    fn LLVMGetGlobalParent(ValueRef Global) -> ModuleRef;
    fn LLVMIsDeclaration(ValueRef Global) -> Bool;
    fn LLVMGetLinkage(ValueRef Global) -> Linkage;
    fn LLVMSetLinkage(ValueRef Global, Linkage Link);
    fn LLVMGetSection(ValueRef Global) -> sbuf;
    fn LLVMSetSection(ValueRef Global, sbuf Section);
    fn LLVMGetVisibility(ValueRef Global) -> Visibility;
    fn LLVMSetVisibility(ValueRef Global, Visibility Viz);
    fn LLVMGetAlignment(ValueRef Global) -> uint;
    fn LLVMSetAlignment(ValueRef Global, uint Bytes);


    /* Operations on global variables */
    fn LLVMAddGlobal(ModuleRef M, TypeRef Ty, sbuf Name) -> ValueRef;
    fn LLVMAddGlobalInAddressSpace(ModuleRef M, TypeRef Ty,
                                   sbuf Name,
                                   uint AddressSpace) -> ValueRef;
    fn LLVMGetNamedGlobal(ModuleRef M, sbuf Name) -> ValueRef;
    fn LLVMGetFirstGlobal(ModuleRef M) -> ValueRef;
    fn LLVMGetLastGlobal(ModuleRef M) -> ValueRef;
    fn LLVMGetNextGlobal(ValueRef GlobalVar) -> ValueRef;
    fn LLVMGetPreviousGlobal(ValueRef GlobalVar) -> ValueRef;
    fn LLVMDeleteGlobal(ValueRef GlobalVar);
    fn LLVMGetInitializer(ValueRef GlobalVar) -> ValueRef;
    fn LLVMSetInitializer(ValueRef GlobalVar, ValueRef ConstantVal);
    fn LLVMIsThreadLocal(ValueRef GlobalVar) -> Bool;
    fn LLVMSetThreadLocal(ValueRef GlobalVar, Bool IsThreadLocal);
    fn LLVMIsGlobalConstant(ValueRef GlobalVar) -> Bool;
    fn LLVMSetGlobalConstant(ValueRef GlobalVar, Bool IsConstant);

    /* Operations on aliases */
    fn LLVMAddAlias(ModuleRef M, TypeRef Ty, ValueRef Aliasee,
                    sbuf Name) -> ValueRef;

    /* Operations on functions */
    fn LLVMAddFunction(ModuleRef M, sbuf Name,
                       TypeRef FunctionTy) -> ValueRef;
    fn LLVMGetNamedFunction(ModuleRef M, sbuf Name) -> ValueRef;
    fn LLVMGetFirstFunction(ModuleRef M) -> ValueRef;
    fn LLVMGetLastFunction(ModuleRef M) -> ValueRef;
    fn LLVMGetNextFunction(ValueRef Fn) -> ValueRef;
    fn LLVMGetPreviousFunction(ValueRef Fn) -> ValueRef;
    fn LLVMDeleteFunction(ValueRef Fn);
    fn LLVMGetIntrinsicID(ValueRef Fn) -> uint;
    fn LLVMGetFunctionCallConv(ValueRef Fn) -> uint;
    fn LLVMSetFunctionCallConv(ValueRef Fn, uint CC);
    fn LLVMGetGC(ValueRef Fn) -> sbuf;
    fn LLVMSetGC(ValueRef Fn, sbuf Name);
    fn LLVMAddFunctionAttr(ValueRef Fn, Attribute PA);
    fn LLVMGetFunctionAttr(ValueRef Fn) -> Attribute;
    fn LLVMRemoveFunctionAttr(ValueRef Fn, Attribute PA);

    /* Operations on parameters */
    fn LLVMCountParams(ValueRef Fn) -> uint;
    fn LLVMGetParams(ValueRef Fn, vbuf Params);
    fn LLVMGetParam(ValueRef Fn, uint Index) -> ValueRef;
    fn LLVMGetParamParent(ValueRef Inst) -> ValueRef;
    fn LLVMGetFirstParam(ValueRef Fn) -> ValueRef;
    fn LLVMGetLastParam(ValueRef Fn) -> ValueRef;
    fn LLVMGetNextParam(ValueRef Arg) -> ValueRef;
    fn LLVMGetPreviousParam(ValueRef Arg) -> ValueRef;
    fn LLVMAddAttribute(ValueRef Arg, Attribute PA);
    fn LLVMRemoveAttribute(ValueRef Arg, Attribute PA);
    fn LLVMGetAttribute(ValueRef Arg) -> Attribute;
    fn LLVMSetParamAlignment(ValueRef Arg, uint align);

    /* Operations on basic blocks */
    fn LLVMBasicBlockAsValue(BasicBlockRef BB) -> ValueRef;
    fn LLVMValueIsBasicBlock(ValueRef Val) -> Bool;
    fn LLVMValueAsBasicBlock(ValueRef Val) -> BasicBlockRef;
    fn LLVMGetBasicBlockParent(BasicBlockRef BB) -> ValueRef;
    fn LLVMCountBasicBlocks(ValueRef Fn) -> uint;
    fn LLVMGetBasicBlocks(ValueRef Fn, vbuf BasicBlocks);
    fn LLVMGetFirstBasicBlock(ValueRef Fn) -> BasicBlockRef;
    fn LLVMGetLastBasicBlock(ValueRef Fn) -> BasicBlockRef;
    fn LLVMGetNextBasicBlock(BasicBlockRef BB) -> BasicBlockRef;
    fn LLVMGetPreviousBasicBlock(BasicBlockRef BB) -> BasicBlockRef;
    fn LLVMGetEntryBasicBlock(ValueRef Fn) -> BasicBlockRef;

    fn LLVMAppendBasicBlockInContext(ContextRef C, ValueRef Fn,
                                     sbuf Name) -> BasicBlockRef;
    fn LLVMInsertBasicBlockInContext(ContextRef C, BasicBlockRef BB,
                                     sbuf Name) -> BasicBlockRef;

    fn LLVMAppendBasicBlock(ValueRef Fn, sbuf Name) -> BasicBlockRef;
    fn LLVMInsertBasicBlock(BasicBlockRef InsertBeforeBB,
                            sbuf Name) -> BasicBlockRef;
    fn LLVMDeleteBasicBlock(BasicBlockRef BB);

    /* Operations on instructions */
    fn LLVMGetInstructionParent(ValueRef Inst) -> BasicBlockRef;
    fn LLVMGetFirstInstruction(BasicBlockRef BB) -> ValueRef;
    fn LLVMGetLastInstruction(BasicBlockRef BB) -> ValueRef;
    fn LLVMGetNextInstruction(ValueRef Inst) -> ValueRef;
    fn LLVMGetPreviousInstruction(ValueRef Inst) -> ValueRef;

    /* Operations on call sites */
    fn LLVMSetInstructionCallConv(ValueRef Instr, uint CC);
    fn LLVMGetInstructionCallConv(ValueRef Instr) -> uint;
    fn LLVMAddInstrAttribute(ValueRef Instr, uint index, Attribute IA);
    fn LLVMRemoveInstrAttribute(ValueRef Instr, uint index, Attribute IA);
    fn LLVMSetInstrParamAlignment(ValueRef Instr, uint index, uint align);

    /* Operations on call instructions (only) */
    fn LLVMIsTailCall(ValueRef CallInst) -> Bool;
    fn LLVMSetTailCall(ValueRef CallInst, Bool IsTailCall);

    /* Operations on phi nodes */
    fn LLVMAddIncoming(ValueRef PhiNode, vbuf IncomingValues,
                       vbuf IncomingBlocks, uint Count);
    fn LLVMCountIncoming(ValueRef PhiNode) -> uint;
    fn LLVMGetIncomingValue(ValueRef PhiNode, uint Index) -> ValueRef;
    fn LLVMGetIncomingBlock(ValueRef PhiNode, uint Index) -> BasicBlockRef;

    /* Instruction builders */
    fn LLVMCreateBuilderInContext(ContextRef C) -> BuilderRef;
    fn LLVMCreateBuilder() -> BuilderRef;
    fn LLVMPositionBuilder(BuilderRef Builder, BasicBlockRef Block,
                           ValueRef Instr);
    fn LLVMPositionBuilderBefore(BuilderRef Builder, ValueRef Instr);
    fn LLVMPositionBuilderAtEnd(BuilderRef Builder, BasicBlockRef Block);
    fn LLVMGetInsertBlock(BuilderRef Builder) -> BasicBlockRef;
    fn LLVMClearInsertionPosition(BuilderRef Builder);
    fn LLVMInsertIntoBuilder(BuilderRef Builder, ValueRef Instr);
    fn LLVMInsertIntoBuilderWithName(BuilderRef Builder, ValueRef Instr,
                                     sbuf Name);
    fn LLVMDisposeBuilder(BuilderRef Builder);

    /* Metadata */
    fn LLVMSetCurrentDebugLocation(BuilderRef Builder, ValueRef L);
    fn LLVMGetCurrentDebugLocation(BuilderRef Builder) -> ValueRef;
    fn LLVMSetInstDebugLocation(BuilderRef Builder, ValueRef Inst);

    /* Terminators */
    fn LLVMBuildRetVoid(BuilderRef B) -> ValueRef;
    fn LLVMBuildRet(BuilderRef B, ValueRef V) -> ValueRef;
    fn LLVMBuildAggregateRet(BuilderRef B, vbuf RetVals,
                             uint N) -> ValueRef;
    fn LLVMBuildBr(BuilderRef B, BasicBlockRef Dest) -> ValueRef;
    fn LLVMBuildCondBr(BuilderRef B, ValueRef If,
                       BasicBlockRef Then, BasicBlockRef Else) -> ValueRef;
    fn LLVMBuildSwitch(BuilderRef B, ValueRef V,
                       BasicBlockRef Else, uint NumCases) -> ValueRef;
    fn LLVMBuildIndirectBr(BuilderRef B, ValueRef Addr,
                           uint NumDests) -> ValueRef;
    fn LLVMBuildInvoke(BuilderRef B, ValueRef Fn,
                       vbuf Args, uint NumArgs,
                       BasicBlockRef Then, BasicBlockRef Catch,
                       sbuf Name) -> ValueRef;
    fn LLVMBuildUnwind(BuilderRef B) -> ValueRef;
    fn LLVMBuildUnreachable(BuilderRef B) -> ValueRef;

    /* Add a case to the switch instruction */
    fn LLVMAddCase(ValueRef Switch, ValueRef OnVal,
                   BasicBlockRef Dest);

    /* Add a destination to the indirectbr instruction */
    fn LLVMAddDestination(ValueRef IndirectBr, BasicBlockRef Dest);

    /* Arithmetic */
    fn LLVMBuildAdd(BuilderRef B, ValueRef LHS, ValueRef RHS,
                    sbuf Name) -> ValueRef;
    fn LLVMBuildNSWAdd(BuilderRef B, ValueRef LHS, ValueRef RHS,
                       sbuf Name) -> ValueRef;
    fn LLVMBuildNUWAdd(BuilderRef B, ValueRef LHS, ValueRef RHS,
                       sbuf Name) -> ValueRef;
    fn LLVMBuildFAdd(BuilderRef B, ValueRef LHS, ValueRef RHS,
                     sbuf Name) -> ValueRef;
    fn LLVMBuildSub(BuilderRef B, ValueRef LHS, ValueRef RHS,
                    sbuf Name) -> ValueRef;
    fn LLVMBuildNSWSub(BuilderRef B, ValueRef LHS, ValueRef RHS,
                       sbuf Name) -> ValueRef;
    fn LLVMBuildNUWSub(BuilderRef B, ValueRef LHS, ValueRef RHS,
                       sbuf Name) -> ValueRef;
    fn LLVMBuildFSub(BuilderRef B, ValueRef LHS, ValueRef RHS,
                     sbuf Name) -> ValueRef;
    fn LLVMBuildMul(BuilderRef B, ValueRef LHS, ValueRef RHS,
                    sbuf Name) -> ValueRef;
    fn LLVMBuildNSWMul(BuilderRef B, ValueRef LHS, ValueRef RHS,
                       sbuf Name) -> ValueRef;
    fn LLVMBuildNUWMul(BuilderRef B, ValueRef LHS, ValueRef RHS,
                       sbuf Name) -> ValueRef;
    fn LLVMBuildFMul(BuilderRef B, ValueRef LHS, ValueRef RHS,
                     sbuf Name) -> ValueRef;
    fn LLVMBuildUDiv(BuilderRef B, ValueRef LHS, ValueRef RHS,
                     sbuf Name) -> ValueRef;
    fn LLVMBuildSDiv(BuilderRef B, ValueRef LHS, ValueRef RHS,
                     sbuf Name) -> ValueRef;
    fn LLVMBuildExactSDiv(BuilderRef B, ValueRef LHS, ValueRef RHS,
                          sbuf Name) -> ValueRef;
    fn LLVMBuildFDiv(BuilderRef B, ValueRef LHS, ValueRef RHS,
                     sbuf Name) -> ValueRef;
    fn LLVMBuildURem(BuilderRef B, ValueRef LHS, ValueRef RHS,
                     sbuf Name) -> ValueRef;
    fn LLVMBuildSRem(BuilderRef B, ValueRef LHS, ValueRef RHS,
                     sbuf Name) -> ValueRef;
    fn LLVMBuildFRem(BuilderRef B, ValueRef LHS, ValueRef RHS,
                     sbuf Name) -> ValueRef;
    fn LLVMBuildShl(BuilderRef B, ValueRef LHS, ValueRef RHS,
                    sbuf Name) -> ValueRef;
    fn LLVMBuildLShr(BuilderRef B, ValueRef LHS, ValueRef RHS,
                     sbuf Name) -> ValueRef;
    fn LLVMBuildAShr(BuilderRef B, ValueRef LHS, ValueRef RHS,
                     sbuf Name) -> ValueRef;
    fn LLVMBuildAnd(BuilderRef B, ValueRef LHS, ValueRef RHS,
                    sbuf Name) -> ValueRef;
    fn LLVMBuildOr(BuilderRef B, ValueRef LHS, ValueRef RHS,
                   sbuf Name) -> ValueRef;
    fn LLVMBuildXor(BuilderRef B, ValueRef LHS, ValueRef RHS,
                    sbuf Name) -> ValueRef;
    fn LLVMBuildBinOp(BuilderRef B, Opcode Op,
                      ValueRef LHS, ValueRef RHS,
                      sbuf Name) -> ValueRef;
    fn LLVMBuildNeg(BuilderRef B, ValueRef V, sbuf Name) -> ValueRef;
    fn LLVMBuildNSWNeg(BuilderRef B, ValueRef V,
                       sbuf Name) -> ValueRef;
    fn LLVMBuildNUWNeg(BuilderRef B, ValueRef V,
                       sbuf Name) -> ValueRef;
    fn LLVMBuildFNeg(BuilderRef B, ValueRef V, sbuf Name) -> ValueRef;
    fn LLVMBuildNot(BuilderRef B, ValueRef V, sbuf Name) -> ValueRef;

    /* Memory */
    fn LLVMBuildMalloc(BuilderRef B, TypeRef Ty, sbuf Name) -> ValueRef;
    fn LLVMBuildArrayMalloc(BuilderRef B, TypeRef Ty,
                            ValueRef Val, sbuf Name) -> ValueRef;
    fn LLVMBuildAlloca(BuilderRef B, TypeRef Ty, sbuf Name) -> ValueRef;
    fn LLVMBuildArrayAlloca(BuilderRef B, TypeRef Ty,
                            ValueRef Val, sbuf Name) -> ValueRef;
    fn LLVMBuildFree(BuilderRef B, ValueRef PointerVal) -> ValueRef;
    fn LLVMBuildLoad(BuilderRef B, ValueRef PointerVal,
                     sbuf Name) -> ValueRef;
    fn LLVMBuildStore(BuilderRef B, ValueRef Val, ValueRef Ptr) -> ValueRef;
    fn LLVMBuildGEP(BuilderRef B, ValueRef Pointer,
                    vbuf Indices, uint NumIndices,
                    sbuf Name) -> ValueRef;
    fn LLVMBuildInBoundsGEP(BuilderRef B, ValueRef Pointer,
                            vbuf Indices, uint NumIndices,
                            sbuf Name) -> ValueRef;
    fn LLVMBuildStructGEP(BuilderRef B, ValueRef Pointer,
                          uint Idx, sbuf Name) -> ValueRef;
    fn LLVMBuildGlobalString(BuilderRef B, sbuf Str,
                             sbuf Name) -> ValueRef;
    fn LLVMBuildGlobalStringPtr(BuilderRef B, sbuf Str,
                                sbuf Name) -> ValueRef;

    /* Casts */
    fn LLVMBuildTrunc(BuilderRef B, ValueRef Val,
                      TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildZExt(BuilderRef B, ValueRef Val,
                     TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildSExt(BuilderRef B, ValueRef Val,
                     TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildFPToUI(BuilderRef B, ValueRef Val,
                       TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildFPToSI(BuilderRef B, ValueRef Val,
                       TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildUIToFP(BuilderRef B, ValueRef Val,
                       TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildSIToFP(BuilderRef B, ValueRef Val,
                       TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildFPTrunc(BuilderRef B, ValueRef Val,
                        TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildFPExt(BuilderRef B, ValueRef Val,
                      TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildPtrToInt(BuilderRef B, ValueRef Val,
                         TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildIntToPtr(BuilderRef B, ValueRef Val,
                         TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildBitCast(BuilderRef B, ValueRef Val,
                        TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildZExtOrBitCast(BuilderRef B, ValueRef Val,
                              TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildSExtOrBitCast(BuilderRef B, ValueRef Val,
                              TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildTruncOrBitCast(BuilderRef B, ValueRef Val,
                               TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildCast(BuilderRef B, Opcode Op, ValueRef Val,
                     TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildPointerCast(BuilderRef B, ValueRef Val,
                            TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildIntCast(BuilderRef B, ValueRef Val,
                        TypeRef DestTy, sbuf Name) -> ValueRef;
    fn LLVMBuildFPCast(BuilderRef B, ValueRef Val,
                       TypeRef DestTy, sbuf Name) -> ValueRef;

    /* Comparisons */
    fn LLVMBuildICmp(BuilderRef B, uint Op,
                     ValueRef LHS, ValueRef RHS,
                     sbuf Name) -> ValueRef;
    fn LLVMBuildFCmp(BuilderRef B, uint Op,
                     ValueRef LHS, ValueRef RHS,
                     sbuf Name) -> ValueRef;

    /* Miscellaneous instructions */
    fn LLVMBuildPhi(BuilderRef B, TypeRef Ty, sbuf Name) -> ValueRef;
    fn LLVMBuildCall(BuilderRef B, ValueRef Fn,
                     vbuf Args, uint NumArgs,
                     sbuf Name) -> ValueRef;
    fn LLVMBuildSelect(BuilderRef B, ValueRef If,
                       ValueRef Then, ValueRef Else,
                       sbuf Name) -> ValueRef;
    fn LLVMBuildVAArg(BuilderRef B, ValueRef list, TypeRef Ty,
                      sbuf Name) -> ValueRef;
    fn LLVMBuildExtractElement(BuilderRef B, ValueRef VecVal,
                               ValueRef Index, sbuf Name) -> ValueRef;
    fn LLVMBuildInsertElement(BuilderRef B, ValueRef VecVal,
                              ValueRef EltVal, ValueRef Index,
                              sbuf Name) -> ValueRef;
    fn LLVMBuildShuffleVector(BuilderRef B, ValueRef V1,
                              ValueRef V2, ValueRef Mask,
                              sbuf Name) -> ValueRef;
    fn LLVMBuildExtractValue(BuilderRef B, ValueRef AggVal,
                             uint Index, sbuf Name) -> ValueRef;
    fn LLVMBuildInsertValue(BuilderRef B, ValueRef AggVal,
                            ValueRef EltVal, uint Index,
                            sbuf Name) -> ValueRef;

    fn LLVMBuildIsNull(BuilderRef B, ValueRef Val,
                       sbuf Name) -> ValueRef;
    fn LLVMBuildIsNotNull(BuilderRef B, ValueRef Val,
                          sbuf Name) -> ValueRef;
    fn LLVMBuildPtrDiff(BuilderRef B, ValueRef LHS,
                        ValueRef RHS, sbuf Name) -> ValueRef;

    /* Selected entries from the downcasts. */
    fn LLVMIsATerminatorInst(ValueRef Inst) -> ValueRef;

    /** Writes a module to the specified path. Returns 0 on success. */
    fn LLVMWriteBitcodeToFile(ModuleRef M, sbuf Path) -> int;

    /** Creates target data from a target layout string. */
    fn LLVMCreateTargetData(sbuf StringRep) -> TargetDataRef;
    /** Adds the target data to the given pass manager. The pass manager
        references the target data only weakly. */
    fn LLVMAddTargetData(TargetDataRef TD, PassManagerRef PM);
    /** Returns the size of a type. FIXME: rv is actually a ULongLong! */
    fn LLVMStoreSizeOfType(TargetDataRef TD, TypeRef Ty) -> uint;
    /** Returns the alignment of a type. */
    fn LLVMPreferredAlignmentOfType(TargetDataRef TD, TypeRef Ty) -> uint;
    /** Disposes target data. */
    fn LLVMDisposeTargetData(TargetDataRef TD);

    /** Creates a pass manager. */
    fn LLVMCreatePassManager() -> PassManagerRef;
    /** Disposes a pass manager. */
    fn LLVMDisposePassManager(PassManagerRef PM);
    /** Runs a pass manager on a module. */
    fn LLVMRunPassManager(PassManagerRef PM, ModuleRef M) -> Bool;

    /** Adds a verification pass. */
    fn LLVMAddVerifierPass(PassManagerRef PM);

    fn LLVMAddGlobalOptimizerPass(PassManagerRef PM);
    fn LLVMAddIPSCCPPass(PassManagerRef PM);
    fn LLVMAddDeadArgEliminationPass(PassManagerRef PM);
    fn LLVMAddInstructionCombiningPass(PassManagerRef PM);
    fn LLVMAddCFGSimplificationPass(PassManagerRef PM);
    fn LLVMAddFunctionInliningPass(PassManagerRef PM);
    fn LLVMAddFunctionAttrsPass(PassManagerRef PM);
    fn LLVMAddScalarReplAggregatesPass(PassManagerRef PM);
    fn LLVMAddScalarReplAggregatesPassSSA(PassManagerRef PM);
    fn LLVMAddJumpThreadingPass(PassManagerRef PM);
    fn LLVMAddConstantPropagationPass(PassManagerRef PM);
    fn LLVMAddReassociatePass(PassManagerRef PM);
    fn LLVMAddLoopRotatePass(PassManagerRef PM);
    fn LLVMAddLICMPass(PassManagerRef PM);
    fn LLVMAddLoopUnswitchPass(PassManagerRef PM);
    fn LLVMAddLoopDeletionPass(PassManagerRef PM);
    fn LLVMAddLoopUnrollPass(PassManagerRef PM);
    fn LLVMAddGVNPass(PassManagerRef PM);
    fn LLVMAddMemCpyOptPass(PassManagerRef PM);
    fn LLVMAddSCCPPass(PassManagerRef PM);
    fn LLVMAddDeadStoreEliminationPass(PassManagerRef PM);
    fn LLVMAddStripDeadPrototypesPass(PassManagerRef PM);
    fn LLVMAddDeadTypeEliminationPass(PassManagerRef PM);
    fn LLVMAddConstantMergePass(PassManagerRef PM);
    fn LLVMAddArgumentPromotionPass(PassManagerRef PM);
    fn LLVMAddTailCallEliminationPass(PassManagerRef PM);
    fn LLVMAddIndVarSimplifyPass(PassManagerRef PM);
    fn LLVMAddAggressiveDCEPass(PassManagerRef PM);
    fn LLVMAddGlobalDCEPass(PassManagerRef PM);
    fn LLVMAddCorrelatedValuePropagationPass(PassManagerRef PM);
    fn LLVMAddPruneEHPass(PassManagerRef PM);
    fn LLVMAddSimplifyLibCallsPass(PassManagerRef PM);
    fn LLVMAddLoopIdiomPass(PassManagerRef PM);
    fn LLVMAddEarlyCSEPass(PassManagerRef PM);
    fn LLVMAddTypeBasedAliasAnalysisPass(PassManagerRef PM);
    fn LLVMAddBasicAliasAnalysisPass(PassManagerRef PM);

    fn LLVMAddStandardFunctionPasses(PassManagerRef PM,
                                     uint OptimizationLevel);
    fn LLVMAddStandardModulePasses(PassManagerRef PM,
                                   uint OptimizationLevel,
                                   Bool OptimizeSize,
                                   Bool UnitAtATime,
                                   Bool UnrollLoops,
                                   Bool SimplifyLibCalls,
                                   uint InliningThreshold);

    /** Destroys a memory buffer. */
    fn LLVMDisposeMemoryBuffer(MemoryBufferRef MemBuf);


    /* Stuff that's in rustllvm/ because it's not upstream yet. */

    type ObjectFileRef;
    type SectionIteratorRef;

    /** Opens an object file. */
    fn LLVMCreateObjectFile(MemoryBufferRef MemBuf) -> ObjectFileRef;
    /** Closes an object file. */
    fn LLVMDisposeObjectFile(ObjectFileRef ObjectFile);

    /** Enumerates the sections in an object file. */
    fn LLVMGetSections(ObjectFileRef ObjectFile) -> SectionIteratorRef;
    /** Destroys a section iterator. */
    fn LLVMDisposeSectionIterator(SectionIteratorRef SI);
    /** Returns true if the section iterator is at the end of the section
        list: */
    fn LLVMIsSectionIteratorAtEnd(ObjectFileRef ObjectFile,
                                  SectionIteratorRef SI) -> Bool;
    /** Moves the section iterator to point to the next section. */
    fn LLVMMoveToNextSection(SectionIteratorRef SI);
    /** Returns the current section name. */
    fn LLVMGetSectionName(SectionIteratorRef SI) -> sbuf;
    /** Returns the current section size.
        FIXME: The return value is actually a uint64_t! */
    fn LLVMGetSectionSize(SectionIteratorRef SI) -> uint;
    /** Returns the current section contents as a string buffer. */
    fn LLVMGetSectionContents(SectionIteratorRef SI) -> sbuf;

    /** Reads the given file and returns it as a memory buffer. Use
        LLVMDisposeMemoryBuffer() to get rid of it. */
    fn LLVMRustCreateMemoryBufferWithContentsOfFile(sbuf Path) ->
        MemoryBufferRef;

    /* FIXME: The FileType is an enum.*/
    fn LLVMRustWriteOutputFile(PassManagerRef PM, ModuleRef M,
                               sbuf Triple, sbuf Output,
                               int FileType);

    /** Returns a string describing the last error caused by an LLVMRust*
        call. */
    fn LLVMRustGetLastError() -> sbuf;

    /** Returns a string describing the hosts triple */
    fn LLVMRustGetHostTriple() -> sbuf;

    /** Parses the bitcode in the given memory buffer. */
    fn LLVMRustParseBitcode(MemoryBufferRef MemBuf) -> ModuleRef;

    /** FiXME: Hacky adaptor for lack of ULongLong in FFI: */
    fn LLVMRustConstSmallInt(TypeRef IntTy, uint N,
                             Bool SignExtend) -> ValueRef;

    /** Turn on LLVM pass-timing. */
    fn LLVMRustEnableTimePasses();

    /** Print the pass timings since static dtors aren't picking them up. */
    fn LLVMRustPrintPassTimings();

    /** Links LLVM modules together. `Src` is destroyed by this call and
        must never be referenced again. */
    fn LLVMLinkModules(ModuleRef Dest, ModuleRef Src) -> Bool;
}

native mod rustllvm = llvm_lib {
}

/* Slightly more terse object-interface to LLVM's 'builder' functions. For the
 * most part, build.Foo() wraps LLVMBuildFoo(), threading the correct
 * BuilderRef B into place.  A BuilderRef is a cursor-like LLVM value that
 * inserts instructions for a particular BasicBlockRef at a particular
 * position; for our purposes, it always inserts at the end of the basic block
 * it's attached to.  
 */

// FIXME: Do we want to support mutable object fields?
obj builder(BuilderRef B, @mutable bool terminated) {

    /* Terminators */
    fn RetVoid()  -> ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildRetVoid(B);
    }

    fn Ret(ValueRef V) -> ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildRet(B, V);
    }

    fn AggregateRet(vec[ValueRef] RetVals) -> ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildAggregateRet(B,
                                       vec::buf[ValueRef](RetVals),
                                       vec::len[ValueRef](RetVals));
    }

    fn Br(BasicBlockRef Dest) -> ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildBr(B, Dest);
    }

    fn CondBr(ValueRef If, BasicBlockRef Then,
              BasicBlockRef Else) -> ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildCondBr(B, If, Then, Else);
    }

    fn Switch(ValueRef V, BasicBlockRef Else, uint NumCases) -> ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildSwitch(B, V, Else, NumCases);
    }

    fn IndirectBr(ValueRef Addr, uint NumDests) -> ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildIndirectBr(B, Addr, NumDests);
    }

    fn Invoke(ValueRef Fn,
              vec[ValueRef] Args,
              BasicBlockRef Then,
              BasicBlockRef Catch) -> ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildInvoke(B, Fn,
                                 vec::buf[ValueRef](Args),
                                 vec::len[ValueRef](Args),
                                 Then, Catch,
                                 str::buf(""));
    }

    fn Unwind() -> ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildUnwind(B);
    }

    fn Unreachable() -> ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildUnreachable(B);
    }

    /* Arithmetic */
    fn Add(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildAdd(B, LHS, RHS, str::buf(""));
    }

    fn NSWAdd(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNSWAdd(B, LHS, RHS, str::buf(""));
    }

    fn NUWAdd(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNUWAdd(B, LHS, RHS, str::buf(""));
    }

    fn FAdd(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFAdd(B, LHS, RHS, str::buf(""));
    }

    fn Sub(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildSub(B, LHS, RHS, str::buf(""));
    }

    fn NSWSub(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNSWSub(B, LHS, RHS, str::buf(""));
    }

    fn NUWSub(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNUWSub(B, LHS, RHS, str::buf(""));
    }

    fn FSub(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFSub(B, LHS, RHS, str::buf(""));
    }

    fn Mul(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildMul(B, LHS, RHS, str::buf(""));
    }

    fn NSWMul(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNSWMul(B, LHS, RHS, str::buf(""));
    }

    fn NUWMul(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNUWMul(B, LHS, RHS, str::buf(""));
    }

    fn FMul(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFMul(B, LHS, RHS, str::buf(""));
    }

    fn UDiv(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildUDiv(B, LHS, RHS, str::buf(""));
    }

    fn SDiv(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildSDiv(B, LHS, RHS, str::buf(""));
    }

    fn ExactSDiv(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildExactSDiv(B, LHS, RHS, str::buf(""));
    }

    fn FDiv(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFDiv(B, LHS, RHS, str::buf(""));
    }

    fn URem(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildURem(B, LHS, RHS, str::buf(""));
    }

    fn SRem(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildSRem(B, LHS, RHS, str::buf(""));
    }

    fn FRem(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFRem(B, LHS, RHS, str::buf(""));
    }

    fn Shl(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildShl(B, LHS, RHS, str::buf(""));
    }

    fn LShr(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildLShr(B, LHS, RHS, str::buf(""));
    }

    fn AShr(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildAShr(B, LHS, RHS, str::buf(""));
    }

    fn And(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildAnd(B, LHS, RHS, str::buf(""));
    }

    fn Or(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildOr(B, LHS, RHS, str::buf(""));
    }

    fn Xor(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildXor(B, LHS, RHS, str::buf(""));
    }

    fn BinOp(Opcode Op, ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildBinOp(B, Op, LHS, RHS, str::buf(""));
    }

    fn Neg(ValueRef V) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNeg(B, V, str::buf(""));
    }

    fn NSWNeg(ValueRef V) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNSWNeg(B, V, str::buf(""));
    }

    fn NUWNeg(ValueRef V) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNUWNeg(B, V, str::buf(""));
    }
    fn FNeg(ValueRef V) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFNeg(B, V, str::buf(""));
    }
    fn Not(ValueRef V) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNot(B, V, str::buf(""));
    }

    /* Memory */
    fn Malloc(TypeRef Ty) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildMalloc(B, Ty, str::buf(""));
    }

    fn ArrayMalloc(TypeRef Ty, ValueRef Val) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildArrayMalloc(B, Ty, Val, str::buf(""));
    }

    fn Alloca(TypeRef Ty) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildAlloca(B, Ty, str::buf(""));
    }

    fn ArrayAlloca(TypeRef Ty, ValueRef Val) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildArrayAlloca(B, Ty, Val, str::buf(""));
    }

    fn Free(ValueRef PointerVal) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFree(B, PointerVal);
    }

    fn Load(ValueRef PointerVal) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildLoad(B, PointerVal, str::buf(""));
    }

    fn Store(ValueRef Val, ValueRef Ptr) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildStore(B, Val, Ptr);
    }

    fn GEP(ValueRef Pointer, vec[ValueRef] Indices) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildGEP(B, Pointer,
                              vec::buf[ValueRef](Indices),
                              vec::len[ValueRef](Indices),
                              str::buf(""));
    }

    fn InBoundsGEP(ValueRef Pointer, vec[ValueRef] Indices) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildInBoundsGEP(B, Pointer,
                                      vec::buf[ValueRef](Indices),
                                      vec::len[ValueRef](Indices),
                                      str::buf(""));
    }

    fn StructGEP(ValueRef Pointer, uint Idx) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildStructGEP(B, Pointer, Idx, str::buf(""));
    }

    fn GlobalString(sbuf _Str) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildGlobalString(B, _Str, str::buf(""));
    }

    fn GlobalStringPtr(sbuf _Str) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildGlobalStringPtr(B, _Str, str::buf(""));
    }

    /* Casts */
    fn Trunc(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildTrunc(B, Val, DestTy, str::buf(""));
    }

    fn ZExt(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildZExt(B, Val, DestTy, str::buf(""));
    }

    fn SExt(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildSExt(B, Val, DestTy, str::buf(""));
    }

    fn FPToUI(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFPToUI(B, Val, DestTy, str::buf(""));
    }

    fn FPToSI(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFPToSI(B, Val, DestTy, str::buf(""));
    }

    fn UIToFP(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildUIToFP(B, Val, DestTy, str::buf(""));
    }

    fn SIToFP(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildSIToFP(B, Val, DestTy, str::buf(""));
    }

    fn FPTrunc(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFPTrunc(B, Val, DestTy, str::buf(""));
    }

    fn FPExt(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFPExt(B, Val, DestTy, str::buf(""));
    }

    fn PtrToInt(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildPtrToInt(B, Val, DestTy, str::buf(""));
    }

    fn IntToPtr(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildIntToPtr(B, Val, DestTy, str::buf(""));
    }

    fn BitCast(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildBitCast(B, Val, DestTy, str::buf(""));
    }

    fn ZExtOrBitCast(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildZExtOrBitCast(B, Val, DestTy, str::buf(""));
    }

    fn SExtOrBitCast(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildSExtOrBitCast(B, Val, DestTy, str::buf(""));
    }

    fn TruncOrBitCast(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildTruncOrBitCast(B, Val, DestTy, str::buf(""));
    }

    fn Cast(Opcode Op, ValueRef Val, TypeRef DestTy, sbuf Name) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildCast(B, Op, Val, DestTy, str::buf(""));
    }

    fn PointerCast(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildPointerCast(B, Val, DestTy, str::buf(""));
    }

    fn IntCast(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildIntCast(B, Val, DestTy, str::buf(""));
    }

    fn FPCast(ValueRef Val, TypeRef DestTy) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFPCast(B, Val, DestTy, str::buf(""));
    }


    /* Comparisons */
    fn ICmp(uint Op, ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildICmp(B, Op, LHS, RHS, str::buf(""));
    }

    fn FCmp(uint Op, ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFCmp(B, Op, LHS, RHS, str::buf(""));
    }


    /* Miscellaneous instructions */
    fn Phi(TypeRef Ty, vec[ValueRef] vals,
           vec[BasicBlockRef] bbs) -> ValueRef {
        assert (!*terminated);
        auto phi = llvm::LLVMBuildPhi(B, Ty, str::buf(""));
        assert (vec::len[ValueRef](vals) == vec::len[BasicBlockRef](bbs));
        llvm::LLVMAddIncoming(phi,
                             vec::buf[ValueRef](vals),
                             vec::buf[BasicBlockRef](bbs),
                             vec::len[ValueRef](vals));
        ret phi;
    }

    fn AddIncomingToPhi(ValueRef phi,
                        vec[ValueRef] vals,
                        vec[BasicBlockRef] bbs) {
        assert (vec::len[ValueRef](vals) == vec::len[BasicBlockRef](bbs));
        llvm::LLVMAddIncoming(phi,
                             vec::buf[ValueRef](vals),
                             vec::buf[BasicBlockRef](bbs),
                             vec::len[ValueRef](vals));
    }

    fn Call(ValueRef Fn, vec[ValueRef] Args) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildCall(B, Fn,
                               vec::buf[ValueRef](Args),
                               vec::len[ValueRef](Args),
                               str::buf(""));
    }

    fn FastCall(ValueRef Fn, vec[ValueRef] Args) -> ValueRef {
        assert (!*terminated);
        auto v = llvm::LLVMBuildCall(B, Fn,
                                    vec::buf[ValueRef](Args),
                                    vec::len[ValueRef](Args),
                                    str::buf(""));
        llvm::LLVMSetInstructionCallConv(v, LLVMFastCallConv);
        ret v;
    }

    fn Select(ValueRef If, ValueRef Then, ValueRef Else) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildSelect(B, If, Then, Else, str::buf(""));
    }

    fn VAArg(ValueRef list, TypeRef Ty) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildVAArg(B, list, Ty, str::buf(""));
    }

    fn ExtractElement(ValueRef VecVal, ValueRef Index) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildExtractElement(B, VecVal, Index, str::buf(""));
    }

    fn InsertElement(ValueRef VecVal, ValueRef EltVal,
                     ValueRef Index) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildInsertElement(B, VecVal, EltVal, Index,
                                        str::buf(""));
    }

    fn ShuffleVector(ValueRef V1, ValueRef V2, ValueRef Mask) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildShuffleVector(B, V1, V2, Mask, str::buf(""));
    }

    fn ExtractValue(ValueRef AggVal, uint Index) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildExtractValue(B, AggVal, Index, str::buf(""));
    }

    fn InsertValue(ValueRef AggVal, ValueRef EltVal,
                   uint Index) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildInsertValue(B, AggVal, EltVal, Index,
                                       str::buf(""));
    }

    fn IsNull(ValueRef Val) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildIsNull(B, Val, str::buf(""));
    }

    fn IsNotNull(ValueRef Val) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildIsNotNull(B, Val, str::buf(""));
    }

    fn PtrDiff(ValueRef LHS, ValueRef RHS) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildPtrDiff(B, LHS, RHS, str::buf(""));
    }

    fn Trap() -> ValueRef {
        assert (!*terminated);
        let BasicBlockRef BB = llvm::LLVMGetInsertBlock(B);
        let ValueRef FN = llvm::LLVMGetBasicBlockParent(BB);
        let ModuleRef M = llvm::LLVMGetGlobalParent(FN);
        let ValueRef T = llvm::LLVMGetNamedFunction(M,
                                                    str::buf("llvm.trap"));
        assert (T as int != 0);
        let vec[ValueRef] Args = [];
        ret llvm::LLVMBuildCall(B, T,
                               vec::buf[ValueRef](Args),
                               vec::len[ValueRef](Args),
                               str::buf(""));
    }

    drop {
        llvm::LLVMDisposeBuilder(B);
    }
}

/* Memory-managed object interface to type handles. */

obj type_handle_dtor(TypeHandleRef TH) {
    drop { llvm::LLVMDisposeTypeHandle(TH); }
}

type type_handle = rec(TypeHandleRef llth, type_handle_dtor dtor);

fn mk_type_handle() -> type_handle {
    auto th = llvm::LLVMCreateTypeHandle(llvm::LLVMOpaqueType());
    ret rec(llth=th, dtor=type_handle_dtor(th));
}


state obj type_names(std::map::hashmap[TypeRef, str] type_names,
                    std::map::hashmap[str, TypeRef] named_types) {

    fn associate(str s, TypeRef t) {
        assert (!named_types.contains_key(s));
        assert (!type_names.contains_key(t));
        type_names.insert(t, s);
        named_types.insert(s, t);
    }

    fn type_has_name(TypeRef t) -> bool {
        ret type_names.contains_key(t);
    }

    fn get_name(TypeRef t) -> str {
        ret type_names.get(t);
    }

    fn name_has_type(str s) -> bool {
        ret named_types.contains_key(s);
    }

    fn get_type(str s) -> TypeRef {
        ret named_types.get(s);
    }
}

fn mk_type_names() -> type_names {
    auto nt = util::common::new_str_hash[TypeRef]();

    fn hash(&TypeRef t) -> uint {
        ret t as uint;
    }

    fn eq(&TypeRef a, &TypeRef b) -> bool {
        ret (a as uint) == (b as uint);
    }

    let std::map::hashfn[TypeRef] hasher = hash;
    let std::map::eqfn[TypeRef] eqer = eq;
    auto tn = std::map::mk_hashmap[TypeRef,str](hasher, eqer);

    ret type_names(tn, nt);
}

fn type_to_str(type_names names, TypeRef ty) -> str {
    let vec[TypeRef] v = [];
    ret type_to_str_inner(names, v, ty);
}

fn type_to_str_inner(type_names names,
                     vec[TypeRef] outer0, TypeRef ty) -> str {

    if (names.type_has_name(ty)) {
        ret names.get_name(ty);
    }

    auto outer = outer0 + [ty];

    let int kind = llvm::LLVMGetTypeKind(ty);

    fn tys_str(type_names names,
               vec[TypeRef] outer, vec[TypeRef] tys) -> str {
        let str s = "";
        let bool first = true;
        for (TypeRef t in tys) {
            if (first) {
                first = false;
            } else {
                s += ", ";
            }
            s += type_to_str_inner(names, outer, t);
        }
        ret s;
    }

    alt (kind) {

        // FIXME: more enum-as-int constants determined from Core::h;
        // horrible, horrible. Complete as needed.

        case (0) { ret "Void"; }
        case (1) { ret "Float"; }
        case (2) { ret "Double"; }
        case (3) { ret "X86_FP80"; }
        case (4) { ret "FP128"; }
        case (5) { ret "PPC_FP128"; }
        case (6) { ret "Label"; }

        case (7) {
            ret "i" + util::common::istr(llvm::LLVMGetIntTypeWidth(ty)
                                         as int);
        }

        case (8) {
            auto s = "fn(";
            let TypeRef out_ty = llvm::LLVMGetReturnType(ty);
            let uint n_args = llvm::LLVMCountParamTypes(ty);
            let vec[TypeRef] args =
                vec::init_elt[TypeRef](0 as TypeRef, n_args);
            llvm::LLVMGetParamTypes(ty, vec::buf[TypeRef](args));
            s += tys_str(names, outer, args);
            s += ") -> ";
            s += type_to_str_inner(names, outer, out_ty);
            ret s;
        }

        case (9) {
            let str s = "{";
            let uint n_elts = llvm::LLVMCountStructElementTypes(ty);
            let vec[TypeRef] elts =
                vec::init_elt[TypeRef](0 as TypeRef, n_elts);
            llvm::LLVMGetStructElementTypes(ty, vec::buf[TypeRef](elts));
            s += tys_str(names, outer, elts);
            s += "}";
            ret s;
        }

        case (10) { 
            auto el_ty = llvm::LLVMGetElementType(ty);
            ret "[" + type_to_str_inner(names, outer, el_ty) + "]"; 
        }

        case (11) {
            let uint i = 0u;
            for (TypeRef tout in outer0) {
                i += 1u;
                if (tout as int == ty as int) {
                    let uint n = vec::len[TypeRef](outer0) - i;
                    ret "*\\" + util::common::istr(n as int);
                }
            }
            ret "*" + type_to_str_inner(names, outer,
                                        llvm::LLVMGetElementType(ty));
        }

        case (12) { ret "Opaque"; }
        case (13) { ret "Vector"; }
        case (14) { ret "Metadata"; }
        case (_) {
            log_err #fmt("unknown TypeKind %d", kind as int);
            fail;
        }
    }
}

/* Memory-managed interface to target data. */

obj target_data_dtor(TargetDataRef TD) {
    drop { llvm::LLVMDisposeTargetData(TD); }
}

type target_data = rec(TargetDataRef lltd, target_data_dtor dtor);

fn mk_target_data(str string_rep) -> target_data {
    auto lltd = llvm::LLVMCreateTargetData(str::buf(string_rep));
    ret rec(lltd=lltd, dtor=target_data_dtor(lltd));
}

/* Memory-managed interface to pass managers. */

obj pass_manager_dtor(PassManagerRef PM) {
    drop { llvm::LLVMDisposePassManager(PM); }
}

type pass_manager = rec(PassManagerRef llpm, pass_manager_dtor dtor);

fn mk_pass_manager() -> pass_manager {
    auto llpm = llvm::LLVMCreatePassManager();
    ret rec(llpm=llpm, dtor=pass_manager_dtor(llpm));
}

/* Memory-managed interface to object files. */

obj object_file_dtor(ObjectFileRef ObjectFile) {
    drop { llvm::LLVMDisposeObjectFile(ObjectFile); }
}

type object_file = rec(ObjectFileRef llof, object_file_dtor dtor);

fn mk_object_file(MemoryBufferRef llmb) -> object_file {
    auto llof = llvm::LLVMCreateObjectFile(llmb);
    ret rec(llof=llof, dtor=object_file_dtor(llof));
}

/* Memory-managed interface to section iterators. */

obj section_iter_dtor(SectionIteratorRef SI) {
    drop { llvm::LLVMDisposeSectionIterator(SI); }
}

type section_iter = rec(SectionIteratorRef llsi, section_iter_dtor dtor);

fn mk_section_iter(ObjectFileRef llof) -> section_iter {
    auto llsi = llvm::LLVMGetSections(llof);
    ret rec(llsi=llsi, dtor=section_iter_dtor(llsi));
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
