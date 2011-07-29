import std::ivec;
import std::str;
import std::str::rustrt::sbuf;

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


const True: Bool = 1;
const False: Bool = 0;

// Consts for the LLVM CallConv type, pre-cast to uint.
// FIXME: figure out a way to merge these with the native
// typedef and/or a tag type in the native module below.

const LLVMCCallConv: uint = 0u;
const LLVMFastCallConv: uint = 8u;
const LLVMColdCallConv: uint = 9u;
const LLVMX86StdcallCallConv: uint = 64u;
const LLVMX86FastcallCallConv: uint = 65u;

const LLVMDefaultVisibility: uint = 0u;
const LLVMHiddenVisibility: uint = 1u;
const LLVMProtectedVisibility: uint = 2u;

const LLVMExternalLinkage: uint = 0u;
const LLVMAvailableExternallyLinkage: uint = 1u;
const LLVMLinkOnceAnyLinkage: uint = 2u;
const LLVMLinkOnceODRLinkage: uint = 3u;
const LLVMWeakAnyLinkage: uint = 4u;
const LLVMWeakODRLinkage: uint = 5u;
const LLVMAppendingLinkage: uint = 6u;
const LLVMInternalLinkage: uint = 7u;
const LLVMPrivateLinkage: uint = 8u;
const LLVMDLLImportLinkage: uint = 9u;
const LLVMDLLExportLinkage: uint = 10u;
const LLVMExternalWeakLinkage: uint = 11u;
const LLVMGhostLinkage: uint = 12u;
const LLVMCommonLinkage: uint = 13u;
const LLVMLinkerPrivateLinkage: uint = 14u;
const LLVMLinkerPrivateWeakLinkage: uint = 15u;

const LLVMZExtAttribute: uint = 1u;
const LLVMSExtAttribute: uint = 2u;
const LLVMNoReturnAttribute: uint = 4u;
const LLVMInRegAttribute: uint = 8u;
const LLVMStructRetAttribute: uint = 16u;
const LLVMNoUnwindAttribute: uint = 32u;
const LLVMNoAliasAttribute: uint = 64u;
const LLVMByValAttribute: uint = 128u;
const LLVMNestAttribute: uint = 256u;
const LLVMReadNoneAttribute: uint = 512u;
const LLVMReadOnlyAttribute: uint = 1024u;
const LLVMNoInlineAttribute: uint = 2048u;
const LLVMAlwaysInlineAttribute: uint = 4096u;
const LLVMOptimizeForSizeAttribute: uint = 8192u;
const LLVMStackProtectAttribute: uint = 16384u;
const LLVMStackProtectReqAttribute: uint = 32768u;
const LLVMAlignmentAttribute: uint = 2031616u;
 // 31 << 16
const LLVMNoCaptureAttribute: uint = 2097152u;
const LLVMNoRedZoneAttribute: uint = 4194304u;
const LLVMNoImplicitFloatAttribute: uint = 8388608u;
const LLVMNakedAttribute: uint = 16777216u;
const LLVMInlineHintAttribute: uint = 33554432u;
const LLVMStackAttribute: uint = 469762048u;
 // 7 << 26
const LLVMUWTableAttribute: uint = 1073741824u;
 // 1 << 30


 // Consts for the LLVM IntPredicate type, pre-cast to uint.
 // FIXME: as above.


const LLVMIntEQ: uint = 32u;
const LLVMIntNE: uint = 33u;
const LLVMIntUGT: uint = 34u;
const LLVMIntUGE: uint = 35u;
const LLVMIntULT: uint = 36u;
const LLVMIntULE: uint = 37u;
const LLVMIntSGT: uint = 38u;
const LLVMIntSGE: uint = 39u;
const LLVMIntSLT: uint = 40u;
const LLVMIntSLE: uint = 41u;


// Consts for the LLVM RealPredicate type, pre-case to uint.
// FIXME: as above.

const LLVMRealOEQ: uint = 1u;
const LLVMRealOGT: uint = 2u;
const LLVMRealOGE: uint = 3u;
const LLVMRealOLT: uint = 4u;
const LLVMRealOLE: uint = 5u;
const LLVMRealONE: uint = 6u;

const LLVMRealORD: uint = 7u;
const LLVMRealUNO: uint = 8u;
const LLVMRealUEQ: uint = 9u;
const LLVMRealUGT: uint = 10u;
const LLVMRealUGE: uint = 11u;
const LLVMRealULT: uint = 12u;
const LLVMRealULE: uint = 13u;
const LLVMRealUNE: uint = 14u;

#[link_args = "-Lrustllvm"]
native "cdecl" mod llvm = "rustllvm" {

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
    fn LLVMContextDispose(C: ContextRef);
    fn LLVMGetMDKindIDInContext(C: ContextRef, Name: sbuf, SLen: uint) ->
       uint;
    fn LLVMGetMDKindID(Name: sbuf, SLen: uint) -> uint;

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

    fn LLVMGetTypeKind(Ty: TypeRef) -> int;

    /** See llvm::LLVMType::getContext. */
    fn LLVMGetTypeContext(Ty: TypeRef) -> ContextRef;

    /* Operations on integer types */
    fn LLVMInt1TypeInContext(C: ContextRef) -> TypeRef;
    fn LLVMInt8TypeInContext(C: ContextRef) -> TypeRef;
    fn LLVMInt16TypeInContext(C: ContextRef) -> TypeRef;
    fn LLVMInt32TypeInContext(C: ContextRef) -> TypeRef;
    fn LLVMInt64TypeInContext(C: ContextRef) -> TypeRef;
    fn LLVMIntTypeInContext(C: ContextRef, NumBits: uint) -> TypeRef;

    fn LLVMInt1Type() -> TypeRef;
    fn LLVMInt8Type() -> TypeRef;
    fn LLVMInt16Type() -> TypeRef;
    fn LLVMInt32Type() -> TypeRef;
    fn LLVMInt64Type() -> TypeRef;
    fn LLVMIntType(NumBits: uint) -> TypeRef;
    fn LLVMGetIntTypeWidth(IntegerTy: TypeRef) -> uint;

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
                        ParamCount: uint, IsVarArg: Bool) -> TypeRef;
    fn LLVMIsFunctionVarArg(FunctionTy: TypeRef) -> Bool;
    fn LLVMGetReturnType(FunctionTy: TypeRef) -> TypeRef;
    fn LLVMCountParamTypes(FunctionTy: TypeRef) -> uint;
    fn LLVMGetParamTypes(FunctionTy: TypeRef, Dest: *TypeRef);

    /* Operations on struct types */
    fn LLVMStructTypeInContext(C: ContextRef, ElementTypes: *TypeRef,
                               ElementCount: uint, Packed: Bool) -> TypeRef;
    fn LLVMStructType(ElementTypes: *TypeRef, ElementCount: uint,
                      Packed: Bool) -> TypeRef;
    fn LLVMCountStructElementTypes(StructTy: TypeRef) -> uint;
    fn LLVMGetStructElementTypes(StructTy: TypeRef, Dest: *TypeRef);
    fn LLVMIsPackedStruct(StructTy: TypeRef) -> Bool;

    /* Operations on array, pointer, and vector types (sequence types) */
    fn LLVMArrayType(ElementType: TypeRef, ElementCount: uint) -> TypeRef;
    fn LLVMPointerType(ElementType: TypeRef, AddressSpace: uint) -> TypeRef;
    fn LLVMVectorType(ElementType: TypeRef, ElementCount: uint) -> TypeRef;

    fn LLVMGetElementType(Ty: TypeRef) -> TypeRef;
    fn LLVMGetArrayLength(ArrayTy: TypeRef) -> uint;
    fn LLVMGetPointerAddressSpace(PointerTy: TypeRef) -> uint;
    fn LLVMGetVectorSize(VectorTy: TypeRef) -> uint;

    /* Operations on other types */
    fn LLVMVoidTypeInContext(C: ContextRef) -> TypeRef;
    fn LLVMLabelTypeInContext(C: ContextRef) -> TypeRef;

    fn LLVMVoidType() -> TypeRef;
    fn LLVMLabelType() -> TypeRef;

    /* Operations on all values */
    fn LLVMTypeOf(Val: ValueRef) -> TypeRef;
    fn LLVMGetValueName(Val: ValueRef) -> sbuf;
    fn LLVMSetValueName(Val: ValueRef, Name: sbuf);
    fn LLVMDumpValue(Val: ValueRef);
    fn LLVMReplaceAllUsesWith(OldVal: ValueRef, NewVal: ValueRef);
    fn LLVMHasMetadata(Val: ValueRef) -> int;
    fn LLVMGetMetadata(Val: ValueRef, KindID: uint) -> ValueRef;
    fn LLVMSetMetadata(Val: ValueRef, KindID: uint, Node: ValueRef);

    /* Operations on Uses */
    fn LLVMGetFirstUse(Val: ValueRef) -> UseRef;
    fn LLVMGetNextUse(U: UseRef) -> UseRef;
    fn LLVMGetUser(U: UseRef) -> ValueRef;
    fn LLVMGetUsedValue(U: UseRef) -> ValueRef;

    /* Operations on Users */
    fn LLVMGetOperand(Val: ValueRef, Index: uint) -> ValueRef;

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
    fn LLVMMDStringInContext(C: ContextRef, Str: sbuf, SLen: uint) ->
       ValueRef;
    fn LLVMMDString(Str: sbuf, SLen: uint) -> ValueRef;
    fn LLVMMDNodeInContext(C: ContextRef, Vals: *ValueRef, Count: uint) ->
       ValueRef;
    fn LLVMMDNode(Vals: *ValueRef, Count: uint) -> ValueRef;

    /* Operations on scalar constants */
    fn LLVMConstInt(IntTy: TypeRef, N: ULongLong, SignExtend: Bool) ->
       ValueRef;
    // FIXME: radix is actually u8, but our native layer can't handle this
    // yet.  lucky for us we're little-endian. Small miracles.
    fn LLVMConstIntOfString(IntTy: TypeRef, Text: sbuf, Radix: int) ->
       ValueRef;
    fn LLVMConstIntOfStringAndSize(IntTy: TypeRef, Text: sbuf, SLen: uint,
                                   Radix: u8) -> ValueRef;
    fn LLVMConstReal(RealTy: TypeRef, N: f64) -> ValueRef;
    fn LLVMConstRealOfString(RealTy: TypeRef, Text: sbuf) -> ValueRef;
    fn LLVMConstRealOfStringAndSize(RealTy: TypeRef, Text: sbuf, SLen: uint)
       -> ValueRef;
    fn LLVMConstIntGetZExtValue(ConstantVal: ValueRef) -> ULongLong;
    fn LLVMConstIntGetSExtValue(ConstantVal: ValueRef) -> LongLong;


    /* Operations on composite constants */
    fn LLVMConstStringInContext(C: ContextRef, Str: sbuf, Length: uint,
                                DontNullTerminate: Bool) -> ValueRef;
    fn LLVMConstStructInContext(C: ContextRef, ConstantVals: *ValueRef,
                                Count: uint, Packed: Bool) -> ValueRef;

    fn LLVMConstString(Str: sbuf, Length: uint, DontNullTerminate: Bool) ->
       ValueRef;
    fn LLVMConstArray(ElementTy: TypeRef, ConstantVals: *ValueRef,
                      Length: uint) -> ValueRef;
    fn LLVMConstStruct(ConstantVals: *ValueRef, Count: uint, Packed: Bool) ->
       ValueRef;
    fn LLVMConstVector(ScalarConstantVals: *ValueRef, Size: uint) -> ValueRef;

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
                    NumIndices: uint) -> ValueRef;
    fn LLVMConstInBoundsGEP(ConstantVal: ValueRef, ConstantIndices: *uint,
                            NumIndices: uint) -> ValueRef;
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
                             NumIdx: uint) -> ValueRef;
    fn LLVMConstInsertValue(AggConstant: ValueRef,
                            ElementValueConstant: ValueRef, IdxList: *uint,
                            NumIdx: uint) -> ValueRef;
    fn LLVMConstInlineAsm(Ty: TypeRef, AsmString: sbuf, Constraints: sbuf,
                          HasSideEffects: Bool, IsAlignStack: Bool) ->
       ValueRef;
    fn LLVMBlockAddress(F: ValueRef, BB: BasicBlockRef) -> ValueRef;



    /* Operations on global variables, functions, and aliases (globals) */
    fn LLVMGetGlobalParent(Global: ValueRef) -> ModuleRef;
    fn LLVMIsDeclaration(Global: ValueRef) -> Bool;
    fn LLVMGetLinkage(Global: ValueRef) -> Linkage;
    fn LLVMSetLinkage(Global: ValueRef, Link: Linkage);
    fn LLVMGetSection(Global: ValueRef) -> sbuf;
    fn LLVMSetSection(Global: ValueRef, Section: sbuf);
    fn LLVMGetVisibility(Global: ValueRef) -> Visibility;
    fn LLVMSetVisibility(Global: ValueRef, Viz: Visibility);
    fn LLVMGetAlignment(Global: ValueRef) -> uint;
    fn LLVMSetAlignment(Global: ValueRef, Bytes: uint);


    /* Operations on global variables */
    fn LLVMAddGlobal(M: ModuleRef, Ty: TypeRef, Name: sbuf) -> ValueRef;
    fn LLVMAddGlobalInAddressSpace(M: ModuleRef, Ty: TypeRef, Name: sbuf,
                                   AddressSpace: uint) -> ValueRef;
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
    fn LLVMGetIntrinsicID(Fn: ValueRef) -> uint;
    fn LLVMGetFunctionCallConv(Fn: ValueRef) -> uint;
    fn LLVMSetFunctionCallConv(Fn: ValueRef, CC: uint);
    fn LLVMGetGC(Fn: ValueRef) -> sbuf;
    fn LLVMSetGC(Fn: ValueRef, Name: sbuf);
    fn LLVMAddFunctionAttr(Fn: ValueRef, PA: Attribute);
    fn LLVMGetFunctionAttr(Fn: ValueRef) -> Attribute;
    fn LLVMRemoveFunctionAttr(Fn: ValueRef, PA: Attribute);

    /* Operations on parameters */
    fn LLVMCountParams(Fn: ValueRef) -> uint;
    fn LLVMGetParams(Fn: ValueRef, Params: *ValueRef);
    fn LLVMGetParam(Fn: ValueRef, Index: uint) -> ValueRef;
    fn LLVMGetParamParent(Inst: ValueRef) -> ValueRef;
    fn LLVMGetFirstParam(Fn: ValueRef) -> ValueRef;
    fn LLVMGetLastParam(Fn: ValueRef) -> ValueRef;
    fn LLVMGetNextParam(Arg: ValueRef) -> ValueRef;
    fn LLVMGetPreviousParam(Arg: ValueRef) -> ValueRef;
    fn LLVMAddAttribute(Arg: ValueRef, PA: Attribute);
    fn LLVMRemoveAttribute(Arg: ValueRef, PA: Attribute);
    fn LLVMGetAttribute(Arg: ValueRef) -> Attribute;
    fn LLVMSetParamAlignment(Arg: ValueRef, align: uint);

    /* Operations on basic blocks */
    fn LLVMBasicBlockAsValue(BB: BasicBlockRef) -> ValueRef;
    fn LLVMValueIsBasicBlock(Val: ValueRef) -> Bool;
    fn LLVMValueAsBasicBlock(Val: ValueRef) -> BasicBlockRef;
    fn LLVMGetBasicBlockParent(BB: BasicBlockRef) -> ValueRef;
    fn LLVMCountBasicBlocks(Fn: ValueRef) -> uint;
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
    fn LLVMSetInstructionCallConv(Instr: ValueRef, CC: uint);
    fn LLVMGetInstructionCallConv(Instr: ValueRef) -> uint;
    fn LLVMAddInstrAttribute(Instr: ValueRef, index: uint, IA: Attribute);
    fn LLVMRemoveInstrAttribute(Instr: ValueRef, index: uint, IA: Attribute);
    fn LLVMSetInstrParamAlignment(Instr: ValueRef, index: uint, align: uint);

    /* Operations on call instructions (only) */
    fn LLVMIsTailCall(CallInst: ValueRef) -> Bool;
    fn LLVMSetTailCall(CallInst: ValueRef, IsTailCall: Bool);

    /* Operations on phi nodes */
    fn LLVMAddIncoming(PhiNode: ValueRef, IncomingValues: *ValueRef,
                       IncomingBlocks: *BasicBlockRef, Count: uint);
    fn LLVMCountIncoming(PhiNode: ValueRef) -> uint;
    fn LLVMGetIncomingValue(PhiNode: ValueRef, Index: uint) -> ValueRef;
    fn LLVMGetIncomingBlock(PhiNode: ValueRef, Index: uint) -> BasicBlockRef;

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
    fn LLVMBuildAggregateRet(B: BuilderRef, RetVals: *ValueRef, N: uint) ->
       ValueRef;
    fn LLVMBuildBr(B: BuilderRef, Dest: BasicBlockRef) -> ValueRef;
    fn LLVMBuildCondBr(B: BuilderRef, If: ValueRef, Then: BasicBlockRef,
                       Else: BasicBlockRef) -> ValueRef;
    fn LLVMBuildSwitch(B: BuilderRef, V: ValueRef, Else: BasicBlockRef,
                       NumCases: uint) -> ValueRef;
    fn LLVMBuildIndirectBr(B: BuilderRef, Addr: ValueRef, NumDests: uint) ->
       ValueRef;
    fn LLVMBuildInvoke(B: BuilderRef, Fn: ValueRef, Args: *ValueRef,
                       NumArgs: uint, Then: BasicBlockRef,
                       Catch: BasicBlockRef, Name: sbuf) -> ValueRef;
    fn LLVMBuildUnwind(B: BuilderRef) -> ValueRef;
    fn LLVMBuildUnreachable(B: BuilderRef) -> ValueRef;

    /* Add a case to the switch instruction */
    fn LLVMAddCase(Switch: ValueRef, OnVal: ValueRef, Dest: BasicBlockRef);

    /* Add a destination to the indirectbr instruction */
    fn LLVMAddDestination(IndirectBr: ValueRef, Dest: BasicBlockRef);

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
                    NumIndices: uint, Name: sbuf) -> ValueRef;
    fn LLVMBuildInBoundsGEP(B: BuilderRef, Pointer: ValueRef,
                            Indices: *ValueRef, NumIndices: uint, Name: sbuf)
       -> ValueRef;
    fn LLVMBuildStructGEP(B: BuilderRef, Pointer: ValueRef, Idx: uint,
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
    fn LLVMBuildICmp(B: BuilderRef, Op: uint, LHS: ValueRef, RHS: ValueRef,
                     Name: sbuf) -> ValueRef;
    fn LLVMBuildFCmp(B: BuilderRef, Op: uint, LHS: ValueRef, RHS: ValueRef,
                     Name: sbuf) -> ValueRef;

    /* Miscellaneous instructions */
    fn LLVMBuildPhi(B: BuilderRef, Ty: TypeRef, Name: sbuf) -> ValueRef;
    fn LLVMBuildCall(B: BuilderRef, Fn: ValueRef, Args: *ValueRef,
                     NumArgs: uint, Name: sbuf) -> ValueRef;
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
    fn LLVMBuildExtractValue(B: BuilderRef, AggVal: ValueRef, Index: uint,
                             Name: sbuf) -> ValueRef;
    fn LLVMBuildInsertValue(B: BuilderRef, AggVal: ValueRef, EltVal: ValueRef,
                            Index: uint, Name: sbuf) -> ValueRef;

    fn LLVMBuildIsNull(B: BuilderRef, Val: ValueRef, Name: sbuf) -> ValueRef;
    fn LLVMBuildIsNotNull(B: BuilderRef, Val: ValueRef, Name: sbuf) ->
       ValueRef;
    fn LLVMBuildPtrDiff(B: BuilderRef, LHS: ValueRef, RHS: ValueRef,
                        Name: sbuf) -> ValueRef;

    /* Selected entries from the downcasts. */
    fn LLVMIsATerminatorInst(Inst: ValueRef) -> ValueRef;

    /** Writes a module to the specified path. Returns 0 on success. */
    fn LLVMWriteBitcodeToFile(M: ModuleRef, Path: sbuf) -> int;

    /** Creates target data from a target layout string. */
    fn LLVMCreateTargetData(StringRep: sbuf) -> TargetDataRef;
    /** Adds the target data to the given pass manager. The pass manager
        references the target data only weakly. */
    fn LLVMAddTargetData(TD: TargetDataRef, PM: PassManagerRef);
    /** Returns the size of a type. FIXME: rv is actually a ULongLong! */
    fn LLVMStoreSizeOfType(TD: TargetDataRef, Ty: TypeRef) -> uint;
    /** Returns the alignment of a type. */
    fn LLVMPreferredAlignmentOfType(TD: TargetDataRef, Ty: TypeRef) -> uint;
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

    fn LLVMAddStandardFunctionPasses(PM: PassManagerRef,
                                     OptimizationLevel: uint);
    fn LLVMAddStandardModulePasses(PM: PassManagerRef,
                                   OptimizationLevel: uint,
                                   OptimizeSize: Bool, UnitAtATime: Bool,
                                   UnrollLoops: Bool, SimplifyLibCalls: Bool,
                                   InliningThreshold: uint);

    /** Destroys a memory buffer. */
    fn LLVMDisposeMemoryBuffer(MemBuf: MemoryBufferRef);


    /* Stuff that's in rustllvm/ because it's not upstream yet. */

    type ObjectFileRef;
    type SectionIteratorRef;

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
    /** Returns the current section size.
        FIXME: The return value is actually a uint64_t! */
    fn LLVMGetSectionSize(SI: SectionIteratorRef) -> uint;
    /** Returns the current section contents as a string buffer. */
    fn LLVMGetSectionContents(SI: SectionIteratorRef) -> sbuf;

    /** Reads the given file and returns it as a memory buffer. Use
        LLVMDisposeMemoryBuffer() to get rid of it. */
    fn LLVMRustCreateMemoryBufferWithContentsOfFile(Path: sbuf) ->
       MemoryBufferRef;

    /* FIXME: The FileType is an enum.*/
    fn LLVMRustWriteOutputFile(PM: PassManagerRef, M: ModuleRef, Triple: sbuf,
                               Output: sbuf, FileType: int, OptLevel: int);

    /** Returns a string describing the last error caused by an LLVMRust*
        call. */
    fn LLVMRustGetLastError() -> sbuf;

    /** Returns a string describing the hosts triple */
    fn LLVMRustGetHostTriple() -> sbuf;

    /** Parses the bitcode in the given memory buffer. */
    fn LLVMRustParseBitcode(MemBuf: MemoryBufferRef) -> ModuleRef;

    /** FiXME: Hacky adaptor for lack of ULongLong in FFI: */
    fn LLVMRustConstSmallInt(IntTy: TypeRef, N: uint, SignExtend: Bool) ->
       ValueRef;

    /** Turn on LLVM pass-timing. */
    fn LLVMRustEnableTimePasses();

    /** Print the pass timings since static dtors aren't picking them up. */
    fn LLVMRustPrintPassTimings();

    fn LLVMStructCreateNamed(C: ContextRef, Name: sbuf) -> TypeRef;

    fn LLVMStructSetBody(StructTy: TypeRef, ElementTypes: *TypeRef,
                         ElementCount: uint, Packed: Bool);

    fn LLVMConstNamedStruct(S: TypeRef, ConstantVals: *ValueRef, Count: uint)
       -> ValueRef;

    /** Links LLVM modules together. `Src` is destroyed by this call and
        must never be referenced again. */
    fn LLVMLinkModules(Dest: ModuleRef, Src: ModuleRef) -> Bool;
}

/* Slightly more terse object-interface to LLVM's 'builder' functions. For the
 * most part, build.Foo() wraps LLVMBuildFoo(), threading the correct
 * BuilderRef B into place.  A BuilderRef is a cursor-like LLVM value that
 * inserts instructions for a particular BasicBlockRef at a particular
 * position; for our purposes, it always inserts at the end of the basic block
 * it's attached to.
 */

resource BuilderRef_res(B: BuilderRef) {
    llvm::LLVMDisposeBuilder(B);
}

obj builder(B: BuilderRef, terminated: @mutable bool,
            // Stored twice so that we don't have to constantly deref
            res: @BuilderRef_res) {
    /* Terminators */
    fn RetVoid() -> ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildRetVoid(B);
    }

    fn Ret(V: ValueRef) -> ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildRet(B, V);
    }

    fn AggregateRet(RetVals: &ValueRef[]) -> ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildAggregateRet(B, ivec::to_ptr(RetVals),
                                        ivec::len(RetVals));
    }

    fn Br(Dest: BasicBlockRef) -> ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildBr(B, Dest);
    }

    fn CondBr(If: ValueRef, Then: BasicBlockRef, Else: BasicBlockRef) ->
       ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildCondBr(B, If, Then, Else);
    }

    fn Switch(V: ValueRef, Else: BasicBlockRef, NumCases: uint) -> ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildSwitch(B, V, Else, NumCases);
    }

    fn IndirectBr(Addr: ValueRef, NumDests: uint) -> ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildIndirectBr(B, Addr, NumDests);
    }

    fn Invoke(Fn: ValueRef, Args: &ValueRef[], Then: BasicBlockRef,
              Catch: BasicBlockRef) -> ValueRef {
        assert (!*terminated);
        *terminated = true;
        ret llvm::LLVMBuildInvoke(B, Fn, ivec::to_ptr(Args), ivec::len(Args),
                                  Then, Catch, str::buf(""));
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
    fn Add(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildAdd(B, LHS, RHS, str::buf(""));
    }

    fn NSWAdd(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNSWAdd(B, LHS, RHS, str::buf(""));
    }

    fn NUWAdd(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNUWAdd(B, LHS, RHS, str::buf(""));
    }

    fn FAdd(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFAdd(B, LHS, RHS, str::buf(""));
    }

    fn Sub(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildSub(B, LHS, RHS, str::buf(""));
    }

    fn NSWSub(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNSWSub(B, LHS, RHS, str::buf(""));
    }

    fn NUWSub(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNUWSub(B, LHS, RHS, str::buf(""));
    }

    fn FSub(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFSub(B, LHS, RHS, str::buf(""));
    }

    fn Mul(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildMul(B, LHS, RHS, str::buf(""));
    }

    fn NSWMul(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNSWMul(B, LHS, RHS, str::buf(""));
    }

    fn NUWMul(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNUWMul(B, LHS, RHS, str::buf(""));
    }

    fn FMul(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFMul(B, LHS, RHS, str::buf(""));
    }

    fn UDiv(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildUDiv(B, LHS, RHS, str::buf(""));
    }

    fn SDiv(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildSDiv(B, LHS, RHS, str::buf(""));
    }

    fn ExactSDiv(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildExactSDiv(B, LHS, RHS, str::buf(""));
    }

    fn FDiv(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFDiv(B, LHS, RHS, str::buf(""));
    }

    fn URem(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildURem(B, LHS, RHS, str::buf(""));
    }

    fn SRem(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildSRem(B, LHS, RHS, str::buf(""));
    }

    fn FRem(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFRem(B, LHS, RHS, str::buf(""));
    }

    fn Shl(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildShl(B, LHS, RHS, str::buf(""));
    }

    fn LShr(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildLShr(B, LHS, RHS, str::buf(""));
    }

    fn AShr(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildAShr(B, LHS, RHS, str::buf(""));
    }

    fn And(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildAnd(B, LHS, RHS, str::buf(""));
    }

    fn Or(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildOr(B, LHS, RHS, str::buf(""));
    }

    fn Xor(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildXor(B, LHS, RHS, str::buf(""));
    }

    fn BinOp(Op: Opcode, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildBinOp(B, Op, LHS, RHS, str::buf(""));
    }

    fn Neg(V: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNeg(B, V, str::buf(""));
    }

    fn NSWNeg(V: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNSWNeg(B, V, str::buf(""));
    }

    fn NUWNeg(V: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNUWNeg(B, V, str::buf(""));
    }
    fn FNeg(V: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFNeg(B, V, str::buf(""));
    }
    fn Not(V: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildNot(B, V, str::buf(""));
    }

    /* Memory */
    fn Malloc(Ty: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildMalloc(B, Ty, str::buf(""));
    }

    fn ArrayMalloc(Ty: TypeRef, Val: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildArrayMalloc(B, Ty, Val, str::buf(""));
    }

    fn Alloca(Ty: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildAlloca(B, Ty, str::buf(""));
    }

    fn ArrayAlloca(Ty: TypeRef, Val: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildArrayAlloca(B, Ty, Val, str::buf(""));
    }

    fn Free(PointerVal: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFree(B, PointerVal);
    }

    fn Load(PointerVal: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildLoad(B, PointerVal, str::buf(""));
    }

    fn Store(Val: ValueRef, Ptr: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildStore(B, Val, Ptr);
    }

    fn GEP(Pointer: ValueRef, Indices: &ValueRef[]) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildGEP(B, Pointer, ivec::to_ptr(Indices),
                               ivec::len(Indices), str::buf(""));
    }

    fn InBoundsGEP(Pointer: ValueRef, Indices: &ValueRef[]) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildInBoundsGEP(B, Pointer, ivec::to_ptr(Indices),
                                       ivec::len(Indices), str::buf(""));
    }

    fn StructGEP(Pointer: ValueRef, Idx: uint) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildStructGEP(B, Pointer, Idx, str::buf(""));
    }

    fn GlobalString(_Str: sbuf) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildGlobalString(B, _Str, str::buf(""));
    }

    fn GlobalStringPtr(_Str: sbuf) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildGlobalStringPtr(B, _Str, str::buf(""));
    }

    /* Casts */
    fn Trunc(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildTrunc(B, Val, DestTy, str::buf(""));
    }

    fn ZExt(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildZExt(B, Val, DestTy, str::buf(""));
    }

    fn SExt(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildSExt(B, Val, DestTy, str::buf(""));
    }

    fn FPToUI(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFPToUI(B, Val, DestTy, str::buf(""));
    }

    fn FPToSI(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFPToSI(B, Val, DestTy, str::buf(""));
    }

    fn UIToFP(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildUIToFP(B, Val, DestTy, str::buf(""));
    }

    fn SIToFP(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildSIToFP(B, Val, DestTy, str::buf(""));
    }

    fn FPTrunc(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFPTrunc(B, Val, DestTy, str::buf(""));
    }

    fn FPExt(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFPExt(B, Val, DestTy, str::buf(""));
    }

    fn PtrToInt(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildPtrToInt(B, Val, DestTy, str::buf(""));
    }

    fn IntToPtr(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildIntToPtr(B, Val, DestTy, str::buf(""));
    }

    fn BitCast(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildBitCast(B, Val, DestTy, str::buf(""));
    }

    fn ZExtOrBitCast(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildZExtOrBitCast(B, Val, DestTy, str::buf(""));
    }

    fn SExtOrBitCast(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildSExtOrBitCast(B, Val, DestTy, str::buf(""));
    }

    fn TruncOrBitCast(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildTruncOrBitCast(B, Val, DestTy, str::buf(""));
    }

    fn Cast(Op: Opcode, Val: ValueRef, DestTy: TypeRef, Name: sbuf) ->
       ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildCast(B, Op, Val, DestTy, str::buf(""));
    }

    fn PointerCast(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildPointerCast(B, Val, DestTy, str::buf(""));
    }

    fn IntCast(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildIntCast(B, Val, DestTy, str::buf(""));
    }

    fn FPCast(Val: ValueRef, DestTy: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFPCast(B, Val, DestTy, str::buf(""));
    }


    /* Comparisons */
    fn ICmp(Op: uint, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildICmp(B, Op, LHS, RHS, str::buf(""));
    }

    fn FCmp(Op: uint, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildFCmp(B, Op, LHS, RHS, str::buf(""));
    }


    /* Miscellaneous instructions */
    fn Phi(Ty: TypeRef, vals: &ValueRef[], bbs: &BasicBlockRef[]) ->
       ValueRef {
        assert (!*terminated);
        let phi = llvm::LLVMBuildPhi(B, Ty, str::buf(""));
        assert (ivec::len[ValueRef](vals) == ivec::len[BasicBlockRef](bbs));
        llvm::LLVMAddIncoming(phi, ivec::to_ptr(vals), ivec::to_ptr(bbs),
                              ivec::len(vals));
        ret phi;
    }

    fn AddIncomingToPhi(phi: ValueRef, vals: &ValueRef[],
                        bbs: &BasicBlockRef[]) {
        assert (ivec::len[ValueRef](vals) == ivec::len[BasicBlockRef](bbs));
        llvm::LLVMAddIncoming(phi, ivec::to_ptr(vals), ivec::to_ptr(bbs),
                              ivec::len(vals));
    }

    fn Call(Fn: ValueRef, Args: &ValueRef[]) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildCall(B, Fn, ivec::to_ptr(Args), ivec::len(Args),
                                str::buf(""));
    }

    fn FastCall(Fn: ValueRef, Args: &ValueRef[]) -> ValueRef {
        assert (!*terminated);
        let v =
            llvm::LLVMBuildCall(B, Fn, ivec::to_ptr(Args), ivec::len(Args),
                                str::buf(""));
        llvm::LLVMSetInstructionCallConv(v, LLVMFastCallConv);
        ret v;
    }

    fn CallWithConv(Fn: ValueRef, Args: &ValueRef[], Conv: uint) -> ValueRef {
        assert (!*terminated);
        let v =
            llvm::LLVMBuildCall(B, Fn, ivec::to_ptr(Args), ivec::len(Args),
                                str::buf(""));
        llvm::LLVMSetInstructionCallConv(v, Conv);
        ret v;
    }

    fn Select(If: ValueRef, Then: ValueRef, Else: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildSelect(B, If, Then, Else, str::buf(""));
    }

    fn VAArg(list: ValueRef, Ty: TypeRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildVAArg(B, list, Ty, str::buf(""));
    }

    fn ExtractElement(VecVal: ValueRef, Index: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildExtractElement(B, VecVal, Index, str::buf(""));
    }

    fn InsertElement(VecVal: ValueRef, EltVal: ValueRef, Index: ValueRef) ->
       ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildInsertElement(B, VecVal, EltVal, Index,
                                         str::buf(""));
    }

    fn ShuffleVector(V1: ValueRef, V2: ValueRef, Mask: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildShuffleVector(B, V1, V2, Mask, str::buf(""));
    }

    fn ExtractValue(AggVal: ValueRef, Index: uint) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildExtractValue(B, AggVal, Index, str::buf(""));
    }

    fn InsertValue(AggVal: ValueRef, EltVal: ValueRef, Index: uint) ->
       ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildInsertValue(B, AggVal, EltVal, Index,
                                       str::buf(""));
    }

    fn IsNull(Val: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildIsNull(B, Val, str::buf(""));
    }

    fn IsNotNull(Val: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildIsNotNull(B, Val, str::buf(""));
    }

    fn PtrDiff(LHS: ValueRef, RHS: ValueRef) -> ValueRef {
        assert (!*terminated);
        ret llvm::LLVMBuildPtrDiff(B, LHS, RHS, str::buf(""));
    }

    fn Trap() -> ValueRef {
        assert (!*terminated);
        let BB: BasicBlockRef = llvm::LLVMGetInsertBlock(B);
        let FN: ValueRef = llvm::LLVMGetBasicBlockParent(BB);
        let M: ModuleRef = llvm::LLVMGetGlobalParent(FN);
        let T: ValueRef =
            llvm::LLVMGetNamedFunction(M, str::buf("llvm.trap"));
        assert (T as int != 0);
        let Args: ValueRef[] = ~[];
        ret llvm::LLVMBuildCall(B, T, ivec::to_ptr(Args), ivec::len(Args),
                                str::buf(""));
    }

    fn is_terminated() -> bool {
        ret *terminated;
    }
}

fn new_builder(llbb: BasicBlockRef) -> builder {
    let llbuild: BuilderRef = llvm::LLVMCreateBuilder();
    llvm::LLVMPositionBuilderAtEnd(llbuild, llbb);
    ret builder(llbuild, @mutable false, @BuilderRef_res(llbuild));
}

/* Memory-managed object interface to type handles. */

obj type_names(type_names: std::map::hashmap[TypeRef, str],
               named_types: std::map::hashmap[str, TypeRef]) {

    fn associate(s: str, t: TypeRef) {
        assert (!named_types.contains_key(s));
        assert (!type_names.contains_key(t));
        type_names.insert(t, s);
        named_types.insert(s, t);
    }

    fn type_has_name(t: TypeRef) -> bool { ret type_names.contains_key(t); }

    fn get_name(t: TypeRef) -> str { ret type_names.get(t); }

    fn name_has_type(s: str) -> bool { ret named_types.contains_key(s); }

    fn get_type(s: str) -> TypeRef { ret named_types.get(s); }
}

fn mk_type_names() -> type_names {
    let nt = std::map::new_str_hash[TypeRef]();

    fn hash(t: &TypeRef) -> uint { ret t as uint; }

    fn eq(a: &TypeRef, b: &TypeRef) -> bool { ret a as uint == b as uint; }

    let hasher: std::map::hashfn[TypeRef] = hash;
    let eqer: std::map::eqfn[TypeRef] = eq;
    let tn = std::map::mk_hashmap[TypeRef, str](hasher, eqer);

    ret type_names(tn, nt);
}

fn type_to_str(names: type_names, ty: TypeRef) -> str {
    ret type_to_str_inner(names, ~[], ty);
}

fn type_to_str_inner(names: type_names, outer0: &TypeRef[], ty: TypeRef) ->
   str {

    if names.type_has_name(ty) { ret names.get_name(ty); }

    let outer = outer0 + ~[ty];

    let kind: int = llvm::LLVMGetTypeKind(ty);

    fn tys_str(names: type_names, outer: &TypeRef[], tys: &TypeRef[]) -> str {
        let s: str = "";
        let first: bool = true;
        for t: TypeRef  in tys {
            if first { first = false; } else { s += ", "; }
            s += type_to_str_inner(names, outer, t);
        }
        ret s;
    }


    alt kind {


      // FIXME: more enum-as-int constants determined from Core::h;
      // horrible, horrible. Complete as needed.

      0 {
        ret "Void";
      }
      1 { ret "Float"; }
      2 { ret "Double"; }
      3 { ret "X86_FP80"; }
      4 { ret "FP128"; }
      5 { ret "PPC_FP128"; }
      6 { ret "Label"; }


      7 {
        ret "i" + std::int::str(llvm::LLVMGetIntTypeWidth(ty) as int);
      }


      8 {
        let s = "fn(";
        let out_ty: TypeRef = llvm::LLVMGetReturnType(ty);
        let n_args: uint = llvm::LLVMCountParamTypes(ty);
        let args: TypeRef[] = ivec::init_elt[TypeRef](0 as TypeRef, n_args);
        llvm::LLVMGetParamTypes(ty, ivec::to_ptr(args));
        s += tys_str(names, outer, args);
        s += ") -> ";
        s += type_to_str_inner(names, outer, out_ty);
        ret s;
      }


      9 {
        let s: str = "{";
        let n_elts: uint = llvm::LLVMCountStructElementTypes(ty);
        let elts: TypeRef[] = ivec::init_elt[TypeRef](0 as TypeRef, n_elts);
        llvm::LLVMGetStructElementTypes(ty, ivec::to_ptr(elts));
        s += tys_str(names, outer, elts);
        s += "}";
        ret s;
      }


      10 {
        let el_ty = llvm::LLVMGetElementType(ty);
        ret "[" + type_to_str_inner(names, outer, el_ty) + "]";
      }


      11 {
        let i: uint = 0u;
        for tout: TypeRef  in outer0 {
            i += 1u;
            if tout as int == ty as int {
                let n: uint = ivec::len[TypeRef](outer0) - i;
                ret "*\\" + std::int::str(n as int);
            }
        }
        ret "*" +
                type_to_str_inner(names, outer, llvm::LLVMGetElementType(ty));
      }


      12 {
        ret "Opaque";
      }
      13 { ret "Vector"; }
      14 { ret "Metadata"; }
      _ { log_err #fmt("unknown TypeKind %d", kind as int); fail; }
    }
}

fn float_width(llt: TypeRef) -> uint {
    ret alt llvm::LLVMGetTypeKind(llt) {
          1 { 32u }
          2 { 64u }
          3 { 80u }
          4 | 5 { 128u }
          _ { fail "llvm_float_width called on a non-float type" }
        };
}

fn fn_ty_param_tys(fn_ty: TypeRef) -> TypeRef[] {
    let args = ivec::init_elt(0 as TypeRef, llvm::LLVMCountParamTypes(fn_ty));
    llvm::LLVMGetParamTypes(fn_ty, ivec::to_ptr(args));
    ret args;
}


/* Memory-managed interface to target data. */

resource target_data_res(TD: TargetDataRef) {
    llvm::LLVMDisposeTargetData(TD);
}

type target_data = {lltd: TargetDataRef, dtor: @target_data_res};

fn mk_target_data(string_rep: str) -> target_data {
    let lltd = llvm::LLVMCreateTargetData(str::buf(string_rep));
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

fn mk_object_file(llmb: MemoryBufferRef) -> object_file {
    let llof = llvm::LLVMCreateObjectFile(llmb);
    ret {llof: llof, dtor: @object_file_res(llof)};
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
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
