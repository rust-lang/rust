import std._str.rustrt.sbuf;
import std._vec.rustrt.vbuf;

import llvm.ModuleRef;
import llvm.ContextRef;
import llvm.TypeRef;
import llvm.TypeHandleRef;
import llvm.ValueRef;
import llvm.BasicBlockRef;
import llvm.BuilderRef;
import llvm.ModuleProviderRef;
import llvm.MemoryBufferRef;
import llvm.PassManagerRef;
import llvm.UseRef;
import llvm.Linkage;
import llvm.Attribute;
import llvm.Visibility;
import llvm.CallConv;
import llvm.IntPredicate;
import llvm.RealPredicate;
import llvm.Opcode;

type ULongLong = u64;
type LongLong = i64;
type Long = i32;
type Bool = int;

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

    /* Operations on union types */
    fn LLVMUnionTypeInContext(ContextRef C, vbuf ElementTypes,
                              uint ElementCount) -> TypeRef;
    fn LLVMUnionType(vbuf ElementTypes, uint ElementCount) -> TypeRef;
    fn LLVMCountUnionElementTypes(TypeRef UnionTy) -> uint;
    fn LLVMGetUnionElementTypes(TypeRef UnionTy, vbuf Dest);

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
    fn LLVMConstIntOfString(TypeRef IntTy, sbuf Text, u8 Radix) -> ValueRef;
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
    fn LLVMConstUnion(TypeRef Ty, ValueRef Val) -> ValueRef;

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
    fn LLVMBuildICmp(BuilderRef B, IntPredicate Op,
                     ValueRef LHS, ValueRef RHS,
                     sbuf Name) -> ValueRef;
    fn LLVMBuildFCmp(BuilderRef B, RealPredicate Op,
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
    fn LLVMBuildVAArg(BuilderRef B, ValueRef List, TypeRef Ty,
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


    /** Writes a module to the specified path. Returns 0 on success. */
    fn LLVMWriteBitcodeToFile(ModuleRef M, sbuf Path) -> int;
}

/* Slightly more terse object-interface to LLVM's 'builder' functions. */

obj builder(BuilderRef B) {

    /* Terminators */
    fn RetVoid()  -> ValueRef {
        ret llvm.LLVMBuildRetVoid(B);
    }

    fn Ret(ValueRef V) -> ValueRef {
        ret llvm.LLVMBuildRet(B, V);
    }

    fn AggregateRet(vbuf RetVals, uint N) -> ValueRef {
        ret llvm.LLVMBuildAggregateRet(B, RetVals, N);
    }

    fn Br(BasicBlockRef Dest) -> ValueRef {
        ret llvm.LLVMBuildBr(B, Dest);
    }

    fn CondBr(ValueRef If, BasicBlockRef Then,
              BasicBlockRef Else) -> ValueRef {
        ret llvm.LLVMBuildCondBr(B, If, Then, Else);
    }

    fn Switch(ValueRef V, BasicBlockRef Else, uint NumCases) -> ValueRef {
        ret llvm.LLVMBuildSwitch(B, V, Else, NumCases);
    }

    fn IndirectBr(ValueRef Addr, uint NumDests) -> ValueRef {
        ret llvm.LLVMBuildIndirectBr(B, Addr, NumDests);
    }

    fn Invoke(ValueRef Fn, vbuf Args, uint NumArgs,
              BasicBlockRef Then, BasicBlockRef Catch,
              sbuf Name) -> ValueRef {
        ret llvm.LLVMBuildInvoke(B, Fn, Args, NumArgs,
                                 Then, Catch, Name);
    }

    fn Unwind() -> ValueRef {
        ret llvm.LLVMBuildUnwind(B);
    }

    fn Unreachable() -> ValueRef {
        ret llvm.LLVMBuildUnreachable(B);
    }

    /* Arithmetic */
    fn Add(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildAdd(B, LHS, RHS, 0 as sbuf);
    }

    fn NSWAdd(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildNSWAdd(B, LHS, RHS, 0 as sbuf);
    }

    fn NUWAdd(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildNUWAdd(B, LHS, RHS, 0 as sbuf);
    }

    fn FAdd(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildFAdd(B, LHS, RHS, 0 as sbuf);
    }

    fn Sub(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildSub(B, LHS, RHS, 0 as sbuf);
    }

    fn NSWSub(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildNSWSub(B, LHS, RHS, 0 as sbuf);
    }

    fn NUWSub(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildNUWSub(B, LHS, RHS, 0 as sbuf);
    }

    fn FSub(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildFSub(B, LHS, RHS, 0 as sbuf);
    }

    fn Mul(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildMul(B, LHS, RHS, 0 as sbuf);
    }

    fn NSWMul(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildNSWMul(B, LHS, RHS, 0 as sbuf);
    }

    fn NUWMul(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildNUWMul(B, LHS, RHS, 0 as sbuf);
    }

    fn FMul(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildFMul(B, LHS, RHS, 0 as sbuf);
    }

    fn UDiv(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildUDiv(B, LHS, RHS, 0 as sbuf);
    }

    fn SDiv(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildSDiv(B, LHS, RHS, 0 as sbuf);
    }

    fn ExactSDiv(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildExactSDiv(B, LHS, RHS, 0 as sbuf);
    }

    fn FDiv(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildFDiv(B, LHS, RHS, 0 as sbuf);
    }

    fn URem(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildURem(B, LHS, RHS, 0 as sbuf);
    }

    fn SRem(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildSRem(B, LHS, RHS, 0 as sbuf);
    }

    fn FRem(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildFRem(B, LHS, RHS, 0 as sbuf);
    }

    fn Shl(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildShl(B, LHS, RHS, 0 as sbuf);
    }

    fn LShr(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildLShr(B, LHS, RHS, 0 as sbuf);
    }

    fn AShr(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildAShr(B, LHS, RHS, 0 as sbuf);
    }

    fn And(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildAnd(B, LHS, RHS, 0 as sbuf);
    }

    fn Or(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildOr(B, LHS, RHS, 0 as sbuf);
    }

    fn Xor(ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildXor(B, LHS, RHS, 0 as sbuf);
    }

    fn BinOp(Opcode Op, ValueRef LHS, ValueRef RHS) -> ValueRef {
        ret llvm.LLVMBuildBinOp(B, Op, LHS, RHS, 0 as sbuf);
    }

    fn Neg(ValueRef V) -> ValueRef {
        ret llvm.LLVMBuildNeg(B, V, 0 as sbuf);
    }

    fn NSWNeg(ValueRef V) -> ValueRef {
        ret llvm.LLVMBuildNSWNeg(B, V, 0 as sbuf);
    }

    fn NUWNeg(ValueRef V) -> ValueRef {
        ret llvm.LLVMBuildNUWNeg(B, V, 0 as sbuf);
    }
    fn FNeg(ValueRef V) -> ValueRef {
        ret llvm.LLVMBuildFNeg(B, V, 0 as sbuf);
    }
    fn Not(ValueRef V) -> ValueRef {
        ret llvm.LLVMBuildNot(B, V, 0 as sbuf);
    }

    /* Memory */
    fn Malloc(TypeRef Ty) -> ValueRef {
        ret llvm.LLVMBuildMalloc(B, Ty, 0 as sbuf);
    }

    fn ArrayMalloc(TypeRef Ty, ValueRef Val) -> ValueRef {
        ret llvm.LLVMBuildArrayMalloc(B, Ty, Val, 0 as sbuf);
    }

    fn Alloca(TypeRef Ty) -> ValueRef {
        ret llvm.LLVMBuildAlloca(B, Ty, 0 as sbuf);
    }

    fn ArrayAlloca(TypeRef Ty, ValueRef Val) -> ValueRef {
        ret llvm.LLVMBuildArrayAlloca(B, Ty, Val, 0 as sbuf);
    }

    fn Free(ValueRef PointerVal) -> ValueRef {
        ret llvm.LLVMBuildFree(B, PointerVal);
    }

    fn Load(ValueRef PointerVal) -> ValueRef {
        ret llvm.LLVMBuildLoad(B, PointerVal, 0 as sbuf);
    }

    fn Store(ValueRef Val, ValueRef Ptr) -> ValueRef {
        ret llvm.LLVMBuildStore(B, Val, Ptr);
    }

    fn GEP(ValueRef Pointer, vbuf Indices, uint NumIndices,
           sbuf Name) -> ValueRef {
        ret llvm.LLVMBuildGEP(B, Pointer, Indices, NumIndices, 0 as sbuf);
    }

    fn InBoundsGEP(ValueRef Pointer, vbuf Indices, uint NumIndices,
                   sbuf Name) -> ValueRef {
        ret llvm.LLVMBuildInBoundsGEP(B, Pointer, Indices,
                                      NumIndices, 0 as sbuf);
    }

    fn StructGEP(ValueRef Pointer, uint Idx) -> ValueRef {
        ret llvm.LLVMBuildStructGEP(B, Pointer, Idx, 0 as sbuf);
    }

    fn GlobalString(sbuf Str) -> ValueRef {
        ret llvm.LLVMBuildGlobalString(B, Str, 0 as sbuf);
    }

    fn GlobalStringPtr(sbuf Str) -> ValueRef {
        ret llvm.LLVMBuildGlobalStringPtr(B, Str, 0 as sbuf);
    }

    drop {
        llvm.LLVMDisposeBuilder(B);
    }
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
