import std._str.rustrt.sbuf;
import std._vec.rustrt.vbuf;

type ULongLong = u64;
type LongLong = i64;
type Long = i32;
type Bool = int;

native mod llvm = "libLLVM-2.7.so" {

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
  fn ContextCreate() -> ContextRef;
  fn GetGlobalContext() -> ContextRef;
  fn ContextDispose(ContextRef C);
  fn GetMDKindIDInContext(ContextRef C, sbuf Name, uint SLen) -> uint;
  fn GetMDKindID(sbuf Name, uint SLen) -> uint;

  /* Create and destroy modules. */
  fn ModuleCreateWithNameInContext(sbuf ModuleID, ContextRef C) -> ModuleRef;
  fn DisposeModule(ModuleRef M);

  /** Data layout. See Module::getDataLayout. */
  fn GetDataLayout(ModuleRef M) -> sbuf;
  fn SetDataLayout(ModuleRef M, sbuf Triple);

  /** Target triple. See Module::getTargetTriple. */
  fn GetTarget(ModuleRef M) -> sbuf;
  fn SetTarget(ModuleRef M, sbuf Triple);

  /** See Module::addTypeName. */
  fn AddTypeName(ModuleRef M, sbuf Name, TypeRef Ty) -> Bool;
  fn DeleteTypeName(ModuleRef M, sbuf Name);
  fn GetTypeByName(ModuleRef M, sbuf Name) -> TypeRef;

  /** See Module::dump. */
  fn DumpModule(ModuleRef M);

  /** See Module::setModuleInlineAsm. */
  fn SetModuleInlineAsm(ModuleRef M, sbuf Asm);

  /** See llvm::LLVMType::getContext. */
  fn GetTypeContext(TypeRef Ty) -> ContextRef;

  /* Operations on integer types */
  fn Int1TypeInContext(ContextRef C) -> TypeRef;
  fn Int8TypeInContext(ContextRef C) -> TypeRef;
  fn Int16TypeInContext(ContextRef C) -> TypeRef;
  fn Int32TypeInContext(ContextRef C) -> TypeRef;
  fn Int64TypeInContext(ContextRef C) -> TypeRef;
  fn IntTypeInContext(ContextRef C, uint NumBits) -> TypeRef;

  fn Int1Type() -> TypeRef;
  fn Int8Type() -> TypeRef;
  fn Int16Type() -> TypeRef;
  fn Int32Type() -> TypeRef;
  fn Int64Type() -> TypeRef;
  fn IntType(uint NumBits) -> TypeRef;
  fn GetIntTypeWidth(TypeRef IntegerTy) -> uint;

  /* Operations on real types */
  fn FloatTypeInContext(ContextRef C) -> TypeRef;
  fn DoubleTypeInContext(ContextRef C) -> TypeRef;
  fn X86FP80TypeInContext(ContextRef C) -> TypeRef;
  fn FP128TypeInContext(ContextRef C) -> TypeRef;
  fn PPCFP128TypeInContext(ContextRef C) -> TypeRef;

  fn FloatType() -> TypeRef;
  fn DoubleType() -> TypeRef;
  fn X86FP80Type() -> TypeRef;
  fn FP128Type() -> TypeRef;
  fn PPCFP128Type() -> TypeRef;

  /* Operations on function types */
  fn FunctionType(TypeRef ReturnType, vbuf ParamTypes,
                  uint ParamCount, Bool IsVarArg) -> TypeRef;
  fn IsFunctionVarArg(TypeRef FunctionTy) -> Bool;
  fn GetReturnType(TypeRef FunctionTy) -> TypeRef;
  fn CountParamTypes(TypeRef FunctionTy) -> uint;
  fn GetParamTypes(TypeRef FunctionTy, vbuf Dest);

  /* Operations on struct types */
  fn StructTypeInContext(ContextRef C, vbuf ElementTypes,
                         uint ElementCount, Bool Packed) -> TypeRef;
  fn StructType(vbuf ElementTypes, uint ElementCount,
                Bool Packed) -> TypeRef;
  fn CountStructElementTypes(TypeRef StructTy) -> uint;
  fn GetStructElementTypes(TypeRef StructTy, vbuf Dest);
  fn IsPackedStruct(TypeRef StructTy) -> Bool;

  /* Operations on union types */
  fn UnionTypeInContext(ContextRef C, vbuf ElementTypes,
                        uint ElementCount) -> TypeRef;
  fn UnionType(vbuf ElementTypes, uint ElementCount) -> TypeRef;
  fn CountUnionElementTypes(TypeRef UnionTy) -> uint;
  fn GetUnionElementTypes(TypeRef UnionTy, vbuf Dest);

  /* Operations on array, pointer, and vector types (sequence types) */
  fn ArrayType(TypeRef ElementType, uint ElementCount) -> TypeRef;
  fn PointerType(TypeRef ElementType, uint AddressSpace) -> TypeRef;
  fn VectorType(TypeRef ElementType, uint ElementCount) -> TypeRef;

  fn GetElementType(TypeRef Ty) -> TypeRef;
  fn GetArrayLength(TypeRef ArrayTy) -> uint;
  fn GetPointerAddressSpace(TypeRef PointerTy) -> uint;
  fn GetVectorSize(TypeRef VectorTy) -> uint;

  /* Operations on other types */
  fn VoidTypeInContext(ContextRef C) -> TypeRef;
  fn LabelTypeInContext(ContextRef C) -> TypeRef;
  fn OpaqueTypeInContext(ContextRef C) -> TypeRef;

  fn VoidType() -> TypeRef;
  fn LabelType() -> TypeRef;
  fn OpaqueType() -> TypeRef;

  /* Operations on type handles */
  fn CreateTypeHandle(TypeRef PotentiallyAbstractTy) -> TypeHandleRef;
  fn RefineType(TypeRef AbstractTy, TypeRef ConcreteTy);
  fn ResolveTypeHandle(TypeHandleRef TypeHandle) -> TypeRef;
  fn DisposeTypeHandle(TypeHandleRef TypeHandle);

  /* Operations on all values */
  fn TypeOf(ValueRef Val) -> TypeRef;
  fn GetValueName(ValueRef Val) -> sbuf;
  fn SetValueName(ValueRef Val, sbuf Name);
  fn DumpValue(ValueRef Val);
  fn ReplaceAllUsesWith(ValueRef OldVal, ValueRef NewVal);
  fn HasMetadata(ValueRef Val) -> int;
  fn GetMetadata(ValueRef Val, uint KindID) -> ValueRef;
  fn SetMetadata(ValueRef Val, uint KindID, ValueRef Node);

  /* Operations on Uses */
  fn GetFirstUse(ValueRef Val) -> UseRef;
  fn GetNextUse(UseRef U) -> UseRef;
  fn GetUser(UseRef U) -> ValueRef;
  fn GetUsedValue(UseRef U) -> ValueRef;

  /* Operations on Users */
  fn GetOperand(ValueRef Val, uint Index) -> ValueRef;

  /* Operations on constants of any type */
  fn ConstNull(TypeRef Ty) -> ValueRef; /* all zeroes */
  fn ConstAllOnes(TypeRef Ty) -> ValueRef; /* only for int/vector */
  fn GetUndef(TypeRef Ty) -> ValueRef;
  fn IsConstant(ValueRef Val) -> Bool;
  fn IsNull(ValueRef Val) -> Bool;
  fn IsUndef(ValueRef Val) -> Bool;
  fn ConstPointerNull(TypeRef Ty) -> ValueRef;

  /* Operations on metadata */
  fn MDStringInContext(ContextRef C, sbuf Str, uint SLen) -> ValueRef;
  fn MDString(sbuf Str, uint SLen) -> ValueRef;
  fn MDNodeInContext(ContextRef C, vbuf Vals, uint Count) -> ValueRef;
  fn MDNode(vbuf Vals, uint Count) -> ValueRef;

  /* Operations on scalar constants */
  fn ConstInt(TypeRef IntTy, ULongLong N, Bool SignExtend) -> ValueRef;
  fn ConstIntOfString(TypeRef IntTy, sbuf Text, u8 Radix) -> ValueRef;
  fn ConstIntOfStringAndSize(TypeRef IntTy, sbuf Text,
                             uint SLen, u8 Radix) -> ValueRef;
  fn ConstReal(TypeRef RealTy, f64 N) -> ValueRef;
  fn ConstRealOfString(TypeRef RealTy, sbuf Text) -> ValueRef;
  fn ConstRealOfStringAndSize(TypeRef RealTy, sbuf Text,
                              uint SLen) -> ValueRef;
  fn ConstIntGetZExtValue(ValueRef ConstantVal) -> ULongLong;
  fn ConstIntGetSExtValue(ValueRef ConstantVal) -> LongLong;


  /* Operations on composite constants */
  fn ConstStringInContext(ContextRef C, sbuf Str,
                          uint Length, Bool DontNullTerminate) -> ValueRef;
  fn ConstStructInContext(ContextRef C, vbuf ConstantVals,
                          uint Count, Bool Packed) -> ValueRef;

  fn ConstString(sbuf Str, uint Length, Bool DontNullTerminate) -> ValueRef;
  fn ConstArray(TypeRef ElementTy,
                vbuf ConstantVals, uint Length) -> ValueRef;
  fn ConstStruct(vbuf ConstantVals, uint Count, Bool Packed) -> ValueRef;
  fn ConstVector(vbuf ScalarConstantVals, uint Size) -> ValueRef;
  fn ConstUnion(TypeRef Ty, ValueRef Val) -> ValueRef;

  /* Constant expressions */
  fn AlignOf(TypeRef Ty) -> ValueRef;
  fn SizeOf(TypeRef Ty) -> ValueRef;
  fn ConstNeg(ValueRef ConstantVal) -> ValueRef;
  fn ConstNSWNeg(ValueRef ConstantVal) -> ValueRef;
  fn ConstNUWNeg(ValueRef ConstantVal) -> ValueRef;
  fn ConstFNeg(ValueRef ConstantVal) -> ValueRef;
  fn ConstNot(ValueRef ConstantVal) -> ValueRef;
  fn ConstAdd(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstNSWAdd(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstNUWAdd(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstFAdd(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstSub(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstNSWSub(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstNUWSub(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstFSub(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstMul(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstNSWMul(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstNUWMul(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstFMul(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstUDiv(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstSDiv(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstExactSDiv(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstFDiv(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstURem(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstSRem(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstFRem(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstAnd(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstOr(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstXor(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstShl(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstLShr(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstAShr(ValueRef LHSConstant, ValueRef RHSConstant) -> ValueRef;
  fn ConstGEP(ValueRef ConstantVal,
              vbuf ConstantIndices, uint NumIndices) -> ValueRef;
  fn ConstInBoundsGEP(ValueRef ConstantVal,
                      vbuf ConstantIndices,
                      uint NumIndices) -> ValueRef;
  fn ConstTrunc(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
  fn ConstSExt(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
  fn ConstZExt(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
  fn ConstFPTrunc(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
  fn ConstFPExt(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
  fn ConstUIToFP(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
  fn ConstSIToFP(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
  fn ConstFPToUI(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
  fn ConstFPToSI(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
  fn ConstPtrToInt(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
  fn ConstIntToPtr(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
  fn ConstBitCast(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
  fn ConstZExtOrBitCast(ValueRef ConstantVal,
                        TypeRef ToType) -> ValueRef;
  fn ConstSExtOrBitCast(ValueRef ConstantVal,
                        TypeRef ToType) -> ValueRef;
  fn ConstTruncOrBitCast(ValueRef ConstantVal,
                         TypeRef ToType) -> ValueRef;
  fn ConstPointerCast(ValueRef ConstantVal,
                      TypeRef ToType) -> ValueRef;
  fn ConstIntCast(ValueRef ConstantVal, TypeRef ToType,
                  Bool isSigned) -> ValueRef;
  fn ConstFPCast(ValueRef ConstantVal, TypeRef ToType) -> ValueRef;
  fn ConstSelect(ValueRef ConstantCondition,
                 ValueRef ConstantIfTrue,
                 ValueRef ConstantIfFalse) -> ValueRef;
  fn ConstExtractElement(ValueRef VectorConstant,
                         ValueRef IndexConstant) -> ValueRef;
  fn ConstInsertElement(ValueRef VectorConstant,
                        ValueRef ElementValueConstant,
                        ValueRef IndexConstant) -> ValueRef;
  fn ConstShuffleVector(ValueRef VectorAConstant,
                        ValueRef VectorBConstant,
                        ValueRef MaskConstant) -> ValueRef;
  fn ConstExtractValue(ValueRef AggConstant, vbuf IdxList,
                       uint NumIdx) -> ValueRef;
  fn ConstInsertValue(ValueRef AggConstant,
                      ValueRef ElementValueConstant,
                      vbuf IdxList, uint NumIdx) -> ValueRef;
  fn ConstInlineAsm(TypeRef Ty,
                    sbuf AsmString, sbuf Constraints,
                    Bool HasSideEffects, Bool IsAlignStack) -> ValueRef;
  fn BlockAddress(ValueRef F, BasicBlockRef BB) -> ValueRef;



  /* Operations on global variables, functions, and aliases (globals) */
  fn GetGlobalParent(ValueRef Global) -> ModuleRef;
  fn IsDeclaration(ValueRef Global) -> Bool;
  fn GetLinkage(ValueRef Global) -> Linkage;
  fn SetLinkage(ValueRef Global, Linkage Link);
  fn GetSection(ValueRef Global) -> sbuf;
  fn SetSection(ValueRef Global, sbuf Section);
  fn GetVisibility(ValueRef Global) -> Visibility;
  fn SetVisibility(ValueRef Global, Visibility Viz);
  fn GetAlignment(ValueRef Global) -> uint;
  fn SetAlignment(ValueRef Global, uint Bytes);


  /* Operations on global variables */
  fn AddGlobal(ModuleRef M, TypeRef Ty, sbuf Name) -> ValueRef;
  fn AddGlobalInAddressSpace(ModuleRef M, TypeRef Ty,
                             sbuf Name,
                             uint AddressSpace) -> ValueRef;
  fn GetNamedGlobal(ModuleRef M, sbuf Name) -> ValueRef;
  fn GetFirstGlobal(ModuleRef M) -> ValueRef;
  fn GetLastGlobal(ModuleRef M) -> ValueRef;
  fn GetNextGlobal(ValueRef GlobalVar) -> ValueRef;
  fn GetPreviousGlobal(ValueRef GlobalVar) -> ValueRef;
  fn DeleteGlobal(ValueRef GlobalVar);
  fn GetInitializer(ValueRef GlobalVar) -> ValueRef;
  fn SetInitializer(ValueRef GlobalVar, ValueRef ConstantVal);
  fn IsThreadLocal(ValueRef GlobalVar) -> Bool;
  fn SetThreadLocal(ValueRef GlobalVar, Bool IsThreadLocal);
  fn IsGlobalConstant(ValueRef GlobalVar) -> Bool;
  fn SetGlobalConstant(ValueRef GlobalVar, Bool IsConstant);

  /* Operations on aliases */
  fn AddAlias(ModuleRef M, TypeRef Ty, ValueRef Aliasee,
              sbuf Name) -> ValueRef;

  /* Operations on functions */
  fn AddFunction(ModuleRef M, sbuf Name,
                 TypeRef FunctionTy) -> ValueRef;
  fn GetNamedFunction(ModuleRef M, sbuf Name) -> ValueRef;
  fn GetFirstFunction(ModuleRef M) -> ValueRef;
  fn GetLastFunction(ModuleRef M) -> ValueRef;
  fn GetNextFunction(ValueRef Fn) -> ValueRef;
  fn GetPreviousFunction(ValueRef Fn) -> ValueRef;
  fn DeleteFunction(ValueRef Fn);
  fn GetIntrinsicID(ValueRef Fn) -> uint;
  fn GetFunctionCallConv(ValueRef Fn) -> uint;
  fn SetFunctionCallConv(ValueRef Fn, uint CC);
  fn GetGC(ValueRef Fn) -> sbuf;
  fn SetGC(ValueRef Fn, sbuf Name);
  fn AddFunctionAttr(ValueRef Fn, Attribute PA);
  fn GetFunctionAttr(ValueRef Fn) -> Attribute;
  fn RemoveFunctionAttr(ValueRef Fn, Attribute PA);

  /* Operations on parameters */
  fn CountParams(ValueRef Fn) -> uint;
  fn GetParams(ValueRef Fn, vbuf Params);
  fn GetParam(ValueRef Fn, uint Index) -> ValueRef;
  fn GetParamParent(ValueRef Inst) -> ValueRef;
  fn GetFirstParam(ValueRef Fn) -> ValueRef;
  fn GetLastParam(ValueRef Fn) -> ValueRef;
  fn GetNextParam(ValueRef Arg) -> ValueRef;
  fn GetPreviousParam(ValueRef Arg) -> ValueRef;
  fn AddAttribute(ValueRef Arg, Attribute PA);
  fn RemoveAttribute(ValueRef Arg, Attribute PA);
  fn GetAttribute(ValueRef Arg) -> Attribute;
  fn SetParamAlignment(ValueRef Arg, uint align);

  /* Operations on basic blocks */
  fn BasicBlockAsValue(BasicBlockRef BB) -> ValueRef;
  fn ValueIsBasicBlock(ValueRef Val) -> Bool;
  fn ValueAsBasicBlock(ValueRef Val) -> BasicBlockRef;
  fn GetBasicBlockParent(BasicBlockRef BB) -> ValueRef;
  fn CountBasicBlocks(ValueRef Fn) -> uint;
  fn GetBasicBlocks(ValueRef Fn, vbuf BasicBlocks);
  fn GetFirstBasicBlock(ValueRef Fn) -> BasicBlockRef;
  fn GetLastBasicBlock(ValueRef Fn) -> BasicBlockRef;
  fn GetNextBasicBlock(BasicBlockRef BB) -> BasicBlockRef;
  fn GetPreviousBasicBlock(BasicBlockRef BB) -> BasicBlockRef;
  fn GetEntryBasicBlock(ValueRef Fn) -> BasicBlockRef;

  fn AppendBasicBlockInContext(ContextRef C, ValueRef Fn,
                               sbuf Name) -> BasicBlockRef;
  fn InsertBasicBlockInContext(ContextRef C, BasicBlockRef BB,
                               sbuf Name) -> BasicBlockRef;

  fn AppendBasicBlock(ValueRef Fn, sbuf Name) -> BasicBlockRef;
  fn InsertBasicBlock(BasicBlockRef InsertBeforeBB,
                      sbuf Name) -> BasicBlockRef;
  fn DeleteBasicBlock(BasicBlockRef BB);

  /* Operations on instructions */
  fn GetInstructionParent(ValueRef Inst) -> BasicBlockRef;
  fn GetFirstInstruction(BasicBlockRef BB) -> ValueRef;
  fn GetLastInstruction(BasicBlockRef BB) -> ValueRef;
  fn GetNextInstruction(ValueRef Inst) -> ValueRef;
  fn GetPreviousInstruction(ValueRef Inst) -> ValueRef;

  /* Operations on call sites */
  fn SetInstructionCallConv(ValueRef Instr, uint CC);
  fn GetInstructionCallConv(ValueRef Instr) -> uint;
  fn AddInstrAttribute(ValueRef Instr, uint index, Attribute IA);
  fn RemoveInstrAttribute(ValueRef Instr, uint index, Attribute IA);
  fn SetInstrParamAlignment(ValueRef Instr, uint index, uint align);

  /* Operations on call instructions (only) */
  fn IsTailCall(ValueRef CallInst) -> Bool;
  fn SetTailCall(ValueRef CallInst, Bool IsTailCall);

  /* Operations on phi nodes */
  fn AddIncoming(ValueRef PhiNode, vbuf IncomingValues,
                 vbuf IncomingBlocks, uint Count);
  fn CountIncoming(ValueRef PhiNode) -> uint;
  fn GetIncomingValue(ValueRef PhiNode, uint Index) -> ValueRef;
  fn GetIncomingBlock(ValueRef PhiNode, uint Index) -> BasicBlockRef;

  /* Instruction builders */
  fn CreateBuilderInContext(ContextRef C) -> BuilderRef;
  fn CreateBuilder() -> BuilderRef;
  fn PositionBuilder(BuilderRef Builder, BasicBlockRef Block,
                     ValueRef Instr);
  fn PositionBuilderBefore(BuilderRef Builder, ValueRef Instr);
  fn PositionBuilderAtEnd(BuilderRef Builder, BasicBlockRef Block);
  fn GetInsertBlock(BuilderRef Builder) -> BasicBlockRef;
  fn ClearInsertionPosition(BuilderRef Builder);
  fn InsertIntoBuilder(BuilderRef Builder, ValueRef Instr);
  fn InsertIntoBuilderWithName(BuilderRef Builder, ValueRef Instr,
                               sbuf Name);
  fn DisposeBuilder(BuilderRef Builder);

  /* Metadata */
  fn SetCurrentDebugLocation(BuilderRef Builder, ValueRef L);
  fn GetCurrentDebugLocation(BuilderRef Builder) -> ValueRef;
  fn SetInstDebugLocation(BuilderRef Builder, ValueRef Inst);

  /* Terminators */
  fn BuildRetVoid(BuilderRef B) -> ValueRef;
  fn BuildRet(BuilderRef B, ValueRef V) -> ValueRef;
  fn BuildAggregateRet(BuilderRef B, vbuf RetVals,
                       uint N) -> ValueRef;
  fn BuildBr(BuilderRef B, BasicBlockRef Dest) -> ValueRef;
  fn BuildCondBr(BuilderRef B, ValueRef If,
                 BasicBlockRef Then, BasicBlockRef Else) -> ValueRef;
  fn BuildSwitch(BuilderRef B, ValueRef V,
                 BasicBlockRef Else, uint NumCases) -> ValueRef;
  fn BuildIndirectBr(BuilderRef B, ValueRef Addr,
                     uint NumDests) -> ValueRef;
  fn BuildInvoke(BuilderRef B, ValueRef Fn,
                 vbuf Args, uint NumArgs,
                 BasicBlockRef Then, BasicBlockRef Catch,
                 sbuf Name) -> ValueRef;
  fn BuildUnwind(BuilderRef B) -> ValueRef;
  fn BuildUnreachable(BuilderRef B) -> ValueRef;

  /* Add a case to the switch instruction */
  fn AddCase(ValueRef Switch, ValueRef OnVal,
             BasicBlockRef Dest);

  /* Add a destination to the indirectbr instruction */
  fn AddDestination(ValueRef IndirectBr, BasicBlockRef Dest);

  /* Arithmetic */
  fn BuildAdd(BuilderRef B, ValueRef LHS, ValueRef RHS,
              sbuf Name) -> ValueRef;
  fn BuildNSWAdd(BuilderRef B, ValueRef LHS, ValueRef RHS,
                 sbuf Name) -> ValueRef;
  fn BuildNUWAdd(BuilderRef B, ValueRef LHS, ValueRef RHS,
                 sbuf Name) -> ValueRef;
  fn BuildFAdd(BuilderRef B, ValueRef LHS, ValueRef RHS,
               sbuf Name) -> ValueRef;
  fn BuildSub(BuilderRef B, ValueRef LHS, ValueRef RHS,
              sbuf Name) -> ValueRef;
  fn BuildNSWSub(BuilderRef B, ValueRef LHS, ValueRef RHS,
                 sbuf Name) -> ValueRef;
  fn BuildNUWSub(BuilderRef B, ValueRef LHS, ValueRef RHS,
                 sbuf Name) -> ValueRef;
  fn BuildFSub(BuilderRef B, ValueRef LHS, ValueRef RHS,
               sbuf Name) -> ValueRef;
  fn BuildMul(BuilderRef B, ValueRef LHS, ValueRef RHS,
              sbuf Name) -> ValueRef;
  fn BuildNSWMul(BuilderRef B, ValueRef LHS, ValueRef RHS,
                 sbuf Name) -> ValueRef;
  fn BuildNUWMul(BuilderRef B, ValueRef LHS, ValueRef RHS,
                 sbuf Name) -> ValueRef;
  fn BuildFMul(BuilderRef B, ValueRef LHS, ValueRef RHS,
               sbuf Name) -> ValueRef;
  fn BuildUDiv(BuilderRef B, ValueRef LHS, ValueRef RHS,
               sbuf Name) -> ValueRef;
  fn BuildSDiv(BuilderRef B, ValueRef LHS, ValueRef RHS,
               sbuf Name) -> ValueRef;
  fn BuildExactSDiv(BuilderRef B, ValueRef LHS, ValueRef RHS,
                    sbuf Name) -> ValueRef;
  fn BuildFDiv(BuilderRef B, ValueRef LHS, ValueRef RHS,
               sbuf Name) -> ValueRef;
  fn BuildURem(BuilderRef B, ValueRef LHS, ValueRef RHS,
               sbuf Name) -> ValueRef;
  fn BuildSRem(BuilderRef B, ValueRef LHS, ValueRef RHS,
               sbuf Name) -> ValueRef;
  fn BuildFRem(BuilderRef B, ValueRef LHS, ValueRef RHS,
               sbuf Name) -> ValueRef;
  fn BuildShl(BuilderRef B, ValueRef LHS, ValueRef RHS,
              sbuf Name) -> ValueRef;
  fn BuildLShr(BuilderRef B, ValueRef LHS, ValueRef RHS,
               sbuf Name) -> ValueRef;
  fn BuildAShr(BuilderRef B, ValueRef LHS, ValueRef RHS,
               sbuf Name) -> ValueRef;
  fn BuildAnd(BuilderRef B, ValueRef LHS, ValueRef RHS,
              sbuf Name) -> ValueRef;
  fn BuildOr(BuilderRef B, ValueRef LHS, ValueRef RHS,
             sbuf Name) -> ValueRef;
  fn BuildXor(BuilderRef B, ValueRef LHS, ValueRef RHS,
              sbuf Name) -> ValueRef;
  fn BuildBinOp(BuilderRef B, Opcode Op,
                ValueRef LHS, ValueRef RHS,
                sbuf Name) -> ValueRef;
  fn BuildNeg(BuilderRef B, ValueRef V, sbuf Name) -> ValueRef;
  fn BuildNSWNeg(BuilderRef B, ValueRef V,
                 sbuf Name) -> ValueRef;
  fn BuildNUWNeg(BuilderRef B, ValueRef V,
                 sbuf Name) -> ValueRef;
  fn BuildFNeg(BuilderRef B, ValueRef V, sbuf Name) -> ValueRef;
  fn BuildNot(BuilderRef B, ValueRef V, sbuf Name) -> ValueRef;

  /* Memory */
  fn BuildMalloc(BuilderRef B, TypeRef Ty, sbuf Name) -> ValueRef;
  fn BuildArrayMalloc(BuilderRef B, TypeRef Ty,
                      ValueRef Val, sbuf Name) -> ValueRef;
  fn BuildAlloca(BuilderRef B, TypeRef Ty, sbuf Name) -> ValueRef;
  fn BuildArrayAlloca(BuilderRef B, TypeRef Ty,
                      ValueRef Val, sbuf Name) -> ValueRef;
  fn BuildFree(BuilderRef B, ValueRef PointerVal) -> ValueRef;
  fn BuildLoad(BuilderRef B, ValueRef PointerVal,
               sbuf Name) -> ValueRef;
  fn BuildStore(BuilderRef B, ValueRef Val, ValueRef Ptr) -> ValueRef;
  fn BuildGEP(BuilderRef B, ValueRef Pointer,
              vbuf Indices, uint NumIndices,
              sbuf Name) -> ValueRef;
  fn BuildInBoundsGEP(BuilderRef B, ValueRef Pointer,
                      vbuf Indices, uint NumIndices,
                      sbuf Name) -> ValueRef;
  fn BuildStructGEP(BuilderRef B, ValueRef Pointer,
                    uint Idx, sbuf Name) -> ValueRef;
  fn BuildGlobalString(BuilderRef B, sbuf Str,
                       sbuf Name) -> ValueRef;
  fn BuildGlobalStringPtr(BuilderRef B, sbuf Str,
                          sbuf Name) -> ValueRef;

  /* Casts */
  fn BuildTrunc(BuilderRef B, ValueRef Val,
                TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildZExt(BuilderRef B, ValueRef Val,
               TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildSExt(BuilderRef B, ValueRef Val,
               TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildFPToUI(BuilderRef B, ValueRef Val,
                 TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildFPToSI(BuilderRef B, ValueRef Val,
                 TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildUIToFP(BuilderRef B, ValueRef Val,
                 TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildSIToFP(BuilderRef B, ValueRef Val,
                 TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildFPTrunc(BuilderRef B, ValueRef Val,
                  TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildFPExt(BuilderRef B, ValueRef Val,
                TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildPtrToInt(BuilderRef B, ValueRef Val,
                   TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildIntToPtr(BuilderRef B, ValueRef Val,
                   TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildBitCast(BuilderRef B, ValueRef Val,
                  TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildZExtOrBitCast(BuilderRef B, ValueRef Val,
                        TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildSExtOrBitCast(BuilderRef B, ValueRef Val,
                        TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildTruncOrBitCast(BuilderRef B, ValueRef Val,
                         TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildCast(BuilderRef B, Opcode Op, ValueRef Val,
               TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildPointerCast(BuilderRef B, ValueRef Val,
                      TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildIntCast(BuilderRef B, ValueRef Val,
                  TypeRef DestTy, sbuf Name) -> ValueRef;
  fn BuildFPCast(BuilderRef B, ValueRef Val,
                 TypeRef DestTy, sbuf Name) -> ValueRef;

  /* Comparisons */
  fn BuildICmp(BuilderRef B, IntPredicate Op,
               ValueRef LHS, ValueRef RHS,
               sbuf Name) -> ValueRef;
  fn BuildFCmp(BuilderRef B, RealPredicate Op,
               ValueRef LHS, ValueRef RHS,
               sbuf Name) -> ValueRef;

  /* Miscellaneous instructions */
  fn BuildPhi(BuilderRef B, TypeRef Ty, sbuf Name) -> ValueRef;
  fn BuildCall(BuilderRef B, ValueRef Fn,
               vbuf Args, uint NumArgs,
               sbuf Name) -> ValueRef;
  fn BuildSelect(BuilderRef B, ValueRef If,
                 ValueRef Then, ValueRef Else,
                 sbuf Name) -> ValueRef;
  fn BuildVAArg(BuilderRef B, ValueRef List, TypeRef Ty,
                sbuf Name) -> ValueRef;
  fn BuildExtractElement(BuilderRef B, ValueRef VecVal,
                         ValueRef Index, sbuf Name) -> ValueRef;
  fn BuildInsertElement(BuilderRef B, ValueRef VecVal,
                        ValueRef EltVal, ValueRef Index,
                        sbuf Name) -> ValueRef;
  fn BuildShuffleVector(BuilderRef B, ValueRef V1,
                        ValueRef V2, ValueRef Mask,
                        sbuf Name) -> ValueRef;
  fn BuildExtractValue(BuilderRef B, ValueRef AggVal,
                       uint Index, sbuf Name) -> ValueRef;
  fn BuildInsertValue(BuilderRef B, ValueRef AggVal,
                      ValueRef EltVal, uint Index,
                      sbuf Name) -> ValueRef;

  fn BuildIsNull(BuilderRef B, ValueRef Val,
                 sbuf Name) -> ValueRef;
  fn BuildIsNotNull(BuilderRef B, ValueRef Val,
                    sbuf Name) -> ValueRef;
  fn BuildPtrDiff(BuilderRef B, ValueRef LHS,
                  ValueRef RHS, sbuf Name) -> ValueRef;



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
