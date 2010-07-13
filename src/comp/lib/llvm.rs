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


  /* Create and destroy contexts. */
  fn ContextCreate() -> ContextRef;
  fn GetGlobalContext() -> ContextRef;
  fn ContextDispose(ContextRef C);
  fn GetMDKindIDInContext(ContextRef C, sbuf Name, uint SLen) -> uint;
  fn GetMDKindID(sbuf Name, uint SLen) -> uint;

  /* Create and destroy modules. */
  fn ModuleCreateWithName(sbuf ModuleID) -> ModuleRef;
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

}