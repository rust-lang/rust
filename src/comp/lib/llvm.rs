import std._str.rustrt.sbuf;
import std._vec.rustrt.vbuf;

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

  fn ModuleCreateWithName(sbuf ModuleID) -> ModuleRef;
  fn DisposeModule(ModuleRef M);

  fn GetDataLayout(ModuleRef M) -> sbuf;
  fn SetDataLayout(ModuleRef M, sbuf Triple);

  fn GetTarget(ModuleRef M) -> sbuf;
  fn SetTarget(ModuleRef M, sbuf Triple);

  fn AddTypeName(ModuleRef M, sbuf Name, TypeRef Ty) -> Bool;
  fn DeleteTypeName(ModuleRef M, sbuf Name);
  fn GetTypeByName(ModuleRef M, sbuf Name) -> TypeRef;

  fn DumpModule(ModuleRef M);

  fn GetTypeContext(TypeRef Ty) -> ContextRef;

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

  fn FunctionType(TypeRef ReturnType, vbuf ParamTypes,
                  uint ParamCount, Bool IsVarArg) -> TypeRef;
  fn IsFunctionVarArg(TypeRef FunctionTy) -> Bool;
  fn GetReturnType(TypeRef FunctionTy) -> TypeRef;
  fn CountParamTypes(TypeRef FunctionTy) -> uint;
  fn GetParamTypes(TypeRef FunctionTy, vbuf Dest);

  fn StructTypeInContext(ContextRef C, vbuf ElementTypes,
                         uint ElementCount, Bool Packed) -> TypeRef;
  fn StructType(vbuf ElementTypes, uint ElementCount,
                Bool Packed) -> TypeRef;
  fn CountStructElementTypes(TypeRef StructTy) -> uint;
  fn GetStructElementTypes(TypeRef StructTy, vbuf Dest);
  fn IsPackedStruct(TypeRef StructTy) -> Bool;

  fn UnionTypeInContext(ContextRef C, vbuf ElementTypes,
                        uint ElementCount) -> TypeRef;
  fn UnionType(vbuf ElementTypes, uint ElementCount) -> TypeRef;
  fn CountUnionElementTypes(TypeRef UnionTy) -> uint;
  fn GetUnionElementTypes(TypeRef UnionTy, vbuf Dest);

  fn ArrayType(TypeRef ElementType, uint ElementCount) -> TypeRef;
  fn PointerType(TypeRef ElementType, uint AddressSpace) -> TypeRef;
  fn VectorType(TypeRef ElementType, uint ElementCount) -> TypeRef;

  fn GetElementType(TypeRef Ty) -> TypeRef;
  fn GetArrayLength(TypeRef ArrayTy) -> uint;
  fn GetPointerAddressSpace(TypeRef PointerTy) -> uint;
  fn GetVectorSize(TypeRef VectorTy) -> uint;

  fn VoidTypeInContext(ContextRef C) -> TypeRef;
  fn LabelTypeInContext(ContextRef C) -> TypeRef;
  fn OpaqueTypeInContext(ContextRef C) -> TypeRef;

  fn VoidType() -> TypeRef;
  fn LabelType() -> TypeRef;
  fn OpaqueType() -> TypeRef;

  fn CreateTypeHandle(TypeRef PotentiallyAbstractTy) -> TypeHandleRef;
  fn RefineType(TypeRef AbstractTy, TypeRef ConcreteTy);
  fn ResolveTypeHandle(TypeHandleRef TypeHandle) -> TypeRef;
  fn DisposeTypeHandle(TypeHandleRef TypeHandle);

  fn TypeOf(ValueRef Val) -> TypeRef;
  fn GetValueName(ValueRef Val) -> sbuf;
  fn SetValueName(ValueRef Val, sbuf Name);
  fn DumpValue(ValueRef Val);
  fn ReplaceAllUsesWith(ValueRef OldVal, ValueRef NewVal);
  fn HasMetadata(ValueRef Val) -> int;
  fn GetMetadata(ValueRef Val, uint KindID) -> ValueRef;
  fn SetMetadata(ValueRef Val, uint KindID, ValueRef Node);


}