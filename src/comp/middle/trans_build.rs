import std::{vec, str};
import str::rustrt::sbuf;
import lib::llvm::llvm;
import llvm::{ValueRef, TypeRef, BasicBlockRef, BuilderRef,
              Opcode, ModuleRef};
import trans_common::block_ctxt;

resource BuilderRef_res(B: llvm::BuilderRef) {
    llvm::LLVMDisposeBuilder(B);
}

fn mk_builder(llbb: BasicBlockRef) -> BuilderRef {
    let B = llvm::LLVMCreateBuilder();
    llvm::LLVMPositionBuilderAtEnd(B, llbb);
    ret B;
}

fn RetVoid(cx: &@block_ctxt) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    ret llvm::LLVMBuildRetVoid(*cx.build);
}

fn Ret(cx: &@block_ctxt, V: ValueRef) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    ret llvm::LLVMBuildRet(*cx.build, V);
}

fn AggregateRet(cx: &@block_ctxt, RetVals: &[ValueRef]) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    ret llvm::LLVMBuildAggregateRet(*cx.build, vec::to_ptr(RetVals),
                                    vec::len(RetVals));
}

fn Br(cx: &@block_ctxt, Dest: BasicBlockRef) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    ret llvm::LLVMBuildBr(*cx.build, Dest);
}

fn CondBr(cx: &@block_ctxt, If: ValueRef, Then: BasicBlockRef,
          Else: BasicBlockRef) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    ret llvm::LLVMBuildCondBr(*cx.build, If, Then, Else);
}

fn Switch(cx: &@block_ctxt, V: ValueRef, Else: BasicBlockRef,
          NumCases: uint) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    ret llvm::LLVMBuildSwitch(*cx.build, V, Else, NumCases);
}

fn IndirectBr(cx: &@block_ctxt, Addr: ValueRef,
              NumDests: uint) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    ret llvm::LLVMBuildIndirectBr(*cx.build, Addr, NumDests);
}

fn Invoke(cx: &@block_ctxt, Fn: ValueRef, Args: &[ValueRef],
          Then: BasicBlockRef, Catch: BasicBlockRef) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    ret llvm::LLVMBuildInvoke(*cx.build, Fn, vec::to_ptr(Args),
                              vec::len(Args), Then, Catch, str::buf(""));
}

fn Unreachable(cx: &@block_ctxt) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    ret llvm::LLVMBuildUnreachable(*cx.build);
}

/* Arithmetic */
fn Add(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildAdd(*cx.build, LHS, RHS, str::buf(""));
}

fn NSWAdd(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildNSWAdd(*cx.build, LHS, RHS, str::buf(""));
}

fn NUWAdd(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildNUWAdd(*cx.build, LHS, RHS, str::buf(""));
}

fn FAdd(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildFAdd(*cx.build, LHS, RHS, str::buf(""));
}

fn Sub(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildSub(*cx.build, LHS, RHS, str::buf(""));
}

fn NSWSub(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildNSWSub(*cx.build, LHS, RHS, str::buf(""));
}

fn NUWSub(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildNUWSub(*cx.build, LHS, RHS, str::buf(""));
}

fn FSub(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildFSub(*cx.build, LHS, RHS, str::buf(""));
}

fn Mul(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildMul(*cx.build, LHS, RHS, str::buf(""));
}

fn NSWMul(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildNSWMul(*cx.build, LHS, RHS, str::buf(""));
}

fn NUWMul(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildNUWMul(*cx.build, LHS, RHS, str::buf(""));
}

fn FMul(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildFMul(*cx.build, LHS, RHS, str::buf(""));
}

fn UDiv(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildUDiv(*cx.build, LHS, RHS, str::buf(""));
}

fn SDiv(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildSDiv(*cx.build, LHS, RHS, str::buf(""));
}

fn ExactSDiv(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildExactSDiv(*cx.build, LHS, RHS, str::buf(""));
}

fn FDiv(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildFDiv(*cx.build, LHS, RHS, str::buf(""));
}

fn URem(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildURem(*cx.build, LHS, RHS, str::buf(""));
}

fn SRem(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildSRem(*cx.build, LHS, RHS, str::buf(""));
}

fn FRem(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildFRem(*cx.build, LHS, RHS, str::buf(""));
}

fn Shl(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildShl(*cx.build, LHS, RHS, str::buf(""));
}

fn LShr(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildLShr(*cx.build, LHS, RHS, str::buf(""));
}

fn AShr(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildAShr(*cx.build, LHS, RHS, str::buf(""));
}

fn And(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildAnd(*cx.build, LHS, RHS, str::buf(""));
}

fn Or(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildOr(*cx.build, LHS, RHS, str::buf(""));
}

fn Xor(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildXor(*cx.build, LHS, RHS, str::buf(""));
}

fn BinOp(cx: &@block_ctxt, Op: Opcode, LHS: ValueRef,
         RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildBinOp(*cx.build, Op, LHS, RHS, str::buf(""));
}

fn Neg(cx: &@block_ctxt, V: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildNeg(*cx.build, V, str::buf(""));
}

fn NSWNeg(cx: &@block_ctxt, V: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildNSWNeg(*cx.build, V, str::buf(""));
}

fn NUWNeg(cx: &@block_ctxt, V: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildNUWNeg(*cx.build, V, str::buf(""));
}
fn FNeg(cx: &@block_ctxt, V: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildFNeg(*cx.build, V, str::buf(""));
}
fn Not(cx: &@block_ctxt, V: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildNot(*cx.build, V, str::buf(""));
}

/* Memory */
fn Malloc(cx: &@block_ctxt, Ty: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildMalloc(*cx.build, Ty, str::buf(""));
}

fn ArrayMalloc(cx: &@block_ctxt, Ty: TypeRef, Val: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildArrayMalloc(*cx.build, Ty, Val, str::buf(""));
}

fn Alloca(cx: &@block_ctxt, Ty: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildAlloca(*cx.build, Ty, str::buf(""));
}

fn ArrayAlloca(cx: &@block_ctxt, Ty: TypeRef, Val: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildArrayAlloca(*cx.build, Ty, Val, str::buf(""));
}

fn Free(cx: &@block_ctxt, PointerVal: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildFree(*cx.build, PointerVal);
}

fn Load(cx: &@block_ctxt, PointerVal: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildLoad(*cx.build, PointerVal, str::buf(""));
}

fn Store(cx: &@block_ctxt, Val: ValueRef, Ptr: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildStore(*cx.build, Val, Ptr);
}

fn GEP(cx: &@block_ctxt, Pointer: ValueRef,
       Indices: &[ValueRef]) -> ValueRef {
    ret llvm::LLVMBuildGEP(*cx.build, Pointer, vec::to_ptr(Indices),
                           vec::len(Indices), str::buf(""));
}

fn InBoundsGEP(cx: &@block_ctxt, Pointer: ValueRef,
               Indices: &[ValueRef]) -> ValueRef {
    ret llvm::LLVMBuildInBoundsGEP(*cx.build, Pointer, vec::to_ptr(Indices),
                                   vec::len(Indices), str::buf(""));
}

fn StructGEP(cx: &@block_ctxt, Pointer: ValueRef, Idx: uint) -> ValueRef {
    ret llvm::LLVMBuildStructGEP(*cx.build, Pointer, Idx, str::buf(""));
}

fn GlobalString(cx: &@block_ctxt, _Str: sbuf) -> ValueRef {
    ret llvm::LLVMBuildGlobalString(*cx.build, _Str, str::buf(""));
}

fn GlobalStringPtr(cx: &@block_ctxt, _Str: sbuf) -> ValueRef {
    ret llvm::LLVMBuildGlobalStringPtr(*cx.build, _Str, str::buf(""));
}

/* Casts */
fn Trunc(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildTrunc(*cx.build, Val, DestTy, str::buf(""));
}

fn ZExt(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildZExt(*cx.build, Val, DestTy, str::buf(""));
}

fn SExt(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildSExt(*cx.build, Val, DestTy, str::buf(""));
}

fn FPToUI(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildFPToUI(*cx.build, Val, DestTy, str::buf(""));
}

fn FPToSI(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildFPToSI(*cx.build, Val, DestTy, str::buf(""));
}

fn UIToFP(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildUIToFP(*cx.build, Val, DestTy, str::buf(""));
}

fn SIToFP(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildSIToFP(*cx.build, Val, DestTy, str::buf(""));
}

fn FPTrunc(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildFPTrunc(*cx.build, Val, DestTy, str::buf(""));
}

fn FPExt(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildFPExt(*cx.build, Val, DestTy, str::buf(""));
}

fn PtrToInt(cx: &@block_ctxt, Val: ValueRef,
            DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildPtrToInt(*cx.build, Val, DestTy, str::buf(""));
}

fn IntToPtr(cx: &@block_ctxt, Val: ValueRef,
            DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildIntToPtr(*cx.build, Val, DestTy, str::buf(""));
}

fn BitCast(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildBitCast(*cx.build, Val, DestTy, str::buf(""));
}

fn ZExtOrBitCast(cx: &@block_ctxt, Val: ValueRef,
                 DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildZExtOrBitCast(*cx.build, Val, DestTy, str::buf(""));
}

fn SExtOrBitCast(cx: &@block_ctxt, Val: ValueRef,
                 DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildSExtOrBitCast(*cx.build, Val, DestTy, str::buf(""));
}

fn TruncOrBitCast(cx: &@block_ctxt, Val: ValueRef,
                  DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildTruncOrBitCast(*cx.build, Val, DestTy, str::buf(""));
}

fn Cast(cx: &@block_ctxt, Op: Opcode, Val: ValueRef,
        DestTy: TypeRef, _Name: sbuf) ->
    ValueRef {
    ret llvm::LLVMBuildCast(*cx.build, Op, Val, DestTy, str::buf(""));
}

fn PointerCast(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildPointerCast(*cx.build, Val, DestTy, str::buf(""));
}

fn IntCast(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildIntCast(*cx.build, Val, DestTy, str::buf(""));
}

fn FPCast(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildFPCast(*cx.build, Val, DestTy, str::buf(""));
}


/* Comparisons */
fn ICmp(cx: &@block_ctxt, Op: uint, LHS: ValueRef,
        RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildICmp(*cx.build, Op, LHS, RHS, str::buf(""));
}

fn FCmp(cx: &@block_ctxt, Op: uint, LHS: ValueRef,
        RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildFCmp(*cx.build, Op, LHS, RHS, str::buf(""));
}


/* Miscellaneous instructions */
fn Phi(cx: &@block_ctxt, Ty: TypeRef, vals: &[ValueRef],
       bbs: &[BasicBlockRef]) -> ValueRef {
    let phi = llvm::LLVMBuildPhi(*cx.build, Ty, str::buf(""));
    assert (vec::len::<ValueRef>(vals) == vec::len::<BasicBlockRef>(bbs));
    llvm::LLVMAddIncoming(phi, vec::to_ptr(vals), vec::to_ptr(bbs),
                          vec::len(vals));
    ret phi;
}

fn AddIncomingToPhi(phi: ValueRef, vals: &[ValueRef], bbs: &[BasicBlockRef]) {
    assert (vec::len::<ValueRef>(vals) == vec::len::<BasicBlockRef>(bbs));
    llvm::LLVMAddIncoming(phi, vec::to_ptr(vals), vec::to_ptr(bbs),
                          vec::len(vals));
}

fn Call(cx: &@block_ctxt, Fn: ValueRef, Args: &[ValueRef]) -> ValueRef {
    ret llvm::LLVMBuildCall(*cx.build, Fn, vec::to_ptr(Args), vec::len(Args),
                            str::buf(""));
}

fn FastCall(cx: &@block_ctxt, Fn: ValueRef, Args: &[ValueRef]) -> ValueRef {
    let v =
        llvm::LLVMBuildCall(*cx.build, Fn, vec::to_ptr(Args), vec::len(Args),
                            str::buf(""));
    llvm::LLVMSetInstructionCallConv(v, lib::llvm::LLVMFastCallConv);
    ret v;
}

fn CallWithConv(cx: &@block_ctxt, Fn: ValueRef, Args: &[ValueRef],
                Conv: uint) -> ValueRef {
    let v =
        llvm::LLVMBuildCall(*cx.build, Fn, vec::to_ptr(Args), vec::len(Args),
                            str::buf(""));
    llvm::LLVMSetInstructionCallConv(v, Conv);
    ret v;
}

fn Select(cx: &@block_ctxt, If: ValueRef, Then: ValueRef,
          Else: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildSelect(*cx.build, If, Then, Else, str::buf(""));
}

fn VAArg(cx: &@block_ctxt, list: ValueRef, Ty: TypeRef) -> ValueRef {
    ret llvm::LLVMBuildVAArg(*cx.build, list, Ty, str::buf(""));
}

fn ExtractElement(cx: &@block_ctxt, VecVal: ValueRef,
                  Index: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildExtractElement(*cx.build, VecVal, Index, str::buf(""));
}

fn InsertElement(cx: &@block_ctxt, VecVal: ValueRef, EltVal: ValueRef,
                 Index: ValueRef) ->
    ValueRef {
    ret llvm::LLVMBuildInsertElement(*cx.build, VecVal, EltVal, Index,
                                     str::buf(""));
}

fn ShuffleVector(cx: &@block_ctxt, V1: ValueRef, V2: ValueRef,
                 Mask: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildShuffleVector(*cx.build, V1, V2, Mask, str::buf(""));
}

fn ExtractValue(cx: &@block_ctxt, AggVal: ValueRef, Index: uint) -> ValueRef {
    ret llvm::LLVMBuildExtractValue(*cx.build, AggVal, Index, str::buf(""));
}

fn InsertValue(cx: &@block_ctxt, AggVal: ValueRef,
               EltVal: ValueRef, Index: uint) -> ValueRef {
    ret llvm::LLVMBuildInsertValue(*cx.build, AggVal, EltVal, Index,
                                   str::buf(""));
}

fn IsNull(cx: &@block_ctxt, Val: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildIsNull(*cx.build, Val, str::buf(""));
}

fn IsNotNull(cx: &@block_ctxt, Val: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildIsNotNull(*cx.build, Val, str::buf(""));
}

fn PtrDiff(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    ret llvm::LLVMBuildPtrDiff(*cx.build, LHS, RHS, str::buf(""));
}

fn Trap(cx: &@block_ctxt) -> ValueRef {
    let BB: BasicBlockRef = llvm::LLVMGetInsertBlock(*cx.build);
    let FN: ValueRef = llvm::LLVMGetBasicBlockParent(BB);
    let M: ModuleRef = llvm::LLVMGetGlobalParent(FN);
    let T: ValueRef =
        llvm::LLVMGetNamedFunction(M, str::buf("llvm.trap"));
    assert (T as int != 0);
    let Args: [ValueRef] = [];
    ret llvm::LLVMBuildCall(*cx.build, T, vec::to_ptr(Args), vec::len(Args),
                            str::buf(""));
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
