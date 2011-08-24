import std::{vec, str};
import str::rustrt::sbuf;
import lib::llvm::llvm;
import llvm::{ValueRef, TypeRef, BasicBlockRef, BuilderRef,
              Opcode, ModuleRef};
import trans_common::block_ctxt;

fn RetVoid(cx: &@block_ctxt) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildRetVoid(B);
}

fn Ret(cx: &@block_ctxt, V: ValueRef) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildRet(B, V);
}

fn AggregateRet(cx: &@block_ctxt, RetVals: &[ValueRef]) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildAggregateRet(B, vec::to_ptr(RetVals),
                                    vec::len(RetVals));
}

fn Br(cx: &@block_ctxt, Dest: BasicBlockRef) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildBr(B, Dest);
}

fn CondBr(cx: &@block_ctxt, If: ValueRef, Then: BasicBlockRef,
          Else: BasicBlockRef) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildCondBr(B, If, Then, Else);
}

fn Switch(cx: &@block_ctxt, V: ValueRef, Else: BasicBlockRef,
          NumCases: uint) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildSwitch(B, V, Else, NumCases);
}

fn IndirectBr(cx: &@block_ctxt, Addr: ValueRef,
              NumDests: uint) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildIndirectBr(B, Addr, NumDests);
}

fn Invoke(cx: &@block_ctxt, Fn: ValueRef, Args: &[ValueRef],
          Then: BasicBlockRef, Catch: BasicBlockRef) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildInvoke(B, Fn, vec::to_ptr(Args),
                              vec::len(Args), Then, Catch, str::buf(""));
}

fn Unreachable(cx: &@block_ctxt) -> ValueRef {
    assert (!cx.terminated);;
    cx.terminated = true;
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildUnreachable(B);
}

/* Arithmetic */
fn Add(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildAdd(B, LHS, RHS, str::buf(""));
}

fn NSWAdd(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildNSWAdd(B, LHS, RHS, str::buf(""));
}

fn NUWAdd(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildNUWAdd(B, LHS, RHS, str::buf(""));
}

fn FAdd(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildFAdd(B, LHS, RHS, str::buf(""));
}

fn Sub(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildSub(B, LHS, RHS, str::buf(""));
}

fn NSWSub(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildNSWSub(B, LHS, RHS, str::buf(""));
}

fn NUWSub(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildNUWSub(B, LHS, RHS, str::buf(""));
}

fn FSub(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildFSub(B, LHS, RHS, str::buf(""));
}

fn Mul(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildMul(B, LHS, RHS, str::buf(""));
}

fn NSWMul(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildNSWMul(B, LHS, RHS, str::buf(""));
}

fn NUWMul(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildNUWMul(B, LHS, RHS, str::buf(""));
}

fn FMul(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildFMul(B, LHS, RHS, str::buf(""));
}

fn UDiv(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildUDiv(B, LHS, RHS, str::buf(""));
}

fn SDiv(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildSDiv(B, LHS, RHS, str::buf(""));
}

fn ExactSDiv(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildExactSDiv(B, LHS, RHS, str::buf(""));
}

fn FDiv(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildFDiv(B, LHS, RHS, str::buf(""));
}

fn URem(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildURem(B, LHS, RHS, str::buf(""));
}

fn SRem(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildSRem(B, LHS, RHS, str::buf(""));
}

fn FRem(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildFRem(B, LHS, RHS, str::buf(""));
}

fn Shl(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildShl(B, LHS, RHS, str::buf(""));
}

fn LShr(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildLShr(B, LHS, RHS, str::buf(""));
}

fn AShr(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildAShr(B, LHS, RHS, str::buf(""));
}

fn And(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildAnd(B, LHS, RHS, str::buf(""));
}

fn Or(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildOr(B, LHS, RHS, str::buf(""));
}

fn Xor(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildXor(B, LHS, RHS, str::buf(""));
}

fn BinOp(cx: &@block_ctxt, Op: Opcode, LHS: ValueRef,
         RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildBinOp(B, Op, LHS, RHS, str::buf(""));
}

fn Neg(cx: &@block_ctxt, V: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildNeg(B, V, str::buf(""));
}

fn NSWNeg(cx: &@block_ctxt, V: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildNSWNeg(B, V, str::buf(""));
}

fn NUWNeg(cx: &@block_ctxt, V: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildNUWNeg(B, V, str::buf(""));
}
fn FNeg(cx: &@block_ctxt, V: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildFNeg(B, V, str::buf(""));
}
fn Not(cx: &@block_ctxt, V: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildNot(B, V, str::buf(""));
}

/* Memory */
fn Malloc(cx: &@block_ctxt, Ty: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildMalloc(B, Ty, str::buf(""));
}

fn ArrayMalloc(cx: &@block_ctxt, Ty: TypeRef, Val: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildArrayMalloc(B, Ty, Val, str::buf(""));
}

fn Alloca(cx: &@block_ctxt, Ty: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildAlloca(B, Ty, str::buf(""));
}

fn ArrayAlloca(cx: &@block_ctxt, Ty: TypeRef, Val: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildArrayAlloca(B, Ty, Val, str::buf(""));
}

fn Free(cx: &@block_ctxt, PointerVal: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildFree(B, PointerVal);
}

fn Load(cx: &@block_ctxt, PointerVal: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildLoad(B, PointerVal, str::buf(""));
}

fn Store(cx: &@block_ctxt, Val: ValueRef, Ptr: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildStore(B, Val, Ptr);
}

fn GEP(cx: &@block_ctxt, Pointer: ValueRef,
       Indices: &[ValueRef]) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildGEP(B, Pointer, vec::to_ptr(Indices),
                           vec::len(Indices), str::buf(""));
}

fn InBoundsGEP(cx: &@block_ctxt, Pointer: ValueRef,
               Indices: &[ValueRef]) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildInBoundsGEP(B, Pointer, vec::to_ptr(Indices),
                                   vec::len(Indices), str::buf(""));
}

fn StructGEP(cx: &@block_ctxt, Pointer: ValueRef, Idx: uint) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildStructGEP(B, Pointer, Idx, str::buf(""));
}

fn GlobalString(cx: &@block_ctxt, _Str: sbuf) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildGlobalString(B, _Str, str::buf(""));
}

fn GlobalStringPtr(cx: &@block_ctxt, _Str: sbuf) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildGlobalStringPtr(B, _Str, str::buf(""));
}

/* Casts */
fn Trunc(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildTrunc(B, Val, DestTy, str::buf(""));
}

fn ZExt(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildZExt(B, Val, DestTy, str::buf(""));
}

fn SExt(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildSExt(B, Val, DestTy, str::buf(""));
}

fn FPToUI(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildFPToUI(B, Val, DestTy, str::buf(""));
}

fn FPToSI(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildFPToSI(B, Val, DestTy, str::buf(""));
}

fn UIToFP(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildUIToFP(B, Val, DestTy, str::buf(""));
}

fn SIToFP(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildSIToFP(B, Val, DestTy, str::buf(""));
}

fn FPTrunc(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildFPTrunc(B, Val, DestTy, str::buf(""));
}

fn FPExt(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildFPExt(B, Val, DestTy, str::buf(""));
}

fn PtrToInt(cx: &@block_ctxt, Val: ValueRef,
            DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildPtrToInt(B, Val, DestTy, str::buf(""));
}

fn IntToPtr(cx: &@block_ctxt, Val: ValueRef,
            DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildIntToPtr(B, Val, DestTy, str::buf(""));
}

fn BitCast(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildBitCast(B, Val, DestTy, str::buf(""));
}

fn ZExtOrBitCast(cx: &@block_ctxt, Val: ValueRef,
                 DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildZExtOrBitCast(B, Val, DestTy, str::buf(""));
}

fn SExtOrBitCast(cx: &@block_ctxt, Val: ValueRef,
                 DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildSExtOrBitCast(B, Val, DestTy, str::buf(""));
}

fn TruncOrBitCast(cx: &@block_ctxt, Val: ValueRef,
                  DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildTruncOrBitCast(B, Val, DestTy, str::buf(""));
}

fn Cast(cx: &@block_ctxt, Op: Opcode, Val: ValueRef,
        DestTy: TypeRef, _Name: sbuf) ->
    ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildCast(B, Op, Val, DestTy, str::buf(""));
}

fn PointerCast(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildPointerCast(B, Val, DestTy, str::buf(""));
}

fn IntCast(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildIntCast(B, Val, DestTy, str::buf(""));
}

fn FPCast(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildFPCast(B, Val, DestTy, str::buf(""));
}


/* Comparisons */
fn ICmp(cx: &@block_ctxt, Op: uint, LHS: ValueRef,
        RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildICmp(B, Op, LHS, RHS, str::buf(""));
}

fn FCmp(cx: &@block_ctxt, Op: uint, LHS: ValueRef,
        RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildFCmp(B, Op, LHS, RHS, str::buf(""));
}


/* Miscellaneous instructions */
fn Phi(cx: &@block_ctxt, Ty: TypeRef, vals: &[ValueRef],
       bbs: &[BasicBlockRef]) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    let phi = llvm::LLVMBuildPhi(B, Ty, str::buf(""));
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
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildCall(B, Fn, vec::to_ptr(Args), vec::len(Args),
                            str::buf(""));
}

fn FastCall(cx: &@block_ctxt, Fn: ValueRef, Args: &[ValueRef]) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    let v =
        llvm::LLVMBuildCall(B, Fn, vec::to_ptr(Args), vec::len(Args),
                            str::buf(""));
    llvm::LLVMSetInstructionCallConv(v, lib::llvm::LLVMFastCallConv);
    ret v;
}

fn CallWithConv(cx: &@block_ctxt, Fn: ValueRef, Args: &[ValueRef],
                Conv: uint) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    let v =
        llvm::LLVMBuildCall(B, Fn, vec::to_ptr(Args), vec::len(Args),
                            str::buf(""));
    llvm::LLVMSetInstructionCallConv(v, Conv);
    ret v;
}

fn Select(cx: &@block_ctxt, If: ValueRef, Then: ValueRef,
          Else: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildSelect(B, If, Then, Else, str::buf(""));
}

fn VAArg(cx: &@block_ctxt, list: ValueRef, Ty: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildVAArg(B, list, Ty, str::buf(""));
}

fn ExtractElement(cx: &@block_ctxt, VecVal: ValueRef,
                  Index: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildExtractElement(B, VecVal, Index, str::buf(""));
}

fn InsertElement(cx: &@block_ctxt, VecVal: ValueRef, EltVal: ValueRef,
                 Index: ValueRef) ->
    ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildInsertElement(B, VecVal, EltVal, Index,
                                     str::buf(""));
}

fn ShuffleVector(cx: &@block_ctxt, V1: ValueRef, V2: ValueRef,
                 Mask: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildShuffleVector(B, V1, V2, Mask, str::buf(""));
}

fn ExtractValue(cx: &@block_ctxt, AggVal: ValueRef, Index: uint) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildExtractValue(B, AggVal, Index, str::buf(""));
}

fn InsertValue(cx: &@block_ctxt, AggVal: ValueRef,
               EltVal: ValueRef, Index: uint) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildInsertValue(B, AggVal, EltVal, Index,
                                   str::buf(""));
}

fn IsNull(cx: &@block_ctxt, Val: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildIsNull(B, Val, str::buf(""));
}

fn IsNotNull(cx: &@block_ctxt, Val: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildIsNotNull(B, Val, str::buf(""));
}

fn PtrDiff(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildPtrDiff(B, LHS, RHS, str::buf(""));
}

fn Trap(cx: &@block_ctxt) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    let BB: BasicBlockRef = llvm::LLVMGetInsertBlock(B);
    let FN: ValueRef = llvm::LLVMGetBasicBlockParent(BB);
    let M: ModuleRef = llvm::LLVMGetGlobalParent(FN);
    let T: ValueRef =
        llvm::LLVMGetNamedFunction(M, str::buf("llvm.trap"));
    assert (T as int != 0);
    let Args: [ValueRef] = [];
    ret llvm::LLVMBuildCall(B, T, vec::to_ptr(Args), vec::len(Args),
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
