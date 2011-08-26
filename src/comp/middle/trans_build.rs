import std::{vec, str, istr};
import std::istr::sbuf;
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
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildInvoke(B, Fn, vec::to_ptr(Args),
                              vec::len(Args), Then, Catch, buf)
    });
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
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildAdd(B, LHS, RHS, buf)
    });
}

fn NSWAdd(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildNSWAdd(B, LHS, RHS, buf)
    });
}

fn NUWAdd(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildNUWAdd(B, LHS, RHS, buf)
    });
}

fn FAdd(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildFAdd(B, LHS, RHS, buf)
    });
}

fn Sub(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildSub(B, LHS, RHS, buf)
    });
}

fn NSWSub(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildNSWSub(B, LHS, RHS, buf)
    });
}

fn NUWSub(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildNUWSub(B, LHS, RHS, buf)
    });
}

fn FSub(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildFSub(B, LHS, RHS, buf)
    });
}

fn Mul(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildMul(B, LHS, RHS, buf)
    });
}

fn NSWMul(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildNSWMul(B, LHS, RHS, buf)
    });
}

fn NUWMul(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildNUWMul(B, LHS, RHS, buf)
    });
}

fn FMul(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildFMul(B, LHS, RHS, buf)
    });
}

fn UDiv(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildUDiv(B, LHS, RHS, buf)
    });
}

fn SDiv(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildSDiv(B, LHS, RHS, buf)
    });
}

fn ExactSDiv(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildExactSDiv(B, LHS, RHS, buf)
    });
}

fn FDiv(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildFDiv(B, LHS, RHS, buf)
    });
}

fn URem(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildURem(B, LHS, RHS, buf)
    });
}

fn SRem(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildSRem(B, LHS, RHS, buf)
    });
}

fn FRem(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildFRem(B, LHS, RHS, buf)
    });
}

fn Shl(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildShl(B, LHS, RHS, buf)
    });
}

fn LShr(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildLShr(B, LHS, RHS, buf)
    });
}

fn AShr(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildAShr(B, LHS, RHS, buf)
    });
}

fn And(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildAnd(B, LHS, RHS, buf)
    });
}

fn Or(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildOr(B, LHS, RHS, buf)
    });
}

fn Xor(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildXor(B, LHS, RHS, buf)
    });
}

fn BinOp(cx: &@block_ctxt, Op: Opcode, LHS: ValueRef,
         RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildBinOp(B, Op, LHS, RHS, buf)
    });
}

fn Neg(cx: &@block_ctxt, V: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildNeg(B, V, buf)
    });
}

fn NSWNeg(cx: &@block_ctxt, V: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildNSWNeg(B, V, buf)
    });
}

fn NUWNeg(cx: &@block_ctxt, V: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildNUWNeg(B, V, buf)
    });
}
fn FNeg(cx: &@block_ctxt, V: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildFNeg(B, V, buf)
    });
}

fn Not(cx: &@block_ctxt, V: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildNot(B, V, buf)
    });
}

/* Memory */
fn Malloc(cx: &@block_ctxt, Ty: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildMalloc(B, Ty, buf)
    });
}

fn ArrayMalloc(cx: &@block_ctxt, Ty: TypeRef, Val: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildArrayMalloc(B, Ty, Val, buf)
    });
}

fn Alloca(cx: &@block_ctxt, Ty: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildAlloca(B, Ty, buf)
    });
}

fn ArrayAlloca(cx: &@block_ctxt, Ty: TypeRef, Val: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildArrayAlloca(B, Ty, Val, buf)
    });
}

fn Free(cx: &@block_ctxt, PointerVal: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret llvm::LLVMBuildFree(B, PointerVal);
}

fn Load(cx: &@block_ctxt, PointerVal: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildLoad(B, PointerVal, buf)
    });
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
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildGEP(B, Pointer, vec::to_ptr(Indices),
                           vec::len(Indices), buf)
    });
}

fn InBoundsGEP(cx: &@block_ctxt, Pointer: ValueRef,
               Indices: &[ValueRef]) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildInBoundsGEP(B, Pointer, vec::to_ptr(Indices),
                                   vec::len(Indices), buf)
    });
}

fn StructGEP(cx: &@block_ctxt, Pointer: ValueRef, Idx: uint) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildStructGEP(B, Pointer, Idx, buf)
    });
}

fn GlobalString(cx: &@block_ctxt, _Str: sbuf) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildGlobalString(B, _Str, buf)
    });
}

fn GlobalStringPtr(cx: &@block_ctxt, _Str: sbuf) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildGlobalStringPtr(B, _Str, buf)
    });
}

/* Casts */
fn Trunc(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildTrunc(B, Val, DestTy, buf)
    });
}

fn ZExt(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildZExt(B, Val, DestTy, buf)
    });
}

fn SExt(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildSExt(B, Val, DestTy, buf)
    });
}

fn FPToUI(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildFPToUI(B, Val, DestTy, buf)
    });
}

fn FPToSI(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildFPToSI(B, Val, DestTy, buf)
    });
}

fn UIToFP(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildUIToFP(B, Val, DestTy, buf)
    });
}

fn SIToFP(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildSIToFP(B, Val, DestTy, buf)
    });
}

fn FPTrunc(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildFPTrunc(B, Val, DestTy, buf)
    });
}

fn FPExt(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildFPExt(B, Val, DestTy, buf)
    });
}

fn PtrToInt(cx: &@block_ctxt, Val: ValueRef,
            DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildPtrToInt(B, Val, DestTy, buf)
    });
}

fn IntToPtr(cx: &@block_ctxt, Val: ValueRef,
            DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildIntToPtr(B, Val, DestTy, buf)
    });
}

fn BitCast(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildBitCast(B, Val, DestTy, buf)
    });
}

fn ZExtOrBitCast(cx: &@block_ctxt, Val: ValueRef,
                 DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildZExtOrBitCast(B, Val, DestTy, buf)
    });
}

fn SExtOrBitCast(cx: &@block_ctxt, Val: ValueRef,
                 DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildSExtOrBitCast(B, Val, DestTy, buf)
    });
}

fn TruncOrBitCast(cx: &@block_ctxt, Val: ValueRef,
                  DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildTruncOrBitCast(B, Val, DestTy, buf)
    });
}

fn Cast(cx: &@block_ctxt, Op: Opcode, Val: ValueRef,
        DestTy: TypeRef, _Name: sbuf) ->
    ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildCast(B, Op, Val, DestTy, buf)
    });
}

fn PointerCast(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildPointerCast(B, Val, DestTy, buf)
    });
}

fn IntCast(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildIntCast(B, Val, DestTy, buf)
    });
}

fn FPCast(cx: &@block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildFPCast(B, Val, DestTy, buf)
    });
}


/* Comparisons */
fn ICmp(cx: &@block_ctxt, Op: uint, LHS: ValueRef,
        RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildICmp(B, Op, LHS, RHS, buf)
    });
}

fn FCmp(cx: &@block_ctxt, Op: uint, LHS: ValueRef,
        RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildFCmp(B, Op, LHS, RHS, buf)
    });
}


/* Miscellaneous instructions */
fn Phi(cx: &@block_ctxt, Ty: TypeRef, vals: &[ValueRef],
       bbs: &[BasicBlockRef]) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    let phi = istr::as_buf(~"", { |buf|
        llvm::LLVMBuildPhi(B, Ty, buf)
    });
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
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildCall(B, Fn, vec::to_ptr(Args),
                            vec::len(Args), buf)
    });
}

fn FastCall(cx: &@block_ctxt, Fn: ValueRef, Args: &[ValueRef]) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    let v = istr::as_buf(~"", { |buf|
        llvm::LLVMBuildCall(B, Fn, vec::to_ptr(Args), vec::len(Args), buf)
    });
    llvm::LLVMSetInstructionCallConv(v, lib::llvm::LLVMFastCallConv);
    ret v;
}

fn CallWithConv(cx: &@block_ctxt, Fn: ValueRef, Args: &[ValueRef],
                Conv: uint) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    let v = istr::as_buf(~"", { |buf|
        llvm::LLVMBuildCall(B, Fn, vec::to_ptr(Args), vec::len(Args), buf)
    });
    llvm::LLVMSetInstructionCallConv(v, Conv);
    ret v;
}

fn Select(cx: &@block_ctxt, If: ValueRef, Then: ValueRef,
          Else: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildSelect(B, If, Then, Else, buf)
    });
}

fn VAArg(cx: &@block_ctxt, list: ValueRef, Ty: TypeRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildVAArg(B, list, Ty, buf)
    });
}

fn ExtractElement(cx: &@block_ctxt, VecVal: ValueRef,
                  Index: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildExtractElement(B, VecVal, Index, buf)
    });
}

fn InsertElement(cx: &@block_ctxt, VecVal: ValueRef, EltVal: ValueRef,
                 Index: ValueRef) ->
    ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildInsertElement(B, VecVal, EltVal, Index, buf)
    });
}

fn ShuffleVector(cx: &@block_ctxt, V1: ValueRef, V2: ValueRef,
                 Mask: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildShuffleVector(B, V1, V2, Mask, buf)
    });
}

fn ExtractValue(cx: &@block_ctxt, AggVal: ValueRef, Index: uint) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildExtractValue(B, AggVal, Index, buf)
    });
}

fn InsertValue(cx: &@block_ctxt, AggVal: ValueRef,
               EltVal: ValueRef, Index: uint) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildInsertValue(B, AggVal, EltVal, Index, buf)
    });
}

fn IsNull(cx: &@block_ctxt, Val: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildIsNull(B, Val, buf)
    });
}

fn IsNotNull(cx: &@block_ctxt, Val: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildIsNotNull(B, Val, buf)
    });
}

fn PtrDiff(cx: &@block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildPtrDiff(B, LHS, RHS, buf)
    });
}

fn Trap(cx: &@block_ctxt) -> ValueRef {
    let B = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(B, cx.llbb);
    let BB: BasicBlockRef = llvm::LLVMGetInsertBlock(B);
    let FN: ValueRef = llvm::LLVMGetBasicBlockParent(BB);
    let M: ModuleRef = llvm::LLVMGetGlobalParent(FN);
    let T: ValueRef = istr::as_buf(~"llvm.trap", { |buf|
        llvm::LLVMGetNamedFunction(M, buf)
    });
    assert (T as int != 0);
    let Args: [ValueRef] = [];
    ret istr::as_buf(~"", { |buf|
        llvm::LLVMBuildCall(B, T, vec::to_ptr(Args), vec::len(Args), buf)
    });
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
