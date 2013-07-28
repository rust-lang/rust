// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use lib::llvm::llvm;
use lib::llvm::{CallConv, AtomicBinOp, AtomicOrdering, AsmDialect};
use lib::llvm::{Opcode, IntPredicate, RealPredicate};
use lib::llvm::{ValueRef, BasicBlockRef};
use lib;
use middle::trans::common::*;
use syntax::codemap::span;

use middle::trans::builder::Builder;
use middle::trans::type_::Type;

use std::cast;
use std::libc::{c_uint, c_ulonglong, c_char};

pub fn terminate(cx: @mut Block, _: &str) {
    cx.terminated = true;
}

pub fn check_not_terminated(cx: @mut Block) {
    if cx.terminated {
        fail!("already terminated!");
    }
}

pub fn B(cx: @mut Block) -> Builder {
    let b = cx.fcx.ccx.builder();
    b.position_at_end(cx.llbb);
    b
}

// The difference between a block being unreachable and being terminated is
// somewhat obscure, and has to do with error checking. When a block is
// terminated, we're saying that trying to add any further statements in the
// block is an error. On the other hand, if something is unreachable, that
// means that the block was terminated in some way that we don't want to check
// for (fail/break/return statements, call to diverging functions, etc), and
// further instructions to the block should simply be ignored.

pub fn RetVoid(cx: @mut Block) {
    if cx.unreachable { return; }
    check_not_terminated(cx);
    terminate(cx, "RetVoid");
    B(cx).ret_void();
}

pub fn Ret(cx: @mut Block, V: ValueRef) {
    if cx.unreachable { return; }
    check_not_terminated(cx);
    terminate(cx, "Ret");
    B(cx).ret(V);
}

pub fn AggregateRet(cx: @mut Block, RetVals: &[ValueRef]) {
    if cx.unreachable { return; }
    check_not_terminated(cx);
    terminate(cx, "AggregateRet");
    B(cx).aggregate_ret(RetVals);
}

pub fn Br(cx: @mut Block, Dest: BasicBlockRef) {
    if cx.unreachable { return; }
    check_not_terminated(cx);
    terminate(cx, "Br");
    B(cx).br(Dest);
}

pub fn CondBr(cx: @mut Block, If: ValueRef, Then: BasicBlockRef,
              Else: BasicBlockRef) {
    if cx.unreachable { return; }
    check_not_terminated(cx);
    terminate(cx, "CondBr");
    B(cx).cond_br(If, Then, Else);
}

pub fn Switch(cx: @mut Block, V: ValueRef, Else: BasicBlockRef, NumCases: uint)
    -> ValueRef {
    if cx.unreachable { return _Undef(V); }
    check_not_terminated(cx);
    terminate(cx, "Switch");
    B(cx).switch(V, Else, NumCases)
}

pub fn AddCase(S: ValueRef, OnVal: ValueRef, Dest: BasicBlockRef) {
    unsafe {
        if llvm::LLVMIsUndef(S) == lib::llvm::True { return; }
        llvm::LLVMAddCase(S, OnVal, Dest);
    }
}

pub fn IndirectBr(cx: @mut Block, Addr: ValueRef, NumDests: uint) {
    if cx.unreachable { return; }
    check_not_terminated(cx);
    terminate(cx, "IndirectBr");
    B(cx).indirect_br(Addr, NumDests);
}

pub fn Invoke(cx: @mut Block,
              Fn: ValueRef,
              Args: &[ValueRef],
              Then: BasicBlockRef,
              Catch: BasicBlockRef)
           -> ValueRef {
    if cx.unreachable {
        return C_null(Type::i8());
    }
    check_not_terminated(cx);
    terminate(cx, "Invoke");
    debug!("Invoke(%s with arguments (%s))",
           cx.val_to_str(Fn),
           Args.map(|a| cx.val_to_str(*a)).connect(", "));
    B(cx).invoke(Fn, Args, Then, Catch)
}

pub fn FastInvoke(cx: @mut Block, Fn: ValueRef, Args: &[ValueRef],
                  Then: BasicBlockRef, Catch: BasicBlockRef) {
    if cx.unreachable { return; }
    check_not_terminated(cx);
    terminate(cx, "FastInvoke");
    B(cx).fast_invoke(Fn, Args, Then, Catch);
}

pub fn Unreachable(cx: @mut Block) {
    if cx.unreachable { return; }
    cx.unreachable = true;
    if !cx.terminated {
        B(cx).unreachable();
    }
}

pub fn _Undef(val: ValueRef) -> ValueRef {
    unsafe {
        return llvm::LLVMGetUndef(val_ty(val).to_ref());
    }
}

/* Arithmetic */
pub fn Add(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).add(LHS, RHS)
}

pub fn NSWAdd(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).nswadd(LHS, RHS)
}

pub fn NUWAdd(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).nuwadd(LHS, RHS)
}

pub fn FAdd(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).fadd(LHS, RHS)
}

pub fn Sub(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).sub(LHS, RHS)
}

pub fn NSWSub(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).nswsub(LHS, RHS)
}

pub fn NUWSub(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).nuwsub(LHS, RHS)
}

pub fn FSub(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).fsub(LHS, RHS)
}

pub fn Mul(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).mul(LHS, RHS)
}

pub fn NSWMul(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).nswmul(LHS, RHS)
}

pub fn NUWMul(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).nuwmul(LHS, RHS)
}

pub fn FMul(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).fmul(LHS, RHS)
}

pub fn UDiv(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).udiv(LHS, RHS)
}

pub fn SDiv(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).sdiv(LHS, RHS)
}

pub fn ExactSDiv(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).exactsdiv(LHS, RHS)
}

pub fn FDiv(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).fdiv(LHS, RHS)
}

pub fn URem(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).urem(LHS, RHS)
}

pub fn SRem(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).srem(LHS, RHS)
}

pub fn FRem(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).frem(LHS, RHS)
}

pub fn Shl(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).shl(LHS, RHS)
}

pub fn LShr(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).lshr(LHS, RHS)
}

pub fn AShr(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).ashr(LHS, RHS)
}

pub fn And(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).and(LHS, RHS)
}

pub fn Or(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).or(LHS, RHS)
}

pub fn Xor(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).xor(LHS, RHS)
}

pub fn BinOp(cx: @mut Block, Op: Opcode, LHS: ValueRef, RHS: ValueRef)
          -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    B(cx).binop(Op, LHS, RHS)
}

pub fn Neg(cx: @mut Block, V: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(V); }
    B(cx).neg(V)
}

pub fn NSWNeg(cx: @mut Block, V: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(V); }
    B(cx).nswneg(V)
}

pub fn NUWNeg(cx: @mut Block, V: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(V); }
    B(cx).nuwneg(V)
}
pub fn FNeg(cx: @mut Block, V: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(V); }
    B(cx).fneg(V)
}

pub fn Not(cx: @mut Block, V: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(V); }
    B(cx).not(V)
}

/* Memory */
pub fn Malloc(cx: @mut Block, Ty: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::i8p().to_ref()); }
        B(cx).malloc(Ty)
    }
}

pub fn ArrayMalloc(cx: @mut Block, Ty: Type, Val: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::i8p().to_ref()); }
        B(cx).array_malloc(Ty, Val)
    }
}

pub fn Alloca(cx: @mut Block, Ty: Type, name: &str) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Ty.ptr_to().to_ref()); }
        let b = cx.fcx.ccx.builder();
        b.position_before(cx.fcx.alloca_insert_pt.get());
        b.alloca(Ty, name)
    }
}

pub fn ArrayAlloca(cx: @mut Block, Ty: Type, Val: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Ty.ptr_to().to_ref()); }
        let b = cx.fcx.ccx.builder();
        b.position_before(cx.fcx.alloca_insert_pt.get());
        b.array_alloca(Ty, Val)
    }
}

pub fn Free(cx: @mut Block, PointerVal: ValueRef) {
    if cx.unreachable { return; }
    B(cx).free(PointerVal)
}

pub fn Load(cx: @mut Block, PointerVal: ValueRef) -> ValueRef {
    unsafe {
        let ccx = cx.fcx.ccx;
        if cx.unreachable {
            let ty = val_ty(PointerVal);
            let eltty = if ty.kind() == lib::llvm::Array {
                ty.element_type()
            } else {
                ccx.int_type
            };
            return llvm::LLVMGetUndef(eltty.to_ref());
        }
        B(cx).load(PointerVal)
    }
}

pub fn AtomicLoad(cx: @mut Block, PointerVal: ValueRef, order: AtomicOrdering) -> ValueRef {
    unsafe {
        let ccx = cx.fcx.ccx;
        if cx.unreachable {
            return llvm::LLVMGetUndef(ccx.int_type.to_ref());
        }
        B(cx).atomic_load(PointerVal, order)
    }
}


pub fn LoadRangeAssert(cx: @mut Block, PointerVal: ValueRef, lo: c_ulonglong,
                       hi: c_ulonglong, signed: lib::llvm::Bool) -> ValueRef {
    if cx.unreachable {
        let ccx = cx.fcx.ccx;
        let ty = val_ty(PointerVal);
        let eltty = if ty.kind() == lib::llvm::Array {
            ty.element_type()
        } else {
            ccx.int_type
        };
        unsafe {
            llvm::LLVMGetUndef(eltty.to_ref())
        }
    } else {
        B(cx).load_range_assert(PointerVal, lo, hi, signed)
    }
}

pub fn Store(cx: @mut Block, Val: ValueRef, Ptr: ValueRef) {
    if cx.unreachable { return; }
    B(cx).store(Val, Ptr)
}

pub fn AtomicStore(cx: @mut Block, Val: ValueRef, Ptr: ValueRef, order: AtomicOrdering) {
    if cx.unreachable { return; }
    B(cx).atomic_store(Val, Ptr, order)
}

pub fn GEP(cx: @mut Block, Pointer: ValueRef, Indices: &[ValueRef]) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::nil().ptr_to().to_ref()); }
        B(cx).gep(Pointer, Indices)
    }
}

// Simple wrapper around GEP that takes an array of ints and wraps them
// in C_i32()
#[inline]
pub fn GEPi(cx: @mut Block, base: ValueRef, ixs: &[uint]) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::nil().ptr_to().to_ref()); }
        B(cx).gepi(base, ixs)
    }
}

pub fn InBoundsGEP(cx: @mut Block, Pointer: ValueRef, Indices: &[ValueRef]) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::nil().ptr_to().to_ref()); }
        B(cx).inbounds_gep(Pointer, Indices)
    }
}

pub fn StructGEP(cx: @mut Block, Pointer: ValueRef, Idx: uint) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::nil().ptr_to().to_ref()); }
        B(cx).struct_gep(Pointer, Idx)
    }
}

pub fn GlobalString(cx: @mut Block, _Str: *c_char) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::i8p().to_ref()); }
        B(cx).global_string(_Str)
    }
}

pub fn GlobalStringPtr(cx: @mut Block, _Str: *c_char) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::i8p().to_ref()); }
        B(cx).global_string_ptr(_Str)
    }
}

/* Casts */
pub fn Trunc(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).trunc(Val, DestTy)
    }
}

pub fn ZExt(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).zext(Val, DestTy)
    }
}

pub fn SExt(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).sext(Val, DestTy)
    }
}

pub fn FPToUI(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).fptoui(Val, DestTy)
    }
}

pub fn FPToSI(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).fptosi(Val, DestTy)
    }
}

pub fn UIToFP(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).uitofp(Val, DestTy)
    }
}

pub fn SIToFP(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).sitofp(Val, DestTy)
    }
}

pub fn FPTrunc(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).fptrunc(Val, DestTy)
    }
}

pub fn FPExt(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).fpext(Val, DestTy)
    }
}

pub fn PtrToInt(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).ptrtoint(Val, DestTy)
    }
}

pub fn IntToPtr(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).inttoptr(Val, DestTy)
    }
}

pub fn BitCast(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).bitcast(Val, DestTy)
    }
}

pub fn ZExtOrBitCast(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).zext_or_bitcast(Val, DestTy)
    }
}

pub fn SExtOrBitCast(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).sext_or_bitcast(Val, DestTy)
    }
}

pub fn TruncOrBitCast(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).trunc_or_bitcast(Val, DestTy)
    }
}

pub fn Cast(cx: @mut Block, Op: Opcode, Val: ValueRef, DestTy: Type, _: *u8)
     -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).cast(Op, Val, DestTy)
    }
}

pub fn PointerCast(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).pointercast(Val, DestTy)
    }
}

pub fn IntCast(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).intcast(Val, DestTy)
    }
}

pub fn FPCast(cx: @mut Block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        B(cx).fpcast(Val, DestTy)
    }
}


/* Comparisons */
pub fn ICmp(cx: @mut Block, Op: IntPredicate, LHS: ValueRef, RHS: ValueRef)
     -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::i1().to_ref()); }
        B(cx).icmp(Op, LHS, RHS)
    }
}

pub fn FCmp(cx: @mut Block, Op: RealPredicate, LHS: ValueRef, RHS: ValueRef)
     -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::i1().to_ref()); }
        B(cx).fcmp(Op, LHS, RHS)
    }
}

/* Miscellaneous instructions */
pub fn EmptyPhi(cx: @mut Block, Ty: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Ty.to_ref()); }
        B(cx).empty_phi(Ty)
    }
}

pub fn Phi(cx: @mut Block, Ty: Type, vals: &[ValueRef], bbs: &[BasicBlockRef]) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Ty.to_ref()); }
        B(cx).phi(Ty, vals, bbs)
    }
}

pub fn AddIncomingToPhi(phi: ValueRef, val: ValueRef, bb: BasicBlockRef) {
    unsafe {
        if llvm::LLVMIsUndef(phi) == lib::llvm::True { return; }
        let valptr = cast::transmute(&val);
        let bbptr = cast::transmute(&bb);
        llvm::LLVMAddIncoming(phi, valptr, bbptr, 1 as c_uint);
    }
}

pub fn _UndefReturn(cx: @mut Block, Fn: ValueRef) -> ValueRef {
    unsafe {
        let ccx = cx.fcx.ccx;
        let ty = val_ty(Fn);
        let retty = if ty.kind() == lib::llvm::Integer {
            ty.return_type()
        } else {
            ccx.int_type
        };
        B(cx).count_insn("ret_undef");
        llvm::LLVMGetUndef(retty.to_ref())
    }
}

pub fn add_span_comment(cx: @mut Block, sp: span, text: &str) {
    B(cx).add_span_comment(sp, text)
}

pub fn add_comment(cx: @mut Block, text: &str) {
    B(cx).add_comment(text)
}

pub fn InlineAsmCall(cx: @mut Block, asm: *c_char, cons: *c_char,
                     inputs: &[ValueRef], output: Type,
                     volatile: bool, alignstack: bool,
                     dia: AsmDialect) -> ValueRef {
    B(cx).inline_asm_call(asm, cons, inputs, output, volatile, alignstack, dia)
}

pub fn Call(cx: @mut Block, Fn: ValueRef, Args: &[ValueRef]) -> ValueRef {
    if cx.unreachable { return _UndefReturn(cx, Fn); }
    B(cx).call(Fn, Args)
}

pub fn FastCall(cx: @mut Block, Fn: ValueRef, Args: &[ValueRef]) -> ValueRef {
    if cx.unreachable { return _UndefReturn(cx, Fn); }
    B(cx).call(Fn, Args)
}

pub fn CallWithConv(cx: @mut Block, Fn: ValueRef, Args: &[ValueRef],
                    Conv: CallConv) -> ValueRef {
    if cx.unreachable { return _UndefReturn(cx, Fn); }
    B(cx).call_with_conv(Fn, Args, Conv)
}

pub fn AtomicFence(cx: @mut Block, order: AtomicOrdering) {
    if cx.unreachable { return; }
    B(cx).atomic_fence(order)
}

pub fn Select(cx: @mut Block, If: ValueRef, Then: ValueRef, Else: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(Then); }
    B(cx).select(If, Then, Else)
}

pub fn VAArg(cx: @mut Block, list: ValueRef, Ty: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Ty.to_ref()); }
        B(cx).va_arg(list, Ty)
    }
}

pub fn ExtractElement(cx: @mut Block, VecVal: ValueRef, Index: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::nil().to_ref()); }
        B(cx).extract_element(VecVal, Index)
    }
}

pub fn InsertElement(cx: @mut Block, VecVal: ValueRef, EltVal: ValueRef,
                     Index: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::nil().to_ref()); }
        B(cx).insert_element(VecVal, EltVal, Index)
    }
}

pub fn ShuffleVector(cx: @mut Block, V1: ValueRef, V2: ValueRef,
                     Mask: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::nil().to_ref()); }
        B(cx).shuffle_vector(V1, V2, Mask)
    }
}

pub fn VectorSplat(cx: @mut Block, NumElts: uint, EltVal: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::nil().to_ref()); }
        B(cx).vector_splat(NumElts, EltVal)
    }
}

pub fn ExtractValue(cx: @mut Block, AggVal: ValueRef, Index: uint) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::nil().to_ref()); }
        B(cx).extract_value(AggVal, Index)
    }
}

pub fn InsertValue(cx: @mut Block, AggVal: ValueRef, EltVal: ValueRef, Index: uint) {
    if cx.unreachable { return; }
    B(cx).insert_value(AggVal, EltVal, Index)
}

pub fn IsNull(cx: @mut Block, Val: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::i1().to_ref()); }
        B(cx).is_null(Val)
    }
}

pub fn IsNotNull(cx: @mut Block, Val: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::i1().to_ref()); }
        B(cx).is_not_null(Val)
    }
}

pub fn PtrDiff(cx: @mut Block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        let ccx = cx.fcx.ccx;
        if cx.unreachable { return llvm::LLVMGetUndef(ccx.int_type.to_ref()); }
        B(cx).ptrdiff(LHS, RHS)
    }
}

pub fn Trap(cx: @mut Block) {
    if cx.unreachable { return; }
    B(cx).trap();
}

pub fn LandingPad(cx: @mut Block, Ty: Type, PersFn: ValueRef,
                  NumClauses: uint) -> ValueRef {
    check_not_terminated(cx);
    assert!(!cx.unreachable);
    B(cx).landing_pad(Ty, PersFn, NumClauses)
}

pub fn SetCleanup(cx: @mut Block, LandingPad: ValueRef) {
    B(cx).set_cleanup(LandingPad)
}

pub fn Resume(cx: @mut Block, Exn: ValueRef) -> ValueRef {
    check_not_terminated(cx);
    terminate(cx, "Resume");
    B(cx).resume(Exn)
}

// Atomic Operations
pub fn AtomicCmpXchg(cx: @mut Block, dst: ValueRef,
                     cmp: ValueRef, src: ValueRef,
                     order: AtomicOrdering) -> ValueRef {
    B(cx).atomic_cmpxchg(dst, cmp, src, order)
}
pub fn AtomicRMW(cx: @mut Block, op: AtomicBinOp,
                 dst: ValueRef, src: ValueRef,
                 order: AtomicOrdering) -> ValueRef {
    B(cx).atomic_rmw(op, dst, src, order)
}
