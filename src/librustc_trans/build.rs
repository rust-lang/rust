// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)] // FFI wrappers
#![allow(non_snake_case)]

use llvm;
use llvm::{AtomicRmwBinOp, AtomicOrdering, SynchronizationScope, AsmDialect};
use llvm::{Opcode, IntPredicate, RealPredicate};
use llvm::{ValueRef, BasicBlockRef};
use common::*;
use syntax_pos::Span;

use type_::Type;
use value::Value;
use debuginfo::DebugLoc;

use libc::{c_uint, c_char};

pub fn RetVoid(cx: &BlockAndBuilder, debug_loc: DebugLoc) {
    debug_loc.apply(cx.fcx());
    cx.ret_void();
}

pub fn Ret(cx: &BlockAndBuilder, v: ValueRef, debug_loc: DebugLoc) {
    debug_loc.apply(cx.fcx());
    cx.ret(v);
}

pub fn AggregateRet(cx: &BlockAndBuilder,
    ret_vals: &[ValueRef],
    debug_loc: DebugLoc) {
    debug_loc.apply(cx.fcx());
    cx.aggregate_ret(ret_vals);
}

pub fn Br(cx: &BlockAndBuilder, dest: BasicBlockRef, debug_loc: DebugLoc) {
    debug_loc.apply(cx.fcx());
    cx.br(dest);
}

pub fn CondBr(cx: &BlockAndBuilder,
    if_: ValueRef,
    then: BasicBlockRef,
    else_: BasicBlockRef,
    debug_loc: DebugLoc) {
    debug_loc.apply(cx.fcx());
    cx.cond_br(if_, then, else_);
}

pub fn Switch(cx: &BlockAndBuilder, v: ValueRef, else_: BasicBlockRef, num_cases: usize)
    -> ValueRef {
        cx.switch(v, else_, num_cases)
    }

pub fn AddCase(s: ValueRef, on_val: ValueRef, dest: BasicBlockRef) {
    unsafe {
        if llvm::LLVMIsUndef(s) == llvm::True { return; }
        llvm::LLVMAddCase(s, on_val, dest);
    }
}

pub fn IndirectBr(cx: &BlockAndBuilder,
    addr: ValueRef,
    num_dests: usize,
    debug_loc: DebugLoc) {
    debug_loc.apply(cx.fcx());
    cx.indirect_br(addr, num_dests);
}

pub fn Invoke(cx: &BlockAndBuilder,
    fn_: ValueRef,
    args: &[ValueRef],
    then: BasicBlockRef,
    catch: BasicBlockRef,
    debug_loc: DebugLoc)
    -> ValueRef {
    debug!("Invoke({:?} with arguments ({}))",
    Value(fn_),
    args.iter().map(|a| {
        format!("{:?}", Value(*a))
    }).collect::<Vec<String>>().join(", "));
    debug_loc.apply(cx.fcx());
    let bundle = cx.lpad().and_then(|b| b.bundle());
    cx.invoke(fn_, args, then, catch, bundle)
}

/* Arithmetic */
pub fn Add(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.add(lhs, rhs)
    }

pub fn NSWAdd(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.nswadd(lhs, rhs)
    }

pub fn NUWAdd(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.nuwadd(lhs, rhs)
    }

pub fn FAdd(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.fadd(lhs, rhs)
    }

pub fn FAddFast(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.fadd_fast(lhs, rhs)
    }

pub fn Sub(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.sub(lhs, rhs)
    }

pub fn NSWSub(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.nswsub(lhs, rhs)
    }

pub fn NUWSub(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.nuwsub(lhs, rhs)
    }

pub fn FSub(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.fsub(lhs, rhs)
    }

pub fn FSubFast(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.fsub_fast(lhs, rhs)
    }

pub fn Mul(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.mul(lhs, rhs)
    }

pub fn NSWMul(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.nswmul(lhs, rhs)
    }

pub fn NUWMul(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.nuwmul(lhs, rhs)
    }

pub fn FMul(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.fmul(lhs, rhs)
    }

pub fn FMulFast(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.fmul_fast(lhs, rhs)
    }

pub fn UDiv(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.udiv(lhs, rhs)
    }

pub fn SDiv(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.sdiv(lhs, rhs)
    }

pub fn ExactSDiv(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.exactsdiv(lhs, rhs)
    }

pub fn FDiv(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.fdiv(lhs, rhs)
    }

pub fn FDivFast(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.fdiv_fast(lhs, rhs)
    }

pub fn URem(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.urem(lhs, rhs)
    }

pub fn SRem(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.srem(lhs, rhs)
    }

pub fn FRem(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.frem(lhs, rhs)
    }

pub fn FRemFast(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.frem_fast(lhs, rhs)
    }

pub fn Shl(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.shl(lhs, rhs)
    }

pub fn LShr(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.lshr(lhs, rhs)
    }

pub fn AShr(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.ashr(lhs, rhs)
    }

pub fn And(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.and(lhs, rhs)
    }

pub fn Or(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.or(lhs, rhs)
    }

pub fn Xor(cx: &BlockAndBuilder,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.xor(lhs, rhs)
    }

pub fn BinOp(cx: &BlockAndBuilder,
    op: Opcode,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.binop(op, lhs, rhs)
    }

pub fn Neg(cx: &BlockAndBuilder, v: ValueRef, debug_loc: DebugLoc) -> ValueRef {
    debug_loc.apply(cx.fcx());
    cx.neg(v)
}

pub fn NSWNeg(cx: &BlockAndBuilder, v: ValueRef, debug_loc: DebugLoc) -> ValueRef {
    debug_loc.apply(cx.fcx());
    cx.nswneg(v)
}

pub fn NUWNeg(cx: &BlockAndBuilder, v: ValueRef, debug_loc: DebugLoc) -> ValueRef {
    debug_loc.apply(cx.fcx());
    cx.nuwneg(v)
}
pub fn FNeg(cx: &BlockAndBuilder, v: ValueRef, debug_loc: DebugLoc) -> ValueRef {
    debug_loc.apply(cx.fcx());
    cx.fneg(v)
}

pub fn Not(cx: &BlockAndBuilder, v: ValueRef, debug_loc: DebugLoc) -> ValueRef {
    debug_loc.apply(cx.fcx());
    cx.not(v)
}

pub fn Alloca(cx: &BlockAndBuilder, ty: Type, name: &str) -> ValueRef {
    AllocaFcx(cx.fcx(), ty, name)
}

pub fn AllocaFcx(fcx: &FunctionContext, ty: Type, name: &str) -> ValueRef {
    let b = fcx.ccx.builder();
    b.position_before(fcx.alloca_insert_pt.get().unwrap());
    DebugLoc::None.apply(fcx);
    b.alloca(ty, name)
}

pub fn Free(cx: &BlockAndBuilder, pointer_val: ValueRef) {
    cx.free(pointer_val)
}

pub fn Load(cx: &BlockAndBuilder, pointer_val: ValueRef) -> ValueRef {
    cx.load(pointer_val)
}

pub fn VolatileLoad(cx: &BlockAndBuilder, pointer_val: ValueRef) -> ValueRef {
    cx.volatile_load(pointer_val)
}

pub fn AtomicLoad(cx: &BlockAndBuilder, pointer_val: ValueRef, order: AtomicOrdering) -> ValueRef {
    cx.atomic_load(pointer_val, order)
}


pub fn LoadRangeAssert(cx: &BlockAndBuilder, pointer_val: ValueRef, lo: u64,
    hi: u64, signed: llvm::Bool) -> ValueRef {
    cx.load_range_assert(pointer_val, lo, hi, signed)
}

pub fn LoadNonNull(cx: &BlockAndBuilder, ptr: ValueRef) -> ValueRef {
    cx.load_nonnull(ptr)
}

pub fn Store(cx: &BlockAndBuilder, val: ValueRef, ptr: ValueRef) -> ValueRef {
    cx.store(val, ptr)
}

pub fn VolatileStore(cx: &BlockAndBuilder, val: ValueRef, ptr: ValueRef) -> ValueRef {
    cx.volatile_store(val, ptr)
}

pub fn AtomicStore(cx: &BlockAndBuilder, val: ValueRef, ptr: ValueRef, order: AtomicOrdering) {
    cx.atomic_store(val, ptr, order)
}

pub fn GEP(cx: &BlockAndBuilder, pointer: ValueRef, indices: &[ValueRef]) -> ValueRef {
    cx.gep(pointer, indices)
}

// Simple wrapper around GEP that takes an array of ints and wraps them
// in C_i32()
#[inline]
pub fn GEPi(cx: &BlockAndBuilder, base: ValueRef, ixs: &[usize]) -> ValueRef {
    cx.gepi(base, ixs)
}

pub fn InBoundsGEP(cx: &BlockAndBuilder, pointer: ValueRef, indices: &[ValueRef]) -> ValueRef {
    cx.inbounds_gep(pointer, indices)
}

pub fn StructGEP(cx: &BlockAndBuilder, pointer: ValueRef, idx: usize) -> ValueRef {
    cx.struct_gep(pointer, idx)
}

pub fn GlobalString(cx: &BlockAndBuilder, _str: *const c_char) -> ValueRef {
    cx.global_string(_str)
}

pub fn GlobalStringPtr(cx: &BlockAndBuilder, _str: *const c_char) -> ValueRef {
    cx.global_string_ptr(_str)
}

/* Casts */
pub fn Trunc(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.trunc(val, dest_ty)
}

pub fn ZExt(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.zext(val, dest_ty)
}

pub fn SExt(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.sext(val, dest_ty)
}

pub fn FPToUI(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.fptoui(val, dest_ty)
}

pub fn FPToSI(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.fptosi(val, dest_ty)
}

pub fn UIToFP(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.uitofp(val, dest_ty)
}

pub fn SIToFP(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.sitofp(val, dest_ty)
}

pub fn FPTrunc(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.fptrunc(val, dest_ty)
}

pub fn FPExt(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.fpext(val, dest_ty)
}

pub fn PtrToInt(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.ptrtoint(val, dest_ty)
}

pub fn IntToPtr(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.inttoptr(val, dest_ty)
}

pub fn BitCast(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.bitcast(val, dest_ty)
}

pub fn ZExtOrBitCast(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.zext_or_bitcast(val, dest_ty)
}

pub fn SExtOrBitCast(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.sext_or_bitcast(val, dest_ty)
}

pub fn TruncOrBitCast(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.trunc_or_bitcast(val, dest_ty)
}

pub fn Cast(cx: &BlockAndBuilder, op: Opcode, val: ValueRef, dest_ty: Type,
    _: *const u8)
    -> ValueRef {
        cx.cast(op, val, dest_ty)
    }

pub fn PointerCast(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.pointercast(val, dest_ty)
}

pub fn IntCast(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.intcast(val, dest_ty)
}

pub fn FPCast(cx: &BlockAndBuilder, val: ValueRef, dest_ty: Type) -> ValueRef {
    cx.fpcast(val, dest_ty)
}


/* Comparisons */
pub fn ICmp(cx: &BlockAndBuilder,
    op: IntPredicate,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.icmp(op, lhs, rhs)
    }

pub fn FCmp(cx: &BlockAndBuilder,
    op: RealPredicate,
    lhs: ValueRef,
    rhs: ValueRef,
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        cx.fcmp(op, lhs, rhs)
    }

/* Miscellaneous instructions */
pub fn EmptyPhi(cx: &BlockAndBuilder, ty: Type) -> ValueRef {
    cx.empty_phi(ty)
}

pub fn Phi(cx: &BlockAndBuilder, ty: Type, vals: &[ValueRef], bbs: &[BasicBlockRef]) -> ValueRef {
    cx.phi(ty, vals, bbs)
}

pub fn AddIncomingToPhi(phi: ValueRef, val: ValueRef, bb: BasicBlockRef) {
    unsafe {
        if llvm::LLVMIsUndef(phi) == llvm::True { return; }
        llvm::LLVMAddIncoming(phi, &val, &bb, 1 as c_uint);
    }
}

pub fn add_span_comment(cx: &BlockAndBuilder, sp: Span, text: &str) {
    cx.add_span_comment(sp, text)
}

pub fn add_comment(cx: &BlockAndBuilder, text: &str) {
    cx.add_comment(text)
}

pub fn InlineAsmCall(cx: &BlockAndBuilder, asm: *const c_char, cons: *const c_char,
    inputs: &[ValueRef], output: Type,
    volatile: bool, alignstack: bool,
    dia: AsmDialect) -> ValueRef {
    cx.inline_asm_call(asm, cons, inputs, output, volatile, alignstack, dia)
}

pub fn Call(cx: &BlockAndBuilder,
    fn_: ValueRef,
    args: &[ValueRef],
    debug_loc: DebugLoc)
    -> ValueRef {
        debug_loc.apply(cx.fcx());
        let bundle = cx.lpad().and_then(|b| b.bundle());
        cx.call(fn_, args, bundle)
    }

pub fn AtomicFence(cx: &BlockAndBuilder, order: AtomicOrdering, scope: SynchronizationScope) {
    cx.atomic_fence(order, scope)
}

pub fn Select(cx: &BlockAndBuilder, if_: ValueRef, then: ValueRef, else_: ValueRef) -> ValueRef {
    cx.select(if_, then, else_)
}

pub fn VAArg(cx: &BlockAndBuilder, list: ValueRef, ty: Type) -> ValueRef {
    cx.va_arg(list, ty)
}

pub fn ExtractElement(cx: &BlockAndBuilder, vec_val: ValueRef, index: ValueRef) -> ValueRef {
    cx.extract_element(vec_val, index)
}

pub fn InsertElement(cx: &BlockAndBuilder, vec_val: ValueRef, elt_val: ValueRef,
    index: ValueRef) -> ValueRef {
    cx.insert_element(vec_val, elt_val, index)
}

pub fn ShuffleVector(cx: &BlockAndBuilder, v1: ValueRef, v2: ValueRef,
    mask: ValueRef) -> ValueRef {
    cx.shuffle_vector(v1, v2, mask)
}

pub fn VectorSplat(cx: &BlockAndBuilder, num_elts: usize, elt_val: ValueRef) -> ValueRef {
    cx.vector_splat(num_elts, elt_val)
}

pub fn ExtractValue(cx: &BlockAndBuilder, agg_val: ValueRef, index: usize) -> ValueRef {
    cx.extract_value(agg_val, index)
}

pub fn InsertValue(cx: &BlockAndBuilder, agg_val: ValueRef, elt_val: ValueRef, index: usize) -> ValueRef {
    cx.insert_value(agg_val, elt_val, index)
}

pub fn IsNull(cx: &BlockAndBuilder, val: ValueRef) -> ValueRef {
    cx.is_null(val)
}

pub fn IsNotNull(cx: &BlockAndBuilder, val: ValueRef) -> ValueRef {
    cx.is_not_null(val)
}

pub fn PtrDiff(cx: &BlockAndBuilder, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
    cx.ptrdiff(lhs, rhs)
}

pub fn Trap(cx: &BlockAndBuilder) {
    cx.trap();
}

pub fn LandingPad(cx: &BlockAndBuilder, ty: Type, pers_fn: ValueRef,
    num_clauses: usize) -> ValueRef {
    cx.landing_pad(ty, pers_fn, num_clauses, cx.fcx().llfn)
}

pub fn AddClause(cx: &BlockAndBuilder, landing_pad: ValueRef, clause: ValueRef) {
    cx.add_clause(landing_pad, clause)
}

pub fn SetCleanup(cx: &BlockAndBuilder, landing_pad: ValueRef) {
    cx.set_cleanup(landing_pad)
}

pub fn SetPersonalityFn(cx: &BlockAndBuilder, f: ValueRef) {
    cx.set_personality_fn(f)
}

// Atomic Operations
pub fn AtomicCmpXchg(cx: &BlockAndBuilder, dst: ValueRef,
    cmp: ValueRef, src: ValueRef,
    order: AtomicOrdering,
    failure_order: AtomicOrdering,
    weak: llvm::Bool) -> ValueRef {
    cx.atomic_cmpxchg(dst, cmp, src, order, failure_order, weak)
}
pub fn AtomicRMW(cx: &BlockAndBuilder, op: AtomicRmwBinOp,
    dst: ValueRef, src: ValueRef,
    order: AtomicOrdering) -> ValueRef {
    cx.atomic_rmw(op, dst, src, order)
}

pub fn CleanupPad(cx: &BlockAndBuilder,
    parent: Option<ValueRef>,
    args: &[ValueRef]) -> ValueRef {
    cx.cleanup_pad(parent, args)
}

pub fn CleanupRet(cx: &BlockAndBuilder,
    cleanup: ValueRef,
    unwind: Option<BasicBlockRef>) -> ValueRef {
    cx.cleanup_ret(cleanup, unwind)
}

pub fn CatchPad(cx: &BlockAndBuilder,
    parent: ValueRef,
    args: &[ValueRef]) -> ValueRef {
    cx.catch_pad(parent, args)
}

pub fn CatchRet(cx: &BlockAndBuilder, pad: ValueRef, unwind: BasicBlockRef) -> ValueRef {
    cx.catch_ret(pad, unwind)
}

pub fn CatchSwitch(cx: &BlockAndBuilder,
    parent: Option<ValueRef>,
    unwind: Option<BasicBlockRef>,
    num_handlers: usize) -> ValueRef {
    cx.catch_switch(parent, unwind, num_handlers)
}

pub fn AddHandler(cx: &BlockAndBuilder, catch_switch: ValueRef, handler: BasicBlockRef) {
    cx.add_handler(catch_switch, handler)
}
