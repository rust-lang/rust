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
use lib::llvm::{Opcode, IntPredicate, RealPredicate, False};
use lib::llvm::{ValueRef, BasicBlockRef, BuilderRef, ModuleRef};
use lib;
use middle::trans::common::*;
use middle::trans::machine::llalign_of_min;
use syntax::codemap::span;

use middle::trans::base;
use middle::trans::type_::Type;

use std::cast;
use std::libc::{c_uint, c_ulonglong, c_char};
use std::hashmap::HashMap;
use std::str;
use std::vec;

pub fn terminate(cx: block, _: &str) {
    cx.terminated = true;
}

pub fn check_not_terminated(cx: block) {
    if cx.terminated {
        fail!("already terminated!");
    }
}

pub fn B(cx: block) -> BuilderRef {
    unsafe {
        let b = cx.fcx.ccx.builder.B;
        llvm::LLVMPositionBuilderAtEnd(b, cx.llbb);
        return b;
    }
}

pub fn count_insn(cx: block, category: &str) {
    if cx.ccx().sess.trans_stats() {
        cx.ccx().stats.n_llvm_insns += 1;
    }
    do base::with_insn_ctxt |v| {
        let h = &mut cx.ccx().stats.llvm_insns;

        // Build version of path with cycles removed.

        // Pass 1: scan table mapping str -> rightmost pos.
        let mut mm = HashMap::new();
        let len = v.len();
        let mut i = 0u;
        while i < len {
            mm.insert(v[i], i);
            i += 1u;
        }

        // Pass 2: concat strings for each elt, skipping
        // forwards over any cycles by advancing to rightmost
        // occurrence of each element in path.
        let mut s = ~".";
        i = 0u;
        while i < len {
            i = *mm.get(&v[i]);
            s.push_char('/');
            s.push_str(v[i]);
            i += 1u;
        }

        s.push_char('/');
        s.push_str(category);

        let n = match h.find(&s) {
          Some(&n) => n,
          _ => 0u
        };
        h.insert(s, n+1u);
    }
}


// The difference between a block being unreachable and being terminated is
// somewhat obscure, and has to do with error checking. When a block is
// terminated, we're saying that trying to add any further statements in the
// block is an error. On the other hand, if something is unreachable, that
// means that the block was terminated in some way that we don't want to check
// for (fail/break/return statements, call to diverging functions, etc), and
// further instructions to the block should simply be ignored.

pub fn RetVoid(cx: block) {
    unsafe {
        if cx.unreachable { return; }
        check_not_terminated(cx);
        terminate(cx, "RetVoid");
        count_insn(cx, "retvoid");
        llvm::LLVMBuildRetVoid(B(cx));
    }
}

pub fn Ret(cx: block, V: ValueRef) {
    unsafe {
        if cx.unreachable { return; }
        check_not_terminated(cx);
        terminate(cx, "Ret");
        count_insn(cx, "ret");
        llvm::LLVMBuildRet(B(cx), V);
    }
}

pub fn AggregateRet(cx: block, RetVals: &[ValueRef]) {
    if cx.unreachable { return; }
    check_not_terminated(cx);
    terminate(cx, "AggregateRet");
    unsafe {
        llvm::LLVMBuildAggregateRet(B(cx), vec::raw::to_ptr(RetVals),
                                    RetVals.len() as c_uint);
    }
}

pub fn Br(cx: block, Dest: BasicBlockRef) {
    unsafe {
        if cx.unreachable { return; }
        check_not_terminated(cx);
        terminate(cx, "Br");
        count_insn(cx, "br");
        llvm::LLVMBuildBr(B(cx), Dest);
    }
}

pub fn CondBr(cx: block, If: ValueRef, Then: BasicBlockRef,
              Else: BasicBlockRef) {
    unsafe {
        if cx.unreachable { return; }
        check_not_terminated(cx);
        terminate(cx, "CondBr");
        count_insn(cx, "condbr");
        llvm::LLVMBuildCondBr(B(cx), If, Then, Else);
    }
}

pub fn Switch(cx: block, V: ValueRef, Else: BasicBlockRef, NumCases: uint)
    -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(V); }
        check_not_terminated(cx);
        terminate(cx, "Switch");
        return llvm::LLVMBuildSwitch(B(cx), V, Else, NumCases as c_uint);
    }
}

pub fn AddCase(S: ValueRef, OnVal: ValueRef, Dest: BasicBlockRef) {
    unsafe {
        if llvm::LLVMIsUndef(S) == lib::llvm::True { return; }
        llvm::LLVMAddCase(S, OnVal, Dest);
    }
}

pub fn IndirectBr(cx: block, Addr: ValueRef, NumDests: uint) {
    unsafe {
        if cx.unreachable { return; }
        check_not_terminated(cx);
        terminate(cx, "IndirectBr");
        count_insn(cx, "indirectbr");
        llvm::LLVMBuildIndirectBr(B(cx), Addr, NumDests as c_uint);
    }
}

// This is a really awful way to get a zero-length c-string, but better (and a
// lot more efficient) than doing str::as_c_str("", ...) every time.
pub fn noname() -> *c_char {
    unsafe {
        static cnull: uint = 0u;
        return cast::transmute(&cnull);
    }
}

pub fn Invoke(cx: block,
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
    unsafe {
        count_insn(cx, "invoke");
        llvm::LLVMBuildInvoke(B(cx),
                              Fn,
                              vec::raw::to_ptr(Args),
                              Args.len() as c_uint,
                              Then,
                              Catch,
                              noname())
    }
}

pub fn FastInvoke(cx: block, Fn: ValueRef, Args: &[ValueRef],
                  Then: BasicBlockRef, Catch: BasicBlockRef) {
    if cx.unreachable { return; }
    check_not_terminated(cx);
    terminate(cx, "FastInvoke");
    unsafe {
        count_insn(cx, "fastinvoke");
        let v = llvm::LLVMBuildInvoke(B(cx), Fn, vec::raw::to_ptr(Args),
                                      Args.len() as c_uint,
                                      Then, Catch, noname());
        lib::llvm::SetInstructionCallConv(v, lib::llvm::FastCallConv);
    }
}

pub fn Unreachable(cx: block) {
    unsafe {
        if cx.unreachable { return; }
        cx.unreachable = true;
        if !cx.terminated {
            count_insn(cx, "unreachable");
            llvm::LLVMBuildUnreachable(B(cx));
        }
    }
}

pub fn _Undef(val: ValueRef) -> ValueRef {
    unsafe {
        return llvm::LLVMGetUndef(val_ty(val).to_ref());
    }
}

/* Arithmetic */
pub fn Add(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "add");
        return llvm::LLVMBuildAdd(B(cx), LHS, RHS, noname());
    }
}

pub fn NSWAdd(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "nswadd");
        return llvm::LLVMBuildNSWAdd(B(cx), LHS, RHS, noname());
    }
}

pub fn NUWAdd(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "nuwadd");
        return llvm::LLVMBuildNUWAdd(B(cx), LHS, RHS, noname());
    }
}

pub fn FAdd(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "fadd");
        return llvm::LLVMBuildFAdd(B(cx), LHS, RHS, noname());
    }
}

pub fn Sub(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "sub");
        return llvm::LLVMBuildSub(B(cx), LHS, RHS, noname());
    }
}

pub fn NSWSub(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "nwsub");
        return llvm::LLVMBuildNSWSub(B(cx), LHS, RHS, noname());
    }
}

pub fn NUWSub(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "nuwsub");
        return llvm::LLVMBuildNUWSub(B(cx), LHS, RHS, noname());
    }
}

pub fn FSub(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "sub");
        return llvm::LLVMBuildFSub(B(cx), LHS, RHS, noname());
    }
}

pub fn Mul(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "mul");
        return llvm::LLVMBuildMul(B(cx), LHS, RHS, noname());
    }
}

pub fn NSWMul(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "nswmul");
        return llvm::LLVMBuildNSWMul(B(cx), LHS, RHS, noname());
    }
}

pub fn NUWMul(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "nuwmul");
        return llvm::LLVMBuildNUWMul(B(cx), LHS, RHS, noname());
    }
}

pub fn FMul(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "fmul");
        return llvm::LLVMBuildFMul(B(cx), LHS, RHS, noname());
    }
}

pub fn UDiv(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "udiv");
        return llvm::LLVMBuildUDiv(B(cx), LHS, RHS, noname());
    }
}

pub fn SDiv(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "sdiv");
        return llvm::LLVMBuildSDiv(B(cx), LHS, RHS, noname());
    }
}

pub fn ExactSDiv(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "extractsdiv");
        return llvm::LLVMBuildExactSDiv(B(cx), LHS, RHS, noname());
    }
}

pub fn FDiv(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "fdiv");
        return llvm::LLVMBuildFDiv(B(cx), LHS, RHS, noname());
    }
}

pub fn URem(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "urem");
        return llvm::LLVMBuildURem(B(cx), LHS, RHS, noname());
    }
}

pub fn SRem(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "srem");
        return llvm::LLVMBuildSRem(B(cx), LHS, RHS, noname());
    }
}

pub fn FRem(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "frem");
        return llvm::LLVMBuildFRem(B(cx), LHS, RHS, noname());
    }
}

pub fn Shl(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "shl");
        return llvm::LLVMBuildShl(B(cx), LHS, RHS, noname());
    }
}

pub fn LShr(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "lshr");
        return llvm::LLVMBuildLShr(B(cx), LHS, RHS, noname());
    }
}

pub fn AShr(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "ashr");
        return llvm::LLVMBuildAShr(B(cx), LHS, RHS, noname());
    }
}

pub fn And(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "and");
        return llvm::LLVMBuildAnd(B(cx), LHS, RHS, noname());
    }
}

pub fn Or(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "or");
        return llvm::LLVMBuildOr(B(cx), LHS, RHS, noname());
    }
}

pub fn Xor(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "xor");
        return llvm::LLVMBuildXor(B(cx), LHS, RHS, noname());
    }
}

pub fn BinOp(cx: block, Op: Opcode, LHS: ValueRef, RHS: ValueRef)
          -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(LHS); }
        count_insn(cx, "binop");
        return llvm::LLVMBuildBinOp(B(cx), Op, LHS, RHS, noname());
    }
}

pub fn Neg(cx: block, V: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(V); }
        count_insn(cx, "neg");
        return llvm::LLVMBuildNeg(B(cx), V, noname());
    }
}

pub fn NSWNeg(cx: block, V: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(V); }
        count_insn(cx, "nswneg");
        return llvm::LLVMBuildNSWNeg(B(cx), V, noname());
    }
}

pub fn NUWNeg(cx: block, V: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(V); }
        count_insn(cx, "nuwneg");
        return llvm::LLVMBuildNUWNeg(B(cx), V, noname());
    }
}
pub fn FNeg(cx: block, V: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(V); }
        count_insn(cx, "fneg");
        return llvm::LLVMBuildFNeg(B(cx), V, noname());
    }
}

pub fn Not(cx: block, V: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(V); }
        count_insn(cx, "not");
        return llvm::LLVMBuildNot(B(cx), V, noname());
    }
}

/* Memory */
pub fn Malloc(cx: block, Ty: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::i8p().to_ref()); }
        count_insn(cx, "malloc");
        return llvm::LLVMBuildMalloc(B(cx), Ty.to_ref(), noname());
    }
}

pub fn ArrayMalloc(cx: block, Ty: Type, Val: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::i8p().to_ref()); }
        count_insn(cx, "arraymalloc");
        return llvm::LLVMBuildArrayMalloc(B(cx), Ty.to_ref(), Val, noname());
    }
}

pub fn Alloca(cx: block, Ty: Type, name: &str) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Ty.ptr_to().to_ref()); }
        count_insn(cx, "alloca");
        if name.is_empty() {
            llvm::LLVMBuildAlloca(B(cx), Ty.to_ref(), noname())
        } else {
            str::as_c_str(
                name,
                |c| llvm::LLVMBuildAlloca(B(cx), Ty.to_ref(), c))
        }
    }
}

pub fn ArrayAlloca(cx: block, Ty: Type, Val: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Ty.ptr_to().to_ref()); }
        count_insn(cx, "arrayalloca");
        return llvm::LLVMBuildArrayAlloca(B(cx), Ty.to_ref(), Val, noname());
    }
}

pub fn Free(cx: block, PointerVal: ValueRef) {
    unsafe {
        if cx.unreachable { return; }
        count_insn(cx, "free");
        llvm::LLVMBuildFree(B(cx), PointerVal);
    }
}

pub fn Load(cx: block, PointerVal: ValueRef) -> ValueRef {
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
        count_insn(cx, "load");
        return llvm::LLVMBuildLoad(B(cx), PointerVal, noname());
    }
}

pub fn AtomicLoad(cx: block, PointerVal: ValueRef, order: AtomicOrdering) -> ValueRef {
    unsafe {
        let ccx = cx.fcx.ccx;
        if cx.unreachable {
            return llvm::LLVMGetUndef(ccx.int_type.to_ref());
        }
        count_insn(cx, "load.atomic");
        let align = llalign_of_min(ccx, ccx.int_type);
        return llvm::LLVMBuildAtomicLoad(B(cx), PointerVal, noname(), order, align as c_uint);
    }
}


pub fn LoadRangeAssert(cx: block, PointerVal: ValueRef, lo: c_ulonglong,
                       hi: c_ulonglong, signed: lib::llvm::Bool) -> ValueRef {
    let value = Load(cx, PointerVal);

    if !cx.unreachable {
        unsafe {
            let t = llvm::LLVMGetElementType(llvm::LLVMTypeOf(PointerVal));
            let min = llvm::LLVMConstInt(t, lo, signed);
            let max = llvm::LLVMConstInt(t, hi, signed);

            do [min, max].as_imm_buf |ptr, len| {
                llvm::LLVMSetMetadata(value, lib::llvm::MD_range as c_uint,
                                      llvm::LLVMMDNodeInContext(cx.fcx.ccx.llcx,
                                                                ptr, len as c_uint));
            }
        }
    }

    value
}

pub fn Store(cx: block, Val: ValueRef, Ptr: ValueRef) {
    unsafe {
        if cx.unreachable { return; }
        debug!("Store %s -> %s",
               cx.val_to_str(Val),
               cx.val_to_str(Ptr));
        count_insn(cx, "store");
        llvm::LLVMBuildStore(B(cx), Val, Ptr);
    }
}

pub fn AtomicStore(cx: block, Val: ValueRef, Ptr: ValueRef, order: AtomicOrdering) {
    unsafe {
        if cx.unreachable { return; }
        debug!("Store %s -> %s",
               cx.val_to_str(Val),
               cx.val_to_str(Ptr));
        count_insn(cx, "store.atomic");
        let align = llalign_of_min(cx.ccx(), cx.ccx().int_type);
        llvm::LLVMBuildAtomicStore(B(cx), Val, Ptr, order, align as c_uint);
    }
}

pub fn GEP(cx: block, Pointer: ValueRef, Indices: &[ValueRef]) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::nil().ptr_to().to_ref()); }
        count_insn(cx, "gep");
        return llvm::LLVMBuildGEP(B(cx), Pointer, vec::raw::to_ptr(Indices),
                                   Indices.len() as c_uint, noname());
    }
}

// Simple wrapper around GEP that takes an array of ints and wraps them
// in C_i32()
#[inline]
pub fn GEPi(cx: block, base: ValueRef, ixs: &[uint]) -> ValueRef {
    // Small vector optimization. This should catch 100% of the cases that
    // we care about.
    if ixs.len() < 16 {
        let mut small_vec = [ C_i32(0), ..16 ];
        for small_vec.mut_iter().zip(ixs.iter()).advance |(small_vec_e, &ix)| {
            *small_vec_e = C_i32(ix as i32);
        }
        InBoundsGEP(cx, base, small_vec.slice(0, ixs.len()))
    } else {
        let v = do ixs.iter().transform |i| { C_i32(*i as i32) }.collect::<~[ValueRef]>();
        count_insn(cx, "gepi");
        InBoundsGEP(cx, base, v)
    }
}

pub fn InBoundsGEP(cx: block, Pointer: ValueRef, Indices: &[ValueRef]) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::nil().ptr_to().to_ref()); }
        count_insn(cx, "inboundsgep");
        return llvm::LLVMBuildInBoundsGEP(
            B(cx), Pointer, vec::raw::to_ptr(Indices), Indices.len() as c_uint, noname());
    }
}

pub fn StructGEP(cx: block, Pointer: ValueRef, Idx: uint) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::nil().ptr_to().to_ref()); }
        count_insn(cx, "structgep");
        return llvm::LLVMBuildStructGEP(B(cx),
                                        Pointer,
                                        Idx as c_uint,
                                        noname());
    }
}

pub fn GlobalString(cx: block, _Str: *c_char) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::i8p().to_ref()); }
        count_insn(cx, "globalstring");
        return llvm::LLVMBuildGlobalString(B(cx), _Str, noname());
    }
}

pub fn GlobalStringPtr(cx: block, _Str: *c_char) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::i8p().to_ref()); }
        count_insn(cx, "globalstringptr");
        return llvm::LLVMBuildGlobalStringPtr(B(cx), _Str, noname());
    }
}

/* Casts */
pub fn Trunc(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "trunc");
        return llvm::LLVMBuildTrunc(B(cx), Val, DestTy.to_ref(), noname());
    }
}

pub fn ZExt(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "zext");
        return llvm::LLVMBuildZExt(B(cx), Val, DestTy.to_ref(), noname());
    }
}

pub fn SExt(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "sext");
        return llvm::LLVMBuildSExt(B(cx), Val, DestTy.to_ref(), noname());
    }
}

pub fn FPToUI(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "fptoui");
        return llvm::LLVMBuildFPToUI(B(cx), Val, DestTy.to_ref(), noname());
    }
}

pub fn FPToSI(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "fptosi");
        return llvm::LLVMBuildFPToSI(B(cx), Val, DestTy.to_ref(),noname());
    }
}

pub fn UIToFP(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "uitofp");
        return llvm::LLVMBuildUIToFP(B(cx), Val, DestTy.to_ref(), noname());
    }
}

pub fn SIToFP(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "sitofp");
        return llvm::LLVMBuildSIToFP(B(cx), Val, DestTy.to_ref(), noname());
    }
}

pub fn FPTrunc(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "fptrunc");
        return llvm::LLVMBuildFPTrunc(B(cx), Val, DestTy.to_ref(), noname());
    }
}

pub fn FPExt(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "fpext");
        return llvm::LLVMBuildFPExt(B(cx), Val, DestTy.to_ref(), noname());
    }
}

pub fn PtrToInt(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "ptrtoint");
        return llvm::LLVMBuildPtrToInt(B(cx), Val, DestTy.to_ref(), noname());
    }
}

pub fn IntToPtr(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "inttoptr");
        return llvm::LLVMBuildIntToPtr(B(cx), Val, DestTy.to_ref(), noname());
    }
}

pub fn BitCast(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "bitcast");
        return llvm::LLVMBuildBitCast(B(cx), Val, DestTy.to_ref(), noname());
    }
}

pub fn ZExtOrBitCast(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "zextorbitcast");
        return llvm::LLVMBuildZExtOrBitCast(B(cx), Val, DestTy.to_ref(), noname());
    }
}

pub fn SExtOrBitCast(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "sextorbitcast");
        return llvm::LLVMBuildSExtOrBitCast(B(cx), Val, DestTy.to_ref(), noname());
    }
}

pub fn TruncOrBitCast(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "truncorbitcast");
        return llvm::LLVMBuildTruncOrBitCast(B(cx), Val, DestTy.to_ref(), noname());
    }
}

pub fn Cast(cx: block, Op: Opcode, Val: ValueRef, DestTy: Type, _: *u8)
     -> ValueRef {
    unsafe {
        count_insn(cx, "cast");
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        return llvm::LLVMBuildCast(B(cx), Op, Val, DestTy.to_ref(), noname());
    }
}

pub fn PointerCast(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "pointercast");
        return llvm::LLVMBuildPointerCast(B(cx), Val, DestTy.to_ref(), noname());
    }
}

pub fn IntCast(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "intcast");
        return llvm::LLVMBuildIntCast(B(cx), Val, DestTy.to_ref(), noname());
    }
}

pub fn FPCast(cx: block, Val: ValueRef, DestTy: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(DestTy.to_ref()); }
        count_insn(cx, "fpcast");
        return llvm::LLVMBuildFPCast(B(cx), Val, DestTy.to_ref(), noname());
    }
}


/* Comparisons */
pub fn ICmp(cx: block, Op: IntPredicate, LHS: ValueRef, RHS: ValueRef)
     -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::i1().to_ref()); }
        count_insn(cx, "icmp");
        return llvm::LLVMBuildICmp(B(cx), Op as c_uint, LHS, RHS, noname());
    }
}

pub fn FCmp(cx: block, Op: RealPredicate, LHS: ValueRef, RHS: ValueRef)
     -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::i1().to_ref()); }
        count_insn(cx, "fcmp");
        return llvm::LLVMBuildFCmp(B(cx), Op as c_uint, LHS, RHS, noname());
    }
}

/* Miscellaneous instructions */
pub fn EmptyPhi(cx: block, Ty: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Ty.to_ref()); }
        count_insn(cx, "emptyphi");
        return llvm::LLVMBuildPhi(B(cx), Ty.to_ref(), noname());
    }
}

pub fn Phi(cx: block, Ty: Type, vals: &[ValueRef], bbs: &[BasicBlockRef])
    -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Ty.to_ref()); }
        assert_eq!(vals.len(), bbs.len());
        let phi = EmptyPhi(cx, Ty);
        count_insn(cx, "addincoming");
        llvm::LLVMAddIncoming(phi, vec::raw::to_ptr(vals),
                              vec::raw::to_ptr(bbs),
                              vals.len() as c_uint);
        return phi;
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

pub fn _UndefReturn(cx: block, Fn: ValueRef) -> ValueRef {
    unsafe {
        let ccx = cx.fcx.ccx;
        let ty = val_ty(Fn);
        let retty = if ty.kind() == lib::llvm::Integer {
            ty.return_type()
        } else {
            ccx.int_type
        };
        count_insn(cx, "ret_undef");
        return llvm::LLVMGetUndef(retty.to_ref());
    }
}

pub fn add_span_comment(bcx: block, sp: span, text: &str) {
    let ccx = bcx.ccx();
    if ccx.sess.asm_comments() {
        let s = fmt!("%s (%s)", text, ccx.sess.codemap.span_to_str(sp));
        debug!("%s", s);
        add_comment(bcx, s);
    }
}

pub fn add_comment(bcx: block, text: &str) {
    unsafe {
        let ccx = bcx.ccx();
        if ccx.sess.asm_comments() {
            let sanitized = text.replace("$", "");
            let comment_text = ~"# " +
                sanitized.replace("\n", "\n\t# ");
            count_insn(bcx, "inlineasm");
            let asm = do comment_text.as_c_str |c| {
                llvm::LLVMConstInlineAsm(Type::func([], &Type::void()).to_ref(),
                                         c, noname(), False, False)
            };
            Call(bcx, asm, []);
        }
    }
}

pub fn InlineAsmCall(cx: block, asm: *c_char, cons: *c_char,
                     inputs: &[ValueRef], output: Type,
                     volatile: bool, alignstack: bool,
                     dia: AsmDialect) -> ValueRef {
    unsafe {
        count_insn(cx, "inlineasm");

        let volatile = if volatile { lib::llvm::True }
                       else        { lib::llvm::False };
        let alignstack = if alignstack { lib::llvm::True }
                         else          { lib::llvm::False };

        let argtys = do inputs.map |v| {
            debug!("Asm Input Type: %?", cx.val_to_str(*v));
            val_ty(*v)
        };

        debug!("Asm Output Type: %?", cx.ccx().tn.type_to_str(output));
        let fty = Type::func(argtys, &output);
        let v = llvm::LLVMInlineAsm(fty.to_ref(), asm, cons, volatile, alignstack, dia as c_uint);

        Call(cx, v, inputs)
    }
}

pub fn Call(cx: block, Fn: ValueRef, Args: &[ValueRef]) -> ValueRef {
    if cx.unreachable { return _UndefReturn(cx, Fn); }
    unsafe {
        count_insn(cx, "call");

        debug!("Call(Fn=%s, Args=%?)",
               cx.val_to_str(Fn),
               Args.map(|arg| cx.val_to_str(*arg)));

        do Args.as_imm_buf |ptr, len| {
            llvm::LLVMBuildCall(B(cx), Fn, ptr, len as c_uint, noname())
        }
    }
}

pub fn FastCall(cx: block, Fn: ValueRef, Args: &[ValueRef]) -> ValueRef {
    if cx.unreachable { return _UndefReturn(cx, Fn); }
    unsafe {
        count_insn(cx, "fastcall");
        let v = llvm::LLVMBuildCall(B(cx), Fn, vec::raw::to_ptr(Args),
                                    Args.len() as c_uint, noname());
        lib::llvm::SetInstructionCallConv(v, lib::llvm::FastCallConv);
        return v;
    }
}

pub fn CallWithConv(cx: block, Fn: ValueRef, Args: &[ValueRef],
                    Conv: CallConv) -> ValueRef {
    if cx.unreachable { return _UndefReturn(cx, Fn); }
    unsafe {
        count_insn(cx, "callwithconv");
        let v = llvm::LLVMBuildCall(B(cx), Fn, vec::raw::to_ptr(Args),
                                    Args.len() as c_uint, noname());
        lib::llvm::SetInstructionCallConv(v, Conv);
        return v;
    }
}

pub fn Select(cx: block, If: ValueRef, Then: ValueRef, Else: ValueRef) ->
   ValueRef {
    unsafe {
        if cx.unreachable { return _Undef(Then); }
        count_insn(cx, "select");
        return llvm::LLVMBuildSelect(B(cx), If, Then, Else, noname());
    }
}

pub fn VAArg(cx: block, list: ValueRef, Ty: Type) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Ty.to_ref()); }
        count_insn(cx, "vaarg");
        return llvm::LLVMBuildVAArg(B(cx), list, Ty.to_ref(), noname());
    }
}

pub fn ExtractElement(cx: block, VecVal: ValueRef, Index: ValueRef) ->
   ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::nil().to_ref()); }
        count_insn(cx, "extractelement");
        return llvm::LLVMBuildExtractElement(B(cx), VecVal, Index, noname());
    }
}

pub fn InsertElement(cx: block, VecVal: ValueRef, EltVal: ValueRef,
                     Index: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::nil().to_ref()); }
        count_insn(cx, "insertelement");
        llvm::LLVMBuildInsertElement(B(cx), VecVal, EltVal, Index, noname())
    }
}

pub fn ShuffleVector(cx: block, V1: ValueRef, V2: ValueRef,
                     Mask: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::nil().to_ref()); }
        count_insn(cx, "shufflevector");
        llvm::LLVMBuildShuffleVector(B(cx), V1, V2, Mask, noname())
    }
}

pub fn VectorSplat(cx: block, NumElts: uint, EltVal: ValueRef) -> ValueRef {
    unsafe {
        let elt_ty = val_ty(EltVal);
        let Undef = llvm::LLVMGetUndef(Type::vector(&elt_ty, NumElts as u64).to_ref());
        let VecVal = InsertElement(cx, Undef, EltVal, C_i32(0));
        ShuffleVector(cx, VecVal, Undef, C_null(Type::vector(&Type::i32(), NumElts as u64)))
    }
}

pub fn ExtractValue(cx: block, AggVal: ValueRef, Index: uint) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::nil().to_ref()); }
        count_insn(cx, "extractvalue");
        return llvm::LLVMBuildExtractValue(
            B(cx), AggVal, Index as c_uint, noname());
    }
}

pub fn InsertValue(cx: block, AggVal: ValueRef, EltVal: ValueRef,
                   Index: uint) {
    unsafe {
        if cx.unreachable { return; }
        count_insn(cx, "insertvalue");
        llvm::LLVMBuildInsertValue(B(cx), AggVal, EltVal, Index as c_uint,
                                   noname());
    }
}

pub fn IsNull(cx: block, Val: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::i1().to_ref()); }
        count_insn(cx, "isnull");
        return llvm::LLVMBuildIsNull(B(cx), Val, noname());
    }
}

pub fn IsNotNull(cx: block, Val: ValueRef) -> ValueRef {
    unsafe {
        if cx.unreachable { return llvm::LLVMGetUndef(Type::i1().to_ref()); }
        count_insn(cx, "isnotnull");
        return llvm::LLVMBuildIsNotNull(B(cx), Val, noname());
    }
}

pub fn PtrDiff(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    unsafe {
        let ccx = cx.fcx.ccx;
        if cx.unreachable { return llvm::LLVMGetUndef(ccx.int_type.to_ref()); }
        count_insn(cx, "ptrdiff");
        return llvm::LLVMBuildPtrDiff(B(cx), LHS, RHS, noname());
    }
}

pub fn Trap(cx: block) {
    unsafe {
        if cx.unreachable { return; }
        let b = B(cx);
        let BB: BasicBlockRef = llvm::LLVMGetInsertBlock(b);
        let FN: ValueRef = llvm::LLVMGetBasicBlockParent(BB);
        let M: ModuleRef = llvm::LLVMGetGlobalParent(FN);
        let T: ValueRef = str::as_c_str("llvm.trap", |buf| {
            llvm::LLVMGetNamedFunction(M, buf)
        });
        assert!((T as int != 0));
        let Args: ~[ValueRef] = ~[];
        count_insn(cx, "trap");
        llvm::LLVMBuildCall(b, T, vec::raw::to_ptr(Args), Args.len() as c_uint, noname());
    }
}

pub fn LandingPad(cx: block, Ty: Type, PersFn: ValueRef,
                  NumClauses: uint) -> ValueRef {
    unsafe {
        check_not_terminated(cx);
        assert!(!cx.unreachable);
        count_insn(cx, "landingpad");
        return llvm::LLVMBuildLandingPad(
            B(cx), Ty.to_ref(), PersFn, NumClauses as c_uint, noname());
    }
}

pub fn SetCleanup(cx: block, LandingPad: ValueRef) {
    unsafe {
        count_insn(cx, "setcleanup");
        llvm::LLVMSetCleanup(LandingPad, lib::llvm::True);
    }
}

pub fn Resume(cx: block, Exn: ValueRef) -> ValueRef {
    unsafe {
        check_not_terminated(cx);
        terminate(cx, "Resume");
        count_insn(cx, "resume");
        return llvm::LLVMBuildResume(B(cx), Exn);
    }
}

// Atomic Operations
pub fn AtomicCmpXchg(cx: block, dst: ValueRef,
                     cmp: ValueRef, src: ValueRef,
                     order: AtomicOrdering) -> ValueRef {
    unsafe {
        llvm::LLVMBuildAtomicCmpXchg(B(cx), dst, cmp, src, order)
    }
}
pub fn AtomicRMW(cx: block, op: AtomicBinOp,
                 dst: ValueRef, src: ValueRef,
                 order: AtomicOrdering) -> ValueRef {
    unsafe {
        llvm::LLVMBuildAtomicRMW(B(cx), op, dst, src, order)
    }
}
