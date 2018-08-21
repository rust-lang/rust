// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::{AtomicOrdering, SynchronizationScope, AsmDialect};
use common::*;
use type_::Type;
use libc::c_char;
use rustc::ty::TyCtxt;
use rustc::ty::layout::{Align, Size};
use rustc::session::Session;
use builder::MemFlags;
use value::Value;

use std::borrow::Cow;
use std::ops::Range;

pub struct OperandBundleDef<'a, Value : 'a> {
    pub name: &'a str,
    pub val: Value
}

impl OperandBundleDef<'ll, &'ll Value> {
    pub fn new(name: &'ll str, val: &'ll Value) -> Self {
        OperandBundleDef {
            name,
            val
        }
    }
}

pub enum IntPredicate {
    IntEQ,
    IntNE,
    IntUGT,
    IntUGE,
    IntULT,
    IntULE,
    IntSGT,
    IntSGE,
    IntSLT,
    IntSLE
}

#[allow(dead_code)]
pub enum RealPredicate {
    RealPredicateFalse,
    RealOEQ,
    RealOGT,
    RealOGE,
    RealOLT,
    RealOLE,
    RealONE,
    RealORD,
    RealUNO,
    RealUEQ,
    RealUGT,
    RealUGE,
    RealULT,
    RealULE,
    RealUNE,
    RealPredicateTrue
}

pub enum AtomicRmwBinOp {
    AtomicXchg,
    AtomicAdd,
    AtomicSub,
    AtomicAnd,
    AtomicNand,
    AtomicOr,
    AtomicXor,
    AtomicMax,
    AtomicMin,
    AtomicUMax,
    AtomicUMin
}

pub trait BuilderMethods<'a, 'll :'a, 'tcx: 'll,
    Value : ?Sized,
    BasicBlock: ?Sized
    > {

    fn new_block<'b>(
        cx: &'a CodegenCx<'ll, 'tcx, &'ll Value>,
        llfn: &'ll Value,
        name: &'b str
    ) -> Self;
    fn with_cx(cx: &'a CodegenCx<'ll, 'tcx, &'ll Value>) -> Self;
    fn build_sibling_block<'b>(&self, name: &'b str) -> Self;
    fn sess(&self) -> &Session;
    fn cx(&self) -> &'a CodegenCx<'ll, 'tcx, &'ll Value>;
    fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx>;
    fn llfn(&self) -> &'ll Value;
    fn llbb(&self) -> &'ll BasicBlock;
    fn count_insn(&self, category: &str);

    fn set_value_name(&self, value: &'ll Value, name: &str);
    fn position_at_end(&self, llbb: &'ll BasicBlock);
    fn position_at_start(&self, llbb: &'ll BasicBlock);
    fn ret_void(&self);
    fn ret(&self, v: &'ll Value);
    fn br(&self, dest: &'ll BasicBlock);
    fn cond_br(
        &self,
        cond: &'ll Value,
        then_llbb: &'ll BasicBlock,
        else_llbb: &'ll BasicBlock,
    );
    fn switch(
        &self,
        v: &'ll Value,
        else_llbb: &'ll BasicBlock,
        num_cases: usize,
    ) -> &'ll Value;
    fn invoke(
        &self,
        llfn: &'ll Value,
        args: &[&'ll Value],
        then: &'ll BasicBlock,
        catch: &'ll BasicBlock,
        bundle: Option<&OperandBundleDef<'ll, &'ll Value>>
    ) -> &'ll Value;
    fn unreachable(&self);
    fn add(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn fadd(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn fadd_fast(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn sub(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn fsub(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn fsub_fast(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn mul(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn fmul(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn fmul_fast(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn udiv(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn exactudiv(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn sdiv(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn exactsdiv(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn fdiv(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn fdiv_fast(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn urem(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn srem(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn frem(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn frem_fast(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn shl(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn lshr(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn ashr(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn and(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn or(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn xor(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn neg(&self, v: &'ll Value) -> &'ll Value;
    fn fneg(&self, v: &'ll Value) -> &'ll Value;
    fn not(&self, v: &'ll Value) -> &'ll Value;

    fn alloca(&self, ty: &'ll Type, name: &str, align: Align) -> &'ll Value;
    fn dynamic_alloca(&self, ty: &'ll Type, name: &str, align: Align) -> &'ll Value;
    fn array_alloca(
        &self,
        ty: &'ll Type,
        len: &'ll Value,
        name: &str,
        align: Align
    ) -> &'ll Value;

    fn load(&self, ptr: &'ll Value, align: Align) -> &'ll Value;
    fn volatile_load(&self, ptr: &'ll Value) -> &'ll Value;
    fn atomic_load(&self, ptr: &'ll Value, order: AtomicOrdering, align: Align) -> &'ll Value;

    fn range_metadata(&self, load: &'ll Value, range: Range<u128>);
    fn nonnull_metadata(&self, load: &'ll Value);

    fn store(&self, val: &'ll Value, ptr: &'ll Value, align: Align) -> &'ll Value;
    fn atomic_store(
        &self,
        val: &'ll Value,
        ptr: &'ll Value,
        order: AtomicOrdering,
        align: Align
    );
    fn store_with_flags(
        &self,
        val: &'ll Value,
        ptr: &'ll Value,
        align: Align,
        flags: MemFlags,
    ) -> &'ll Value;

    fn gep(&self, ptr: &'ll Value, indices: &[&'ll Value]) -> &'ll Value;
    fn inbounds_gep(&self, ptr: &'ll Value, indices: &[&'ll Value]) -> &'ll Value;
    fn struct_gep(&self, ptr: &'ll Value, idx: u64) -> &'ll Value;

    fn trunc(&self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value;
    fn sext(&self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value;
    fn fptoui(&self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value;
    fn fptosi(&self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value;
    fn uitofp(&self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value;
    fn sitofp(&self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value;
    fn fptrunc(&self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value;
    fn fpext(&self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value;
    fn ptrtoint(&self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value;
    fn inttoptr(&self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value;
    fn bitcast(&self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value;
    fn intcast(&self, val: &'ll Value, dest_ty: &'ll Type, is_signed: bool) -> &'ll Value;
    fn pointercast(&self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value;

    fn icmp(&self, op: IntPredicate, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn fcmp(&self, op: RealPredicate, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;

    fn empty_phi(&self, ty: &'ll Type) -> &'ll Value;
    fn phi(&self, ty: &'ll Type, vals: &[&'ll Value], bbs: &[&'ll BasicBlock]) -> &'ll Value;
    fn inline_asm_call(
        &self,
        asm: *const c_char,
        cons: *const c_char,
        inputs: &[&'ll Value],
        output: &'ll Type,
        volatile: bool,
        alignstack: bool,
        dia: AsmDialect
    ) -> &'ll Value;

    fn minnum(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn maxnum(&self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value;
    fn select(
        &self, cond: &'ll Value,
        then_val: &'ll Value,
        else_val: &'ll Value,
    ) -> &'ll Value;

    fn va_arg(&self, list: &'ll Value, ty: &'ll Type) -> &'ll Value;
    fn extract_element(&self, vec: &'ll Value, idx: &'ll Value) -> &'ll Value;
    fn insert_element(
        &self, vec: &'ll Value,
        elt: &'ll Value,
        idx: &'ll Value,
    ) -> &'ll Value;
    fn shuffle_vector(&self, v1: &'ll Value, v2: &'ll Value, mask: &'ll Value) -> &'ll Value;
    fn vector_splat(&self, num_elts: usize, elt: &'ll Value) -> &'ll Value;
    fn vector_reduce_fadd_fast(&self, acc: &'ll Value, src: &'ll Value) -> &'ll Value;
    fn vector_reduce_fmul_fast(&self, acc: &'ll Value, src: &'ll Value) -> &'ll Value;
    fn vector_reduce_add(&self, src: &'ll Value) -> &'ll Value;
    fn vector_reduce_mul(&self, src: &'ll Value) -> &'ll Value;
    fn vector_reduce_and(&self, src: &'ll Value) -> &'ll Value;
    fn vector_reduce_or(&self, src: &'ll Value) -> &'ll Value;
    fn vector_reduce_xor(&self, src: &'ll Value) -> &'ll Value;
    fn vector_reduce_fmin(&self, src: &'ll Value) -> &'ll Value;
    fn vector_reduce_fmax(&self, src: &'ll Value) -> &'ll Value;
    fn vector_reduce_fmin_fast(&self, src: &'ll Value) -> &'ll Value;
    fn vector_reduce_fmax_fast(&self, src: &'ll Value) -> &'ll Value;
    fn vector_reduce_min(&self, src: &'ll Value, is_signed: bool) -> &'ll Value;
    fn vector_reduce_max(&self, src: &'ll Value, is_signed: bool) -> &'ll Value;
    fn extract_value(&self, agg_val: &'ll Value, idx: u64) -> &'ll Value;
    fn insert_value(
        &self,
        agg_val: &'ll Value,
        elt: &'ll Value,
        idx: u64
    ) -> &'ll Value;

    fn landing_pad(
        &self,
        ty: &'ll Type,
        pers_fn: &'ll Value,
        num_clauses: usize
    ) -> &'ll Value;
    fn add_clause(&self, landing_pad: &'ll Value, clause: &'ll Value);
    fn set_cleanup(&self, landing_pad: &'ll Value);
    fn resume(&self, exn: &'ll Value) -> &'ll Value;
    fn cleanup_pad(
        &self,
        parent: Option<&'ll Value>,
        args: &[&'ll Value]
    ) -> &'ll Value;
    fn cleanup_ret(
        &self, cleanup: &'ll Value,
        unwind: Option<&'ll BasicBlock>,
    ) -> &'ll Value;
    fn catch_pad(
        &self,
        parent: &'ll Value,
        args: &[&'ll Value]
    ) -> &'ll Value;
    fn catch_ret(&self, pad: &'ll Value, unwind: &'ll BasicBlock) -> &'ll Value;
    fn catch_switch(
        &self,
        parent: Option<&'ll Value>,
        unwind: Option<&'ll BasicBlock>,
        num_handlers: usize,
    ) -> &'ll Value;
    fn add_handler(&self, catch_switch: &'ll Value, handler: &'ll BasicBlock);
    fn set_personality_fn(&self, personality: &'ll Value);

    fn atomic_cmpxchg(
        &self,
        dst: &'ll Value,
        cmp: &'ll Value,
        src: &'ll Value,
        order: AtomicOrdering,
        failure_order: AtomicOrdering,
        weak: bool,
    ) -> &'ll Value;
    fn atomic_rmw(
        &self,
        op: AtomicRmwBinOp,
        dst: &'ll Value,
        src: &'ll Value,
        order: AtomicOrdering,
    ) -> &'ll Value;
    fn atomic_fence(&self, order: AtomicOrdering, scope: SynchronizationScope);
    fn add_case(&self, s: &'ll Value, on_val: &'ll Value, dest: &'ll BasicBlock);
    fn add_incoming_to_phi(&self, phi: &'ll Value, val: &'ll Value, bb: &'ll BasicBlock);
    fn set_invariant_load(&self, load: &'ll Value);

    fn check_store(
        &self,
        val: &'ll Value,
        ptr: &'ll Value
    ) -> &'ll Value;
    fn check_call<'b>(
        &self,
        typ: &str,
        llfn: &'ll Value,
        args: &'b [&'ll Value]
    ) -> Cow<'b, [&'ll Value]>;
    fn lifetime_start(&self, ptr: &'ll Value, size: Size);
    fn lifetime_end(&self, ptr: &'ll Value, size: Size);

    fn call_lifetime_intrinsic(&self, intrinsic: &str, ptr: &'ll Value, size: Size);

    fn call(&self, llfn: &'ll Value, args: &[&'ll Value],
                bundle: Option<&OperandBundleDef<'ll, &'ll Value>>) -> &'ll Value;
    fn zext(&self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value;
}
