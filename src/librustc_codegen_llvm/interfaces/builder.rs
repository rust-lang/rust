// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use common::*;
use libc::c_char;
use rustc::ty::TyCtxt;
use rustc::ty::layout::{Align, Size};
use rustc::session::Session;
use builder::MemFlags;
use super::backend::Backend;

use std::borrow::Cow;
use std::ops::Range;
use syntax::ast::AsmDialect;



pub trait BuilderMethods<'a, 'll :'a, 'tcx: 'll> : Backend {

    fn new_block<'b>(
        cx: &'a CodegenCx<'ll, 'tcx, Self::Value>,
        llfn: Self::Value,
        name: &'b str
    ) -> Self;
    fn with_cx(cx: &'a CodegenCx<'ll, 'tcx, Self::Value>) -> Self;
    fn build_sibling_block<'b>(&self, name: &'b str) -> Self;
    fn sess(&self) -> &Session;
    fn cx(&self) -> &'a CodegenCx<'ll, 'tcx, Self::Value>;
    fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx>;
    fn llfn(&self) -> Self::Value;
    fn llbb(&self) -> Self::BasicBlock;
    fn count_insn(&self, category: &str);

    fn set_value_name(&self, value: Self::Value, name: &str);
    fn position_at_end(&self, llbb: Self::BasicBlock);
    fn position_at_start(&self, llbb: Self::BasicBlock);
    fn ret_void(&self);
    fn ret(&self, v: Self::Value);
    fn br(&self, dest: Self::BasicBlock);
    fn cond_br(
        &self,
        cond: Self::Value,
        then_llbb: Self::BasicBlock,
        else_llbb: Self::BasicBlock,
    );
    fn switch(
        &self,
        v: Self::Value,
        else_llbb: Self::BasicBlock,
        num_cases: usize,
    ) -> Self::Value;
    fn invoke(
        &self,
        llfn: Self::Value,
        args: &[Self::Value],
        then: Self::BasicBlock,
        catch: Self::BasicBlock,
        bundle: Option<&OperandBundleDef<'ll, Self::Value>>
    ) -> Self::Value;
    fn unreachable(&self);
    fn add(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fadd(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fadd_fast(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn sub(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fsub(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fsub_fast(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn mul(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fmul(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fmul_fast(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn udiv(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn exactudiv(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn sdiv(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn exactsdiv(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fdiv(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fdiv_fast(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn urem(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn srem(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn frem(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn frem_fast(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn shl(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn lshr(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn ashr(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn and(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn or(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn xor(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn neg(&self, v: Self::Value) -> Self::Value;
    fn fneg(&self, v: Self::Value) -> Self::Value;
    fn not(&self, v: Self::Value) -> Self::Value;

    fn alloca(&self, ty: Self::Type, name: &str, align: Align) -> Self::Value;
    fn dynamic_alloca(&self, ty: Self::Type, name: &str, align: Align) -> Self::Value;
    fn array_alloca(
        &self,
        ty: Self::Type,
        len: Self::Value,
        name: &str,
        align: Align
    ) -> Self::Value;

    fn load(&self, ptr: Self::Value, align: Align) -> Self::Value;
    fn volatile_load(&self, ptr: Self::Value) -> Self::Value;
    fn atomic_load(&self, ptr: Self::Value, order: AtomicOrdering, align: Align) -> Self::Value;

    fn range_metadata(&self, load: Self::Value, range: Range<u128>);
    fn nonnull_metadata(&self, load: Self::Value);

    fn store(&self, val: Self::Value, ptr: Self::Value, align: Align) -> Self::Value;
    fn atomic_store(
        &self,
        val: Self::Value,
        ptr: Self::Value,
        order: AtomicOrdering,
        align: Align
    );
    fn store_with_flags(
        &self,
        val: Self::Value,
        ptr: Self::Value,
        align: Align,
        flags: MemFlags,
    ) -> Self::Value;

    fn gep(&self, ptr: Self::Value, indices: &[Self::Value]) -> Self::Value;
    fn inbounds_gep(&self, ptr: Self::Value, indices: &[Self::Value]) -> Self::Value;
    fn struct_gep(&self, ptr: Self::Value, idx: u64) -> Self::Value;

    fn trunc(&self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn sext(&self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn fptoui(&self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn fptosi(&self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn uitofp(&self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn sitofp(&self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn fptrunc(&self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn fpext(&self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn ptrtoint(&self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn inttoptr(&self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn bitcast(&self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn intcast(&self, val: Self::Value, dest_ty: Self::Type, is_signed: bool) -> Self::Value;
    fn pointercast(&self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;

    fn icmp(&self, op: IntPredicate, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fcmp(&self, op: RealPredicate, lhs: Self::Value, rhs: Self::Value) -> Self::Value;

    fn empty_phi(&self, ty: Self::Type) -> Self::Value;
    fn phi(&self, ty: Self::Type, vals: &[Self::Value], bbs: &[Self::BasicBlock]) -> Self::Value;
    fn inline_asm_call(
        &self,
        asm: *const c_char,
        cons: *const c_char,
        inputs: &[Self::Value],
        output: Self::Type,
        volatile: bool,
        alignstack: bool,
        dia: AsmDialect
    ) -> Self::Value;

    fn minnum(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn maxnum(&self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn select(
        &self, cond: Self::Value,
        then_val: Self::Value,
        else_val: Self::Value,
    ) -> Self::Value;

    fn va_arg(&self, list: Self::Value, ty: Self::Type) -> Self::Value;
    fn extract_element(&self, vec: Self::Value, idx: Self::Value) -> Self::Value;
    fn insert_element(
        &self, vec: Self::Value,
        elt: Self::Value,
        idx: Self::Value,
    ) -> Self::Value;
    fn shuffle_vector(&self, v1: Self::Value, v2: Self::Value, mask: Self::Value) -> Self::Value;
    fn vector_splat(&self, num_elts: usize, elt: Self::Value) -> Self::Value;
    fn vector_reduce_fadd_fast(&self, acc: Self::Value, src: Self::Value) -> Self::Value;
    fn vector_reduce_fmul_fast(&self, acc: Self::Value, src: Self::Value) -> Self::Value;
    fn vector_reduce_add(&self, src: Self::Value) -> Self::Value;
    fn vector_reduce_mul(&self, src: Self::Value) -> Self::Value;
    fn vector_reduce_and(&self, src: Self::Value) -> Self::Value;
    fn vector_reduce_or(&self, src: Self::Value) -> Self::Value;
    fn vector_reduce_xor(&self, src: Self::Value) -> Self::Value;
    fn vector_reduce_fmin(&self, src: Self::Value) -> Self::Value;
    fn vector_reduce_fmax(&self, src: Self::Value) -> Self::Value;
    fn vector_reduce_fmin_fast(&self, src: Self::Value) -> Self::Value;
    fn vector_reduce_fmax_fast(&self, src: Self::Value) -> Self::Value;
    fn vector_reduce_min(&self, src: Self::Value, is_signed: bool) -> Self::Value;
    fn vector_reduce_max(&self, src: Self::Value, is_signed: bool) -> Self::Value;
    fn extract_value(&self, agg_val: Self::Value, idx: u64) -> Self::Value;
    fn insert_value(
        &self,
        agg_val: Self::Value,
        elt: Self::Value,
        idx: u64
    ) -> Self::Value;

    fn landing_pad(
        &self,
        ty: Self::Type,
        pers_fn: Self::Value,
        num_clauses: usize
    ) -> Self::Value;
    fn add_clause(&self, landing_pad: Self::Value, clause: Self::Value);
    fn set_cleanup(&self, landing_pad: Self::Value);
    fn resume(&self, exn: Self::Value) -> Self::Value;
    fn cleanup_pad(
        &self,
        parent: Option<Self::Value>,
        args: &[Self::Value]
    ) -> Self::Value;
    fn cleanup_ret(
        &self, cleanup: Self::Value,
        unwind: Option<Self::BasicBlock>,
    ) -> Self::Value;
    fn catch_pad(
        &self,
        parent: Self::Value,
        args: &[Self::Value]
    ) -> Self::Value;
    fn catch_ret(&self, pad: Self::Value, unwind: Self::BasicBlock) -> Self::Value;
    fn catch_switch(
        &self,
        parent: Option<Self::Value>,
        unwind: Option<Self::BasicBlock>,
        num_handlers: usize,
    ) -> Self::Value;
    fn add_handler(&self, catch_switch: Self::Value, handler: Self::BasicBlock);
    fn set_personality_fn(&self, personality: Self::Value);

    fn atomic_cmpxchg(
        &self,
        dst: Self::Value,
        cmp: Self::Value,
        src: Self::Value,
        order: AtomicOrdering,
        failure_order: AtomicOrdering,
        weak: bool,
    ) -> Self::Value;
    fn atomic_rmw(
        &self,
        op: AtomicRmwBinOp,
        dst: Self::Value,
        src: Self::Value,
        order: AtomicOrdering,
    ) -> Self::Value;
    fn atomic_fence(&self, order: AtomicOrdering, scope: SynchronizationScope);
    fn add_case(&self, s: Self::Value, on_val: Self::Value, dest: Self::BasicBlock);
    fn add_incoming_to_phi(&self, phi: Self::Value, val: Self::Value, bb: Self::BasicBlock);
    fn set_invariant_load(&self, load: Self::Value);

    fn check_store(
        &self,
        val: Self::Value,
        ptr: Self::Value
    ) -> Self::Value;
    fn check_call<'b>(
        &self,
        typ: &str,
        llfn: Self::Value,
        args: &'b [Self::Value]
    ) -> Cow<'b, [Self::Value]> where [Self::Value] : ToOwned;
    fn lifetime_start(&self, ptr: Self::Value, size: Size);
    fn lifetime_end(&self, ptr: Self::Value, size: Size);

    fn call_lifetime_intrinsic(&self, intrinsic: &str, ptr: Self::Value, size: Size);

    fn call(&self, llfn: Self::Value, args: &[Self::Value],
                bundle: Option<&OperandBundleDef<'ll, Self::Value>>) -> Self::Value;
    fn zext(&self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
}
