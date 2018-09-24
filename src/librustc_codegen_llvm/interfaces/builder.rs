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
use builder::MemFlags;
use super::backend::Backend;
use super::CodegenMethods;
use super::debuginfo::DebugInfoBuilderMethods;
use super::intrinsic::IntrinsicCallMethods;
use super::type_::ArgTypeMethods;
use super::abi::AbiBuilderMethods;
use super::asm::AsmBuilderMethods;
use mir::place::PlaceRef;
use mir::operand::OperandRef;

use std::borrow::Cow;
use std::ops::Range;
use syntax::ast::AsmDialect;

pub trait HasCodegen<'a, 'll: 'a, 'tcx :'ll> {
    type CodegenCx : 'a + CodegenMethods<'ll, 'tcx>;
}

pub trait BuilderMethods<'a, 'll :'a, 'tcx: 'll> : HasCodegen<'a, 'll, 'tcx> +
    DebugInfoBuilderMethods<'a, 'll, 'tcx> + ArgTypeMethods<'a, 'll, 'tcx> +
    AbiBuilderMethods<'a, 'll, 'tcx> + IntrinsicCallMethods<'a, 'll, 'tcx> +
    AsmBuilderMethods<'a, 'll, 'tcx>
{
    fn new_block<'b>(
        cx: &'a Self::CodegenCx,
        llfn: <Self::CodegenCx as Backend<'ll>>::Value,
        name: &'b str
    ) -> Self;
    fn with_cx(cx: &'a Self::CodegenCx) -> Self;
    fn build_sibling_block<'b>(&self, name: &'b str) -> Self;
    fn cx(&self) -> &'a Self::CodegenCx;
    fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx>;
    fn llfn(&self) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn llbb(&self) -> <Self::CodegenCx as Backend<'ll>>::BasicBlock;
    fn count_insn(&self, category: &str);

    fn set_value_name(&self, value: <Self::CodegenCx as Backend<'ll>>::Value, name: &str);
    fn position_at_end(&self, llbb: <Self::CodegenCx as Backend<'ll>>::BasicBlock);
    fn position_at_start(&self, llbb: <Self::CodegenCx as Backend<'ll>>::BasicBlock);
    fn ret_void(&self);
    fn ret(&self, v: <Self::CodegenCx as Backend<'ll>>::Value);
    fn br(&self, dest: <Self::CodegenCx as Backend<'ll>>::BasicBlock);
    fn cond_br(
        &self,
        cond: <Self::CodegenCx as Backend<'ll>>::Value,
        then_llbb: <Self::CodegenCx as Backend<'ll>>::BasicBlock,
        else_llbb: <Self::CodegenCx as Backend<'ll>>::BasicBlock,
    );
    fn switch(
        &self,
        v: <Self::CodegenCx as Backend<'ll>>::Value,
        else_llbb: <Self::CodegenCx as Backend<'ll>>::BasicBlock,
        num_cases: usize,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn invoke(
        &self,
        llfn: <Self::CodegenCx as Backend<'ll>>::Value,
        args: &[<Self::CodegenCx as Backend<'ll>>::Value],
        then: <Self::CodegenCx as Backend<'ll>>::BasicBlock,
        catch: <Self::CodegenCx as Backend<'ll>>::BasicBlock,
        bundle: Option<&OperandBundleDef<'ll, <Self::CodegenCx as Backend<'ll>>::Value>>
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn unreachable(&self);
    fn add(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fadd(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fadd_fast(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn sub(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fsub(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fsub_fast(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn mul(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fmul(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fmul_fast(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn udiv(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn exactudiv(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn sdiv(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn exactsdiv(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fdiv(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fdiv_fast(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn urem(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn srem(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn frem(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn frem_fast(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn shl(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn lshr(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn ashr(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn and(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn or(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn xor(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn neg(
        &self,
        v: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fneg(
        &self,
        v: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn not(
        &self,
        v: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn alloca(
        &self,
        ty: <Self::CodegenCx as Backend<'ll>>::Type,
        name: &str, align: Align
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn dynamic_alloca(
        &self,
        ty: <Self::CodegenCx as Backend<'ll>>::Type,
        name: &str, align: Align
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn array_alloca(
        &self,
        ty: <Self::CodegenCx as Backend<'ll>>::Type,
        len: <Self::CodegenCx as Backend<'ll>>::Value,
        name: &str,
        align: Align
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn load(
        &self,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        align: Align
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn volatile_load(
        &self,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn atomic_load(
        &self,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        order: AtomicOrdering, align: Align
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn load_ref(
        &self,
        &PlaceRef<'tcx,<Self::CodegenCx as Backend<'ll>>::Value>
    ) -> OperandRef<'tcx, <Self::CodegenCx as Backend<'ll>>::Value>;

    fn range_metadata(&self, load: <Self::CodegenCx as Backend<'ll>>::Value, range: Range<u128>);
    fn nonnull_metadata(&self, load: <Self::CodegenCx as Backend<'ll>>::Value);

    fn store(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        align: Align
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn atomic_store(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        order: AtomicOrdering,
        align: Align
    );
    fn store_with_flags(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        align: Align,
        flags: MemFlags,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn gep(
        &self,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        indices: &[<Self::CodegenCx as Backend<'ll>>::Value]
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn inbounds_gep(
        &self,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        indices: &[<Self::CodegenCx as Backend<'ll>>::Value]
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn struct_gep(
        &self,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        idx: u64
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn trunc(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn sext(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fptoui(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fptosi(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn uitofp(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn sitofp(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fptrunc(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fpext(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn ptrtoint(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn inttoptr(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn bitcast(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn intcast(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type, is_signed: bool
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn pointercast(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn icmp(
        &self,
        op: IntPredicate,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value, rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fcmp(
        &self,
        op: RealPredicate,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value, rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn empty_phi(
        &self,
        ty: <Self::CodegenCx as Backend<'ll>>::Type) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn phi(
        &self,
        ty: <Self::CodegenCx as Backend<'ll>>::Type,
        vals: &[<Self::CodegenCx as Backend<'ll>>::Value],
        bbs: &[<Self::CodegenCx as Backend<'ll>>::BasicBlock]
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn inline_asm_call(
        &self,
        asm: *const c_char,
        cons: *const c_char,
        inputs: &[<Self::CodegenCx as Backend<'ll>>::Value],
        output: <Self::CodegenCx as Backend<'ll>>::Type,
        volatile: bool,
        alignstack: bool,
        dia: AsmDialect
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn minnum(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn maxnum(
        &self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn select(
        &self, cond: <Self::CodegenCx as Backend<'ll>>::Value,
        then_val: <Self::CodegenCx as Backend<'ll>>::Value,
        else_val: <Self::CodegenCx as Backend<'ll>>::Value,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn va_arg(
        &self,
        list: <Self::CodegenCx as Backend<'ll>>::Value,
        ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn extract_element(&self,
        vec: <Self::CodegenCx as Backend<'ll>>::Value,
        idx: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn insert_element(
        &self, vec: <Self::CodegenCx as Backend<'ll>>::Value,
        elt: <Self::CodegenCx as Backend<'ll>>::Value,
        idx: <Self::CodegenCx as Backend<'ll>>::Value,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn shuffle_vector(
        &self,
        v1: <Self::CodegenCx as Backend<'ll>>::Value,
        v2: <Self::CodegenCx as Backend<'ll>>::Value,
        mask: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_splat(
        &self,
        num_elts: usize,
        elt: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_fadd_fast(
        &self,
        acc: <Self::CodegenCx as Backend<'ll>>::Value,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_fmul_fast(
        &self,
        acc: <Self::CodegenCx as Backend<'ll>>::Value,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_add(
        &self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_mul(
        &self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_and(
        &self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_or(
        &self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_xor(
        &self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_fmin(
        &self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_fmax(
        &self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_fmin_fast(
        &self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_fmax_fast(
        &self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_min(
        &self,
        src: <Self::CodegenCx as Backend<'ll>>::Value,
        is_signed: bool
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_max(
        &self,
        src: <Self::CodegenCx as Backend<'ll>>::Value,
        is_signed: bool
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn extract_value(
        &self,
        agg_val: <Self::CodegenCx as Backend<'ll>>::Value,
        idx: u64
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn insert_value(
        &self,
        agg_val: <Self::CodegenCx as Backend<'ll>>::Value,
        elt: <Self::CodegenCx as Backend<'ll>>::Value,
        idx: u64
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn landing_pad(
        &self,
        ty: <Self::CodegenCx as Backend<'ll>>::Type,
        pers_fn: <Self::CodegenCx as Backend<'ll>>::Value,
        num_clauses: usize
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn add_clause(
        &self,
        landing_pad: <Self::CodegenCx as Backend<'ll>>::Value,
        clause: <Self::CodegenCx as Backend<'ll>>::Value
    );
    fn set_cleanup(
        &self,
        landing_pad: <Self::CodegenCx as Backend<'ll>>::Value
    );
    fn resume(
        &self,
        exn: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn cleanup_pad(
        &self,
        parent: Option<<Self::CodegenCx as Backend<'ll>>::Value>,
        args: &[<Self::CodegenCx as Backend<'ll>>::Value]
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn cleanup_ret(
        &self, cleanup: <Self::CodegenCx as Backend<'ll>>::Value,
        unwind: Option<<Self::CodegenCx as Backend<'ll>>::BasicBlock>,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn catch_pad(
        &self,
        parent: <Self::CodegenCx as Backend<'ll>>::Value,
        args: &[<Self::CodegenCx as Backend<'ll>>::Value]
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn catch_ret(
        &self,
        pad: <Self::CodegenCx as Backend<'ll>>::Value,
        unwind: <Self::CodegenCx as Backend<'ll>>::BasicBlock
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn catch_switch(
        &self,
        parent: Option<<Self::CodegenCx as Backend<'ll>>::Value>,
        unwind: Option<<Self::CodegenCx as Backend<'ll>>::BasicBlock>,
        num_handlers: usize,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn add_handler(
        &self,
        catch_switch: <Self::CodegenCx as Backend<'ll>>::Value,
        handler: <Self::CodegenCx as Backend<'ll>>::BasicBlock
    );
    fn set_personality_fn(&self, personality: <Self::CodegenCx as Backend<'ll>>::Value);

    fn atomic_cmpxchg(
        &self,
        dst: <Self::CodegenCx as Backend<'ll>>::Value,
        cmp: <Self::CodegenCx as Backend<'ll>>::Value,
        src: <Self::CodegenCx as Backend<'ll>>::Value,
        order: AtomicOrdering,
        failure_order: AtomicOrdering,
        weak: bool,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn atomic_rmw(
        &self,
        op: AtomicRmwBinOp,
        dst: <Self::CodegenCx as Backend<'ll>>::Value,
        src: <Self::CodegenCx as Backend<'ll>>::Value,
        order: AtomicOrdering,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn atomic_fence(&self, order: AtomicOrdering, scope: SynchronizationScope);
    fn add_case(
        &self,
        s: <Self::CodegenCx as Backend<'ll>>::Value,
        on_val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest: <Self::CodegenCx as Backend<'ll>>::BasicBlock
    );
    fn add_incoming_to_phi(
        &self,
        phi: <Self::CodegenCx as Backend<'ll>>::Value,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        bb: <Self::CodegenCx as Backend<'ll>>::BasicBlock
    );
    fn set_invariant_load(&self, load: <Self::CodegenCx as Backend<'ll>>::Value);

    /// Returns the ptr value that should be used for storing `val`.
    fn check_store(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    /// Returns the args that should be used for a call to `llfn`.
    fn check_call<'b>(
        &self,
        typ: &str,
        llfn: <Self::CodegenCx as Backend<'ll>>::Value,
        args: &'b [<Self::CodegenCx as Backend<'ll>>::Value]
    ) -> Cow<'b, [<Self::CodegenCx as Backend<'ll>>::Value]>
        where [<Self::CodegenCx as Backend<'ll>>::Value] : ToOwned;

    fn lifetime_start(&self, ptr: <Self::CodegenCx as Backend<'ll>>::Value, size: Size);
    fn lifetime_end(&self, ptr: <Self::CodegenCx as Backend<'ll>>::Value, size: Size);

    /// If LLVM lifetime intrinsic support is enabled (i.e. optimizations
    /// on), and `ptr` is nonzero-sized, then extracts the size of `ptr`
    /// and the intrinsic for `lt` and passes them to `emit`, which is in
    /// charge of generating code to call the passed intrinsic on whatever
    /// block of generated code is targeted for the intrinsic.
    ///
    /// If LLVM lifetime intrinsic support is disabled (i.e.  optimizations
    /// off) or `ptr` is zero-sized, then no-op (does not call `emit`).
    fn call_lifetime_intrinsic(
        &self,
        intrinsic: &str,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value, size: Size
    );

    fn call(
        &self,
        llfn: <Self::CodegenCx as Backend<'ll>>::Value,
        args: &[<Self::CodegenCx as Backend<'ll>>::Value],
        bundle: Option<&OperandBundleDef<'ll, <Self::CodegenCx as Backend<'ll>>::Value>>
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn call_memcpy(
        &self,
        dst: <Self::CodegenCx as Backend<'ll>>::Value,
        src: <Self::CodegenCx as Backend<'ll>>::Value,
        n_bytes: <Self::CodegenCx as Backend<'ll>>::Value,
        align: Align,
        flags: MemFlags,
    );

    fn call_memset(
        &self,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        fill_byte: <Self::CodegenCx as Backend<'ll>>::Value,
        size: <Self::CodegenCx as Backend<'ll>>::Value,
        align: <Self::CodegenCx as Backend<'ll>>::Value,
        volatile: bool,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn zext(
        &self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn delete_basic_block(&self, bb: <Self::CodegenCx as Backend<'ll>>::BasicBlock);
    fn do_not_inline(&self, llret: <Self::CodegenCx as Backend<'ll>>::Value);
}
