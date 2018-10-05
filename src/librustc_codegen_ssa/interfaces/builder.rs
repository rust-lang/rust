// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use common::{IntPredicate, RealPredicate, AtomicOrdering,
    SynchronizationScope, AtomicRmwBinOp, OperandBundleDef};
use libc::c_char;
use rustc::ty::TyCtxt;
use rustc::ty::layout::{Align, Size};
use MemFlags;
use super::Backend;
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

    fn set_value_name(&mut self, value: <Self::CodegenCx as Backend<'ll>>::Value, name: &str);
    fn position_at_end(&mut self, llbb: <Self::CodegenCx as Backend<'ll>>::BasicBlock);
    fn position_at_start(&mut self, llbb: <Self::CodegenCx as Backend<'ll>>::BasicBlock);
    fn ret_void(&mut self);
    fn ret(&mut self, v: <Self::CodegenCx as Backend<'ll>>::Value);
    fn br(&mut self, dest: <Self::CodegenCx as Backend<'ll>>::BasicBlock);
    fn cond_br(
        &mut self,
        cond: <Self::CodegenCx as Backend<'ll>>::Value,
        then_llbb: <Self::CodegenCx as Backend<'ll>>::BasicBlock,
        else_llbb: <Self::CodegenCx as Backend<'ll>>::BasicBlock,
    );
    fn switch(
        &mut self,
        v: <Self::CodegenCx as Backend<'ll>>::Value,
        else_llbb: <Self::CodegenCx as Backend<'ll>>::BasicBlock,
        num_cases: usize,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn invoke(
        &mut self,
        llfn: <Self::CodegenCx as Backend<'ll>>::Value,
        args: &[<Self::CodegenCx as Backend<'ll>>::Value],
        then: <Self::CodegenCx as Backend<'ll>>::BasicBlock,
        catch: <Self::CodegenCx as Backend<'ll>>::BasicBlock,
        bundle: Option<&OperandBundleDef<'ll, <Self::CodegenCx as Backend<'ll>>::Value>>
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn unreachable(&mut self);
    fn add(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fadd(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fadd_fast(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn sub(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fsub(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fsub_fast(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn mul(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fmul(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fmul_fast(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn udiv(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn exactudiv(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn sdiv(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn exactsdiv(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fdiv(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fdiv_fast(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn urem(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn srem(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn frem(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn frem_fast(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn shl(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn lshr(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn ashr(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn and(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn or(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn xor(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn neg(
        &mut self,
        v: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fneg(
        &mut self,
        v: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn not(
        &mut self,
        v: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn alloca(
        &mut self,
        ty: <Self::CodegenCx as Backend<'ll>>::Type,
        name: &str, align: Align
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn dynamic_alloca(
        &mut self,
        ty: <Self::CodegenCx as Backend<'ll>>::Type,
        name: &str, align: Align
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn array_alloca(
        &mut self,
        ty: <Self::CodegenCx as Backend<'ll>>::Type,
        len: <Self::CodegenCx as Backend<'ll>>::Value,
        name: &str,
        align: Align
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn load(
        &mut self,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        align: Align
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn volatile_load(
        &mut self,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn atomic_load(
        &mut self,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        order: AtomicOrdering, align: Align
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn load_ref(
        &mut self,
        &PlaceRef<'tcx,<Self::CodegenCx as Backend<'ll>>::Value>
    ) -> OperandRef<'tcx, <Self::CodegenCx as Backend<'ll>>::Value>;

    fn range_metadata(&mut self, load: <Self::CodegenCx as Backend<'ll>>::Value, range: Range<u128>);
    fn nonnull_metadata(&mut self, load: <Self::CodegenCx as Backend<'ll>>::Value);

    fn store(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        align: Align
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn atomic_store(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        order: AtomicOrdering,
        align: Align
    );
    fn store_with_flags(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        align: Align,
        flags: MemFlags,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn gep(
        &mut self,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        indices: &[<Self::CodegenCx as Backend<'ll>>::Value]
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn inbounds_gep(
        &mut self,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        indices: &[<Self::CodegenCx as Backend<'ll>>::Value]
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn struct_gep(
        &mut self,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        idx: u64
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn trunc(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn sext(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fptoui(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fptosi(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn uitofp(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn sitofp(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fptrunc(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fpext(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn ptrtoint(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn inttoptr(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn bitcast(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn intcast(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type, is_signed: bool
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn pointercast(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn icmp(
        &mut self,
        op: IntPredicate,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value, rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn fcmp(
        &mut self,
        op: RealPredicate,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value, rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn empty_phi(
        &mut self,
        ty: <Self::CodegenCx as Backend<'ll>>::Type) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn phi(
        &mut self,
        ty: <Self::CodegenCx as Backend<'ll>>::Type,
        vals: &[<Self::CodegenCx as Backend<'ll>>::Value],
        bbs: &[<Self::CodegenCx as Backend<'ll>>::BasicBlock]
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn inline_asm_call(
        &mut self,
        asm: *const c_char,
        cons: *const c_char,
        inputs: &[<Self::CodegenCx as Backend<'ll>>::Value],
        output: <Self::CodegenCx as Backend<'ll>>::Type,
        volatile: bool,
        alignstack: bool,
        dia: AsmDialect
    ) -> Option<<Self::CodegenCx as Backend<'ll>>::Value>;

    fn minnum(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn maxnum(
        &mut self,
        lhs: <Self::CodegenCx as Backend<'ll>>::Value,
        rhs: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn select(
        &mut self, cond: <Self::CodegenCx as Backend<'ll>>::Value,
        then_val: <Self::CodegenCx as Backend<'ll>>::Value,
        else_val: <Self::CodegenCx as Backend<'ll>>::Value,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn va_arg(
        &mut self,
        list: <Self::CodegenCx as Backend<'ll>>::Value,
        ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn extract_element(&mut self,
        vec: <Self::CodegenCx as Backend<'ll>>::Value,
        idx: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn insert_element(
        &mut self, vec: <Self::CodegenCx as Backend<'ll>>::Value,
        elt: <Self::CodegenCx as Backend<'ll>>::Value,
        idx: <Self::CodegenCx as Backend<'ll>>::Value,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn shuffle_vector(
        &mut self,
        v1: <Self::CodegenCx as Backend<'ll>>::Value,
        v2: <Self::CodegenCx as Backend<'ll>>::Value,
        mask: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_splat(
        &mut self,
        num_elts: usize,
        elt: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_fadd_fast(
        &mut self,
        acc: <Self::CodegenCx as Backend<'ll>>::Value,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_fmul_fast(
        &mut self,
        acc: <Self::CodegenCx as Backend<'ll>>::Value,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_add(
        &mut self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_mul(
        &mut self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_and(
        &mut self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_or(
        &mut self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_xor(
        &mut self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_fmin(
        &mut self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_fmax(
        &mut self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_fmin_fast(
        &mut self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_fmax_fast(
        &mut self,
        src: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_min(
        &mut self,
        src: <Self::CodegenCx as Backend<'ll>>::Value,
        is_signed: bool
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn vector_reduce_max(
        &mut self,
        src: <Self::CodegenCx as Backend<'ll>>::Value,
        is_signed: bool
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn extract_value(
        &mut self,
        agg_val: <Self::CodegenCx as Backend<'ll>>::Value,
        idx: u64
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn insert_value(
        &mut self,
        agg_val: <Self::CodegenCx as Backend<'ll>>::Value,
        elt: <Self::CodegenCx as Backend<'ll>>::Value,
        idx: u64
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn landing_pad(
        &mut self,
        ty: <Self::CodegenCx as Backend<'ll>>::Type,
        pers_fn: <Self::CodegenCx as Backend<'ll>>::Value,
        num_clauses: usize
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn add_clause(
        &mut self,
        landing_pad: <Self::CodegenCx as Backend<'ll>>::Value,
        clause: <Self::CodegenCx as Backend<'ll>>::Value
    );
    fn set_cleanup(
        &mut self,
        landing_pad: <Self::CodegenCx as Backend<'ll>>::Value
    );
    fn resume(
        &mut self,
        exn: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn cleanup_pad(
        &mut self,
        parent: Option<<Self::CodegenCx as Backend<'ll>>::Value>,
        args: &[<Self::CodegenCx as Backend<'ll>>::Value]
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn cleanup_ret(
        &mut self, cleanup: <Self::CodegenCx as Backend<'ll>>::Value,
        unwind: Option<<Self::CodegenCx as Backend<'ll>>::BasicBlock>,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn catch_pad(
        &mut self,
        parent: <Self::CodegenCx as Backend<'ll>>::Value,
        args: &[<Self::CodegenCx as Backend<'ll>>::Value]
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn catch_ret(
        &mut self,
        pad: <Self::CodegenCx as Backend<'ll>>::Value,
        unwind: <Self::CodegenCx as Backend<'ll>>::BasicBlock
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn catch_switch(
        &mut self,
        parent: Option<<Self::CodegenCx as Backend<'ll>>::Value>,
        unwind: Option<<Self::CodegenCx as Backend<'ll>>::BasicBlock>,
        num_handlers: usize,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn add_handler(
        &mut self,
        catch_switch: <Self::CodegenCx as Backend<'ll>>::Value,
        handler: <Self::CodegenCx as Backend<'ll>>::BasicBlock
    );
    fn set_personality_fn(&mut self, personality: <Self::CodegenCx as Backend<'ll>>::Value);

    fn atomic_cmpxchg(
        &mut self,
        dst: <Self::CodegenCx as Backend<'ll>>::Value,
        cmp: <Self::CodegenCx as Backend<'ll>>::Value,
        src: <Self::CodegenCx as Backend<'ll>>::Value,
        order: AtomicOrdering,
        failure_order: AtomicOrdering,
        weak: bool,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn atomic_rmw(
        &mut self,
        op: AtomicRmwBinOp,
        dst: <Self::CodegenCx as Backend<'ll>>::Value,
        src: <Self::CodegenCx as Backend<'ll>>::Value,
        order: AtomicOrdering,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;
    fn atomic_fence(&mut self, order: AtomicOrdering, scope: SynchronizationScope);
    fn add_case(
        &mut self,
        s: <Self::CodegenCx as Backend<'ll>>::Value,
        on_val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest: <Self::CodegenCx as Backend<'ll>>::BasicBlock
    );
    fn add_incoming_to_phi(
        &mut self,
        phi: <Self::CodegenCx as Backend<'ll>>::Value,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        bb: <Self::CodegenCx as Backend<'ll>>::BasicBlock
    );
    fn set_invariant_load(&mut self, load: <Self::CodegenCx as Backend<'ll>>::Value);

    /// Returns the ptr value that should be used for storing `val`.
    fn check_store(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    /// Returns the args that should be used for a call to `llfn`.
    fn check_call<'b>(
        &mut self,
        typ: &str,
        llfn: <Self::CodegenCx as Backend<'ll>>::Value,
        args: &'b [<Self::CodegenCx as Backend<'ll>>::Value]
    ) -> Cow<'b, [<Self::CodegenCx as Backend<'ll>>::Value]>
        where [<Self::CodegenCx as Backend<'ll>>::Value] : ToOwned;

    fn lifetime_start(&mut self, ptr: <Self::CodegenCx as Backend<'ll>>::Value, size: Size);
    fn lifetime_end(&mut self, ptr: <Self::CodegenCx as Backend<'ll>>::Value, size: Size);

    /// If LLVM lifetime intrinsic support is enabled (i.e. optimizations
    /// on), and `ptr` is nonzero-sized, then extracts the size of `ptr`
    /// and the intrinsic for `lt` and passes them to `emit`, which is in
    /// charge of generating code to call the passed intrinsic on whatever
    /// block of generated code is targeted for the intrinsic.
    ///
    /// If LLVM lifetime intrinsic support is disabled (i.e.  optimizations
    /// off) or `ptr` is zero-sized, then no-op (does not call `emit`).
    fn call_lifetime_intrinsic(
        &mut self,
        intrinsic: &str,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value, size: Size
    );

    fn call(
        &mut self,
        llfn: <Self::CodegenCx as Backend<'ll>>::Value,
        args: &[<Self::CodegenCx as Backend<'ll>>::Value],
        bundle: Option<&OperandBundleDef<'ll, <Self::CodegenCx as Backend<'ll>>::Value>>
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn call_memcpy(
        &mut self,
        dst: <Self::CodegenCx as Backend<'ll>>::Value,
        src: <Self::CodegenCx as Backend<'ll>>::Value,
        n_bytes: <Self::CodegenCx as Backend<'ll>>::Value,
        align: Align,
        flags: MemFlags,
    );

    fn call_memset(
        &mut self,
        ptr: <Self::CodegenCx as Backend<'ll>>::Value,
        fill_byte: <Self::CodegenCx as Backend<'ll>>::Value,
        size: <Self::CodegenCx as Backend<'ll>>::Value,
        align: <Self::CodegenCx as Backend<'ll>>::Value,
        volatile: bool,
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn zext(
        &mut self,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dest_ty: <Self::CodegenCx as Backend<'ll>>::Type
    ) -> <Self::CodegenCx as Backend<'ll>>::Value;

    fn delete_basic_block(&mut self, bb: <Self::CodegenCx as Backend<'ll>>::BasicBlock);
    fn do_not_inline(&mut self, llret: <Self::CodegenCx as Backend<'ll>>::Value);
}
