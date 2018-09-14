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
use super::CodegenMethods;
use mir::place::PlaceRef;
use mir::operand::OperandRef;

use std::borrow::Cow;
use std::ops::Range;
use syntax::ast::AsmDialect;

pub trait HasCodegen<'a, 'll: 'a, 'tcx :'ll> {
    type CodegenCx : 'a + CodegenMethods<'ll, 'tcx>;
}

pub trait BuilderMethods<'a, 'll :'a, 'tcx: 'll> : HasCodegen<'a, 'll, 'tcx> {
    fn new_block<'b>(
        cx: &'a Self::CodegenCx,
        llfn: <Self::CodegenCx as Backend>::Value,
        name: &'b str
    ) -> Self;
    fn with_cx(cx: &'a Self::CodegenCx) -> Self;
    fn build_sibling_block<'b>(&self, name: &'b str) -> Self;
    fn sess(&self) -> &Session;
    fn cx(&self) -> &'a Self::CodegenCx;
    fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx>;
    fn llfn(&self) -> <Self::CodegenCx as Backend>::Value;
    fn llbb(&self) -> <Self::CodegenCx as Backend>::BasicBlock;
    fn count_insn(&self, category: &str);

    fn set_value_name(&self, value: <Self::CodegenCx as Backend>::Value, name: &str);
    fn position_at_end(&self, llbb: <Self::CodegenCx as Backend>::BasicBlock);
    fn position_at_start(&self, llbb: <Self::CodegenCx as Backend>::BasicBlock);
    fn ret_void(&self);
    fn ret(&self, v: <Self::CodegenCx as Backend>::Value);
    fn br(&self, dest: <Self::CodegenCx as Backend>::BasicBlock);
    fn cond_br(
        &self,
        cond: <Self::CodegenCx as Backend>::Value,
        then_llbb: <Self::CodegenCx as Backend>::BasicBlock,
        else_llbb: <Self::CodegenCx as Backend>::BasicBlock,
    );
    fn switch(
        &self,
        v: <Self::CodegenCx as Backend>::Value,
        else_llbb: <Self::CodegenCx as Backend>::BasicBlock,
        num_cases: usize,
    ) -> <Self::CodegenCx as Backend>::Value;
    fn invoke(
        &self,
        llfn: <Self::CodegenCx as Backend>::Value,
        args: &[<Self::CodegenCx as Backend>::Value],
        then: <Self::CodegenCx as Backend>::BasicBlock,
        catch: <Self::CodegenCx as Backend>::BasicBlock,
        bundle: Option<&OperandBundleDef<'ll, <Self::CodegenCx as Backend>::Value>>
    ) -> <Self::CodegenCx as Backend>::Value;
    fn unreachable(&self);
    fn add(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn fadd(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn fadd_fast(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn sub(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn fsub(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn fsub_fast(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn mul(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn fmul(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn fmul_fast(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn udiv(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn exactudiv(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn sdiv(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn exactsdiv(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn fdiv(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn fdiv_fast(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn urem(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn srem(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn frem(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn frem_fast(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn shl(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn lshr(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn ashr(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn and(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn or(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn xor(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn neg(&self, v: <Self::CodegenCx as Backend>::Value) -> <Self::CodegenCx as Backend>::Value;
    fn fneg(&self, v: <Self::CodegenCx as Backend>::Value) -> <Self::CodegenCx as Backend>::Value;
    fn not(&self, v: <Self::CodegenCx as Backend>::Value) -> <Self::CodegenCx as Backend>::Value;

    fn alloca(
        &self,
        ty: <Self::CodegenCx as Backend>::Type,
        name: &str, align: Align
    ) -> <Self::CodegenCx as Backend>::Value;
    fn dynamic_alloca(
        &self,
        ty: <Self::CodegenCx as Backend>::Type,
        name: &str, align: Align
    ) -> <Self::CodegenCx as Backend>::Value;
    fn array_alloca(
        &self,
        ty: <Self::CodegenCx as Backend>::Type,
        len: <Self::CodegenCx as Backend>::Value,
        name: &str,
        align: Align
    ) -> <Self::CodegenCx as Backend>::Value;

    fn load(
        &self,
        ptr: <Self::CodegenCx as Backend>::Value,
        align: Align
    ) -> <Self::CodegenCx as Backend>::Value;
    fn volatile_load(
        &self,
        ptr: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn atomic_load(
        &self,
        ptr: <Self::CodegenCx as Backend>::Value,
        order: AtomicOrdering, align: Align
    ) -> <Self::CodegenCx as Backend>::Value;
    fn load_ref(
        &self,
        &PlaceRef<'tcx,<Self::CodegenCx as Backend>::Value>
    ) -> OperandRef<'tcx, <Self::CodegenCx as Backend>::Value>;

    fn range_metadata(&self, load: <Self::CodegenCx as Backend>::Value, range: Range<u128>);
    fn nonnull_metadata(&self, load: <Self::CodegenCx as Backend>::Value);

    fn store(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        ptr: <Self::CodegenCx as Backend>::Value,
        align: Align
    ) -> <Self::CodegenCx as Backend>::Value;
    fn atomic_store(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        ptr: <Self::CodegenCx as Backend>::Value,
        order: AtomicOrdering,
        align: Align
    );
    fn store_with_flags(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        ptr: <Self::CodegenCx as Backend>::Value,
        align: Align,
        flags: MemFlags,
    ) -> <Self::CodegenCx as Backend>::Value;

    fn gep(
        &self,
        ptr: <Self::CodegenCx as Backend>::Value,
        indices: &[<Self::CodegenCx as Backend>::Value]
    ) -> <Self::CodegenCx as Backend>::Value;
    fn inbounds_gep(
        &self,
        ptr: <Self::CodegenCx as Backend>::Value,
        indices: &[<Self::CodegenCx as Backend>::Value]
    ) -> <Self::CodegenCx as Backend>::Value;
    fn struct_gep(
        &self,
        ptr: <Self::CodegenCx as Backend>::Value,
        idx: u64
    ) -> <Self::CodegenCx as Backend>::Value;

    fn trunc(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        dest_ty: <Self::CodegenCx as Backend>::Type
    ) -> <Self::CodegenCx as Backend>::Value;
    fn sext(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        dest_ty: <Self::CodegenCx as Backend>::Type
    ) -> <Self::CodegenCx as Backend>::Value;
    fn fptoui(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        dest_ty: <Self::CodegenCx as Backend>::Type
    ) -> <Self::CodegenCx as Backend>::Value;
    fn fptosi(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        dest_ty: <Self::CodegenCx as Backend>::Type
    ) -> <Self::CodegenCx as Backend>::Value;
    fn uitofp(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        dest_ty: <Self::CodegenCx as Backend>::Type
    ) -> <Self::CodegenCx as Backend>::Value;
    fn sitofp(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        dest_ty: <Self::CodegenCx as Backend>::Type
    ) -> <Self::CodegenCx as Backend>::Value;
    fn fptrunc(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        dest_ty: <Self::CodegenCx as Backend>::Type
    ) -> <Self::CodegenCx as Backend>::Value;
    fn fpext(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        dest_ty: <Self::CodegenCx as Backend>::Type
    ) -> <Self::CodegenCx as Backend>::Value;
    fn ptrtoint(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        dest_ty: <Self::CodegenCx as Backend>::Type
    ) -> <Self::CodegenCx as Backend>::Value;
    fn inttoptr(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        dest_ty: <Self::CodegenCx as Backend>::Type
    ) -> <Self::CodegenCx as Backend>::Value;
    fn bitcast(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        dest_ty: <Self::CodegenCx as Backend>::Type
    ) -> <Self::CodegenCx as Backend>::Value;
    fn intcast(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        dest_ty: <Self::CodegenCx as Backend>::Type, is_signed: bool
    ) -> <Self::CodegenCx as Backend>::Value;
    fn pointercast(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        dest_ty: <Self::CodegenCx as Backend>::Type
    ) -> <Self::CodegenCx as Backend>::Value;

    fn icmp(
        &self,
        op: IntPredicate,
        lhs: <Self::CodegenCx as Backend>::Value, rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn fcmp(
        &self,
        op: RealPredicate,
        lhs: <Self::CodegenCx as Backend>::Value, rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;

    fn empty_phi(
        &self,
        ty: <Self::CodegenCx as Backend>::Type) -> <Self::CodegenCx as Backend>::Value;
    fn phi(
        &self,
        ty: <Self::CodegenCx as Backend>::Type,
        vals: &[<Self::CodegenCx as Backend>::Value],
        bbs: &[<Self::CodegenCx as Backend>::BasicBlock]
    ) -> <Self::CodegenCx as Backend>::Value;
    fn inline_asm_call(
        &self,
        asm: *const c_char,
        cons: *const c_char,
        inputs: &[<Self::CodegenCx as Backend>::Value],
        output: <Self::CodegenCx as Backend>::Type,
        volatile: bool,
        alignstack: bool,
        dia: AsmDialect
    ) -> <Self::CodegenCx as Backend>::Value;

    fn minnum(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn maxnum(
        &self,
        lhs: <Self::CodegenCx as Backend>::Value,
        rhs: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn select(
        &self, cond: <Self::CodegenCx as Backend>::Value,
        then_val: <Self::CodegenCx as Backend>::Value,
        else_val: <Self::CodegenCx as Backend>::Value,
    ) -> <Self::CodegenCx as Backend>::Value;

    fn va_arg(
        &self,
        list: <Self::CodegenCx as Backend>::Value,
        ty: <Self::CodegenCx as Backend>::Type
    ) -> <Self::CodegenCx as Backend>::Value;
    fn extract_element(&self,
        vec: <Self::CodegenCx as Backend>::Value,
        idx: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn insert_element(
        &self, vec: <Self::CodegenCx as Backend>::Value,
        elt: <Self::CodegenCx as Backend>::Value,
        idx: <Self::CodegenCx as Backend>::Value,
    ) -> <Self::CodegenCx as Backend>::Value;
    fn shuffle_vector(
        &self,
        v1: <Self::CodegenCx as Backend>::Value,
        v2: <Self::CodegenCx as Backend>::Value,
        mask: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn vector_splat(
        &self,
        num_elts: usize,
        elt: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn vector_reduce_fadd_fast(
        &self,
        acc: <Self::CodegenCx as Backend>::Value,
        src: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn vector_reduce_fmul_fast(
        &self,
        acc: <Self::CodegenCx as Backend>::Value,
        src: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn vector_reduce_add(
        &self,
        src: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn vector_reduce_mul(
        &self,
        src: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn vector_reduce_and(
        &self,
        src: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn vector_reduce_or(
        &self,
        src: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn vector_reduce_xor(
        &self,
        src: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn vector_reduce_fmin(
        &self,
        src: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn vector_reduce_fmax(
        &self,
        src: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn vector_reduce_fmin_fast(
        &self,
        src: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn vector_reduce_fmax_fast(
        &self,
        src: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn vector_reduce_min(
        &self,
        src: <Self::CodegenCx as Backend>::Value,
        is_signed: bool
    ) -> <Self::CodegenCx as Backend>::Value;
    fn vector_reduce_max(
        &self,
        src: <Self::CodegenCx as Backend>::Value,
        is_signed: bool
    ) -> <Self::CodegenCx as Backend>::Value;
    fn extract_value(
        &self,
        agg_val: <Self::CodegenCx as Backend>::Value,
        idx: u64
    ) -> <Self::CodegenCx as Backend>::Value;
    fn insert_value(
        &self,
        agg_val: <Self::CodegenCx as Backend>::Value,
        elt: <Self::CodegenCx as Backend>::Value,
        idx: u64
    ) -> <Self::CodegenCx as Backend>::Value;

    fn landing_pad(
        &self,
        ty: <Self::CodegenCx as Backend>::Type,
        pers_fn: <Self::CodegenCx as Backend>::Value,
        num_clauses: usize
    ) -> <Self::CodegenCx as Backend>::Value;
    fn add_clause(
        &self,
        landing_pad: <Self::CodegenCx as Backend>::Value,
        clause: <Self::CodegenCx as Backend>::Value
    );
    fn set_cleanup(
        &self,
        landing_pad: <Self::CodegenCx as Backend>::Value
    );
    fn resume(
        &self,
        exn: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn cleanup_pad(
        &self,
        parent: Option<<Self::CodegenCx as Backend>::Value>,
        args: &[<Self::CodegenCx as Backend>::Value]
    ) -> <Self::CodegenCx as Backend>::Value;
    fn cleanup_ret(
        &self, cleanup: <Self::CodegenCx as Backend>::Value,
        unwind: Option<<Self::CodegenCx as Backend>::BasicBlock>,
    ) -> <Self::CodegenCx as Backend>::Value;
    fn catch_pad(
        &self,
        parent: <Self::CodegenCx as Backend>::Value,
        args: &[<Self::CodegenCx as Backend>::Value]
    ) -> <Self::CodegenCx as Backend>::Value;
    fn catch_ret(
        &self,
        pad: <Self::CodegenCx as Backend>::Value,
        unwind: <Self::CodegenCx as Backend>::BasicBlock
    ) -> <Self::CodegenCx as Backend>::Value;
    fn catch_switch(
        &self,
        parent: Option<<Self::CodegenCx as Backend>::Value>,
        unwind: Option<<Self::CodegenCx as Backend>::BasicBlock>,
        num_handlers: usize,
    ) -> <Self::CodegenCx as Backend>::Value;
    fn add_handler(
        &self,
        catch_switch: <Self::CodegenCx as Backend>::Value,
        handler: <Self::CodegenCx as Backend>::BasicBlock
    );
    fn set_personality_fn(&self, personality: <Self::CodegenCx as Backend>::Value);

    fn atomic_cmpxchg(
        &self,
        dst: <Self::CodegenCx as Backend>::Value,
        cmp: <Self::CodegenCx as Backend>::Value,
        src: <Self::CodegenCx as Backend>::Value,
        order: AtomicOrdering,
        failure_order: AtomicOrdering,
        weak: bool,
    ) -> <Self::CodegenCx as Backend>::Value;
    fn atomic_rmw(
        &self,
        op: AtomicRmwBinOp,
        dst: <Self::CodegenCx as Backend>::Value,
        src: <Self::CodegenCx as Backend>::Value,
        order: AtomicOrdering,
    ) -> <Self::CodegenCx as Backend>::Value;
    fn atomic_fence(&self, order: AtomicOrdering, scope: SynchronizationScope);
    fn add_case(
        &self,
        s: <Self::CodegenCx as Backend>::Value,
        on_val: <Self::CodegenCx as Backend>::Value,
        dest: <Self::CodegenCx as Backend>::BasicBlock
    );
    fn add_incoming_to_phi(
        &self,
        phi: <Self::CodegenCx as Backend>::Value,
        val: <Self::CodegenCx as Backend>::Value,
        bb: <Self::CodegenCx as Backend>::BasicBlock
    );
    fn set_invariant_load(&self, load: <Self::CodegenCx as Backend>::Value);

    fn check_store(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        ptr: <Self::CodegenCx as Backend>::Value
    ) -> <Self::CodegenCx as Backend>::Value;
    fn check_call<'b>(
        &self,
        typ: &str,
        llfn: <Self::CodegenCx as Backend>::Value,
        args: &'b [<Self::CodegenCx as Backend>::Value]
    ) -> Cow<'b, [<Self::CodegenCx as Backend>::Value]>
        where [<Self::CodegenCx as Backend>::Value] : ToOwned;
    fn lifetime_start(&self, ptr: <Self::CodegenCx as Backend>::Value, size: Size);
    fn lifetime_end(&self, ptr: <Self::CodegenCx as Backend>::Value, size: Size);

    fn call_lifetime_intrinsic(
        &self,
        intrinsic: &str,
        ptr: <Self::CodegenCx as Backend>::Value, size: Size
    );

    fn call(
        &self,
        llfn: <Self::CodegenCx as Backend>::Value,
        args: &[<Self::CodegenCx as Backend>::Value],
        bundle: Option<&OperandBundleDef<'ll, <Self::CodegenCx as Backend>::Value>>
    ) -> <Self::CodegenCx as Backend>::Value;

    fn call_memcpy(
        &self,
        dst: <Self::CodegenCx as Backend>::Value,
        src: <Self::CodegenCx as Backend>::Value,
        n_bytes: <Self::CodegenCx as Backend>::Value,
        align: Align,
        flags: MemFlags,
    );

    fn call_memset(
        &self,
        ptr: <Self::CodegenCx as Backend>::Value,
        fill_byte: <Self::CodegenCx as Backend>::Value,
        size: <Self::CodegenCx as Backend>::Value,
        align: <Self::CodegenCx as Backend>::Value,
        volatile: bool,
    ) -> <Self::CodegenCx as Backend>::Value;

    fn zext(
        &self,
        val: <Self::CodegenCx as Backend>::Value,
        dest_ty: <Self::CodegenCx as Backend>::Type
    ) -> <Self::CodegenCx as Backend>::Value;
}
