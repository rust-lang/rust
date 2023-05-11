use super::abi::AbiBuilderMethods;
use super::asm::AsmBuilderMethods;
use super::coverageinfo::CoverageInfoBuilderMethods;
use super::debuginfo::DebugInfoBuilderMethods;
use super::intrinsic::IntrinsicCallMethods;
use super::misc::MiscMethods;
use super::type_::{ArgAbiMethods, BaseTypeMethods};
use super::{HasCodegen, StaticBuilderMethods};

use crate::common::{
    AtomicOrdering, AtomicRmwBinOp, IntPredicate, RealPredicate, SynchronizationScope, TypeKind,
};
use crate::mir::operand::OperandRef;
use crate::mir::place::PlaceRef;
use crate::MemFlags;

use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::ty::layout::{HasParamEnv, TyAndLayout};
use rustc_middle::ty::Ty;
use rustc_span::Span;
use rustc_target::abi::call::FnAbi;
use rustc_target::abi::{Abi, Align, Scalar, Size, WrappingRange};
use rustc_target::spec::HasTargetSpec;

#[derive(Copy, Clone)]
pub enum OverflowOp {
    Add,
    Sub,
    Mul,
}

pub trait BuilderMethods<'a, 'tcx>:
    HasCodegen<'tcx>
    + CoverageInfoBuilderMethods<'tcx>
    + DebugInfoBuilderMethods
    + ArgAbiMethods<'tcx>
    + AbiBuilderMethods<'tcx>
    + IntrinsicCallMethods<'tcx>
    + AsmBuilderMethods<'tcx>
    + StaticBuilderMethods
    + HasParamEnv<'tcx>
    + HasTargetSpec
{
    fn build(cx: &'a Self::CodegenCx, llbb: Self::BasicBlock) -> Self;

    fn cx(&self) -> &Self::CodegenCx;
    fn llbb(&self) -> Self::BasicBlock;

    fn set_span(&mut self, span: Span);

    // FIXME(eddyb) replace uses of this with `append_sibling_block`.
    fn append_block(cx: &'a Self::CodegenCx, llfn: Self::Function, name: &str) -> Self::BasicBlock;

    fn append_sibling_block(&mut self, name: &str) -> Self::BasicBlock;

    fn switch_to_block(&mut self, llbb: Self::BasicBlock);

    fn ret_void(&mut self);
    fn ret(&mut self, v: Self::Value);
    fn br(&mut self, dest: Self::BasicBlock);
    fn cond_br(
        &mut self,
        cond: Self::Value,
        then_llbb: Self::BasicBlock,
        else_llbb: Self::BasicBlock,
    );
    fn switch(
        &mut self,
        v: Self::Value,
        else_llbb: Self::BasicBlock,
        cases: impl ExactSizeIterator<Item = (u128, Self::BasicBlock)>,
    );
    fn invoke(
        &mut self,
        llty: Self::Type,
        fn_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        llfn: Self::Value,
        args: &[Self::Value],
        then: Self::BasicBlock,
        catch: Self::BasicBlock,
        funclet: Option<&Self::Funclet>,
    ) -> Self::Value;
    fn unreachable(&mut self);

    fn add(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fadd_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn sub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fsub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fsub_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn mul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fmul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fmul_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn udiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn exactudiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn sdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn exactsdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fdiv_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn urem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn srem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn frem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn frem_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn shl(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn lshr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn ashr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn unchecked_sadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn unchecked_uadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn unchecked_ssub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn unchecked_usub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn unchecked_smul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn unchecked_umul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn and(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn or(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn xor(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn neg(&mut self, v: Self::Value) -> Self::Value;
    fn fneg(&mut self, v: Self::Value) -> Self::Value;
    fn not(&mut self, v: Self::Value) -> Self::Value;

    fn checked_binop(
        &mut self,
        oop: OverflowOp,
        ty: Ty<'_>,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> (Self::Value, Self::Value);

    fn from_immediate(&mut self, val: Self::Value) -> Self::Value;
    fn to_immediate(&mut self, val: Self::Value, layout: TyAndLayout<'_>) -> Self::Value {
        if let Abi::Scalar(scalar) = layout.abi {
            self.to_immediate_scalar(val, scalar)
        } else {
            val
        }
    }
    fn to_immediate_scalar(&mut self, val: Self::Value, scalar: Scalar) -> Self::Value;

    fn alloca(&mut self, ty: Self::Type, align: Align) -> Self::Value;
    fn byte_array_alloca(&mut self, len: Self::Value, align: Align) -> Self::Value;

    fn load(&mut self, ty: Self::Type, ptr: Self::Value, align: Align) -> Self::Value;
    fn volatile_load(&mut self, ty: Self::Type, ptr: Self::Value) -> Self::Value;
    fn atomic_load(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        order: AtomicOrdering,
        size: Size,
    ) -> Self::Value;
    fn load_operand(&mut self, place: PlaceRef<'tcx, Self::Value>)
    -> OperandRef<'tcx, Self::Value>;

    /// Called for Rvalue::Repeat when the elem is neither a ZST nor optimizable using memset.
    fn write_operand_repeatedly(
        &mut self,
        elem: OperandRef<'tcx, Self::Value>,
        count: u64,
        dest: PlaceRef<'tcx, Self::Value>,
    );

    fn range_metadata(&mut self, load: Self::Value, range: WrappingRange);
    fn nonnull_metadata(&mut self, load: Self::Value);

    fn store(&mut self, val: Self::Value, ptr: Self::Value, align: Align) -> Self::Value;
    fn store_with_flags(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        align: Align,
        flags: MemFlags,
    ) -> Self::Value;
    fn atomic_store(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        order: AtomicOrdering,
        size: Size,
    );

    fn gep(&mut self, ty: Self::Type, ptr: Self::Value, indices: &[Self::Value]) -> Self::Value;
    fn inbounds_gep(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        indices: &[Self::Value],
    ) -> Self::Value;
    fn struct_gep(&mut self, ty: Self::Type, ptr: Self::Value, idx: u64) -> Self::Value;

    fn trunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn sext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn fptoui_sat(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn fptosi_sat(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn fptoui(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn fptosi(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn uitofp(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn sitofp(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn fptrunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn fpext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn ptrtoint(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn inttoptr(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn bitcast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn intcast(&mut self, val: Self::Value, dest_ty: Self::Type, is_signed: bool) -> Self::Value;
    fn pointercast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;

    fn cast_float_to_int(
        &mut self,
        signed: bool,
        x: Self::Value,
        dest_ty: Self::Type,
    ) -> Self::Value {
        let in_ty = self.cx().val_ty(x);
        let (float_ty, int_ty) = if self.cx().type_kind(dest_ty) == TypeKind::Vector
            && self.cx().type_kind(in_ty) == TypeKind::Vector
        {
            (self.cx().element_type(in_ty), self.cx().element_type(dest_ty))
        } else {
            (in_ty, dest_ty)
        };
        assert!(matches!(self.cx().type_kind(float_ty), TypeKind::Float | TypeKind::Double));
        assert_eq!(self.cx().type_kind(int_ty), TypeKind::Integer);

        if let Some(false) = self.cx().sess().opts.unstable_opts.saturating_float_casts {
            return if signed { self.fptosi(x, dest_ty) } else { self.fptoui(x, dest_ty) };
        }

        if signed { self.fptosi_sat(x, dest_ty) } else { self.fptoui_sat(x, dest_ty) }
    }

    fn icmp(&mut self, op: IntPredicate, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fcmp(&mut self, op: RealPredicate, lhs: Self::Value, rhs: Self::Value) -> Self::Value;

    fn memcpy(
        &mut self,
        dst: Self::Value,
        dst_align: Align,
        src: Self::Value,
        src_align: Align,
        size: Self::Value,
        flags: MemFlags,
    );
    fn memmove(
        &mut self,
        dst: Self::Value,
        dst_align: Align,
        src: Self::Value,
        src_align: Align,
        size: Self::Value,
        flags: MemFlags,
    );
    fn memset(
        &mut self,
        ptr: Self::Value,
        fill_byte: Self::Value,
        size: Self::Value,
        align: Align,
        flags: MemFlags,
    );

    fn select(
        &mut self,
        cond: Self::Value,
        then_val: Self::Value,
        else_val: Self::Value,
    ) -> Self::Value;

    fn va_arg(&mut self, list: Self::Value, ty: Self::Type) -> Self::Value;
    fn extract_element(&mut self, vec: Self::Value, idx: Self::Value) -> Self::Value;
    fn vector_splat(&mut self, num_elts: usize, elt: Self::Value) -> Self::Value;
    fn extract_value(&mut self, agg_val: Self::Value, idx: u64) -> Self::Value;
    fn insert_value(&mut self, agg_val: Self::Value, elt: Self::Value, idx: u64) -> Self::Value;

    fn set_personality_fn(&mut self, personality: Self::Value);

    // These are used by everyone except msvc
    fn cleanup_landing_pad(&mut self, pers_fn: Self::Value) -> (Self::Value, Self::Value);
    fn filter_landing_pad(&mut self, pers_fn: Self::Value) -> (Self::Value, Self::Value);
    fn resume(&mut self, exn0: Self::Value, exn1: Self::Value);

    // These are used only by msvc
    fn cleanup_pad(&mut self, parent: Option<Self::Value>, args: &[Self::Value]) -> Self::Funclet;
    fn cleanup_ret(&mut self, funclet: &Self::Funclet, unwind: Option<Self::BasicBlock>);
    fn catch_pad(&mut self, parent: Self::Value, args: &[Self::Value]) -> Self::Funclet;
    fn catch_switch(
        &mut self,
        parent: Option<Self::Value>,
        unwind: Option<Self::BasicBlock>,
        handlers: &[Self::BasicBlock],
    ) -> Self::Value;

    fn atomic_cmpxchg(
        &mut self,
        dst: Self::Value,
        cmp: Self::Value,
        src: Self::Value,
        order: AtomicOrdering,
        failure_order: AtomicOrdering,
        weak: bool,
    ) -> Self::Value;
    fn atomic_rmw(
        &mut self,
        op: AtomicRmwBinOp,
        dst: Self::Value,
        src: Self::Value,
        order: AtomicOrdering,
    ) -> Self::Value;
    fn atomic_fence(&mut self, order: AtomicOrdering, scope: SynchronizationScope);
    fn set_invariant_load(&mut self, load: Self::Value);

    /// Called for `StorageLive`
    fn lifetime_start(&mut self, ptr: Self::Value, size: Size);

    /// Called for `StorageDead`
    fn lifetime_end(&mut self, ptr: Self::Value, size: Size);

    fn instrprof_increment(
        &mut self,
        fn_name: Self::Value,
        hash: Self::Value,
        num_counters: Self::Value,
        index: Self::Value,
    );

    fn call(
        &mut self,
        llty: Self::Type,
        fn_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        llfn: Self::Value,
        args: &[Self::Value],
        funclet: Option<&Self::Funclet>,
    ) -> Self::Value;
    fn zext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;

    fn do_not_inline(&mut self, llret: Self::Value);
}
