use super::abi::AbiBuilderMethods;
use super::asm::AsmBuilderMethods;
use super::coverageinfo::CoverageInfoBuilderMethods;
use super::debuginfo::DebugInfoBuilderMethods;
use super::intrinsic::IntrinsicCallMethods;
use super::type_::ArgAbiMethods;
use super::{HasCodegen, StaticBuilderMethods};

use crate::common::{
    AtomicOrdering, AtomicRmwBinOp, IntPredicate, RealPredicate, SynchronizationScope,
};
use crate::mir::operand::OperandRef;
use crate::mir::place::PlaceRef;
use crate::MemFlags;

use rustc_middle::ty::layout::{HasParamEnv, TyAndLayout};
use rustc_middle::ty::Ty;
use rustc_span::Span;
use rustc_target::abi::call::FnAbi;
use rustc_target::abi::{Abi, Align, Scalar, Size};
use rustc_target::spec::HasTargetSpec;

use std::ops::Range;

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
    + AbiBuilderMethods
    + IntrinsicCallMethods<'tcx>
    + AsmBuilderMethods<'tcx>
    + StaticBuilderMethods
    + HasParamEnv<'tcx>
    + HasTargetSpec
{
    /// IR builder (like `Self`) that doesn't have a set (insert) position, and
    /// cannot be used until positioned (which converted it to `Self`).
    // FIXME(eddyb) maybe move this associated type to a different trait, and/or
    // provide an `UnpositionedBuilderMethods` trait for operations involving it.
    type Unpositioned;

    fn unpositioned(cx: &'a Self::CodegenCx) -> Self::Unpositioned;
    fn position_at_end(bx: Self::Unpositioned, llbb: Self::BasicBlock) -> Self;
    fn into_unpositioned(self) -> Self::Unpositioned;

    fn new_block<'b>(cx: &'a Self::CodegenCx, llfn: Self::Function, name: &'b str) -> Self;
    fn build_sibling_block(&self, name: &str) -> Self;
    fn cx(&self) -> &Self::CodegenCx;
    fn llbb(&self) -> Self::BasicBlock;
    fn set_span(&mut self, span: Span);

    // Terminator instructions (the final instruction in a block).
    // These methods take the IR builder by value and return an unpositioned one
    // (in order to make it impossible to accidentally add more instructions).

    fn ret_void(self) -> Self::Unpositioned;
    fn ret(self, v: Self::Value) -> Self::Unpositioned;
    fn br(self, dest: Self::BasicBlock) -> Self::Unpositioned;
    fn cond_br(
        self,
        cond: Self::Value,
        then_llbb: Self::BasicBlock,
        else_llbb: Self::BasicBlock,
    ) -> Self::Unpositioned;
    fn switch(
        self,
        v: Self::Value,
        else_llbb: Self::BasicBlock,
        cases: impl ExactSizeIterator<Item = (u128, Self::BasicBlock)>,
    ) -> Self::Unpositioned;
    fn unreachable(self) -> Self::Unpositioned;

    // EH (exception handling) terminator instructions.
    // Just like regular terminators, these methods transform the IR builder type,
    // but they can also return values (for various reasons).
    // FIXME(eddyb) a lot of these are LLVM-specific, redesign them.

    fn invoke(
        self,
        llfn: Self::Value,
        args: &[Self::Value],
        then: Self::BasicBlock,
        catch: Self::BasicBlock,
        funclet: Option<&Self::Funclet>,
        fn_abi_for_attrs: Option<&FnAbi<'tcx, Ty<'tcx>>>,
    ) -> (Self::Unpositioned, Self::Value);
    fn resume(self, exn: Self::Value) -> (Self::Unpositioned, Self::Value);
    fn cleanup_ret(
        self,
        funclet: &Self::Funclet,
        unwind: Option<Self::BasicBlock>,
    ) -> (Self::Unpositioned, Self::Value);
    fn catch_switch(
        self,
        parent: Option<Self::Value>,
        unwind: Option<Self::BasicBlock>,
        handlers: &[Self::BasicBlock],
    ) -> (Self::Unpositioned, Self::Value);

    // Regular (intra-block) instructions.

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
        if let Abi::Scalar(ref scalar) = layout.abi {
            self.to_immediate_scalar(val, scalar)
        } else {
            val
        }
    }
    fn to_immediate_scalar(&mut self, val: Self::Value, scalar: &Scalar) -> Self::Value;

    fn alloca(&mut self, ty: Self::Type, align: Align) -> Self::Value;
    fn dynamic_alloca(&mut self, ty: Self::Type, align: Align) -> Self::Value;
    fn array_alloca(&mut self, ty: Self::Type, len: Self::Value, align: Align) -> Self::Value;

    fn load(&mut self, ptr: Self::Value, align: Align) -> Self::Value;
    fn volatile_load(&mut self, ptr: Self::Value) -> Self::Value;
    fn atomic_load(&mut self, ptr: Self::Value, order: AtomicOrdering, size: Size) -> Self::Value;
    fn load_operand(&mut self, place: PlaceRef<'tcx, Self::Value>)
    -> OperandRef<'tcx, Self::Value>;

    /// Called for Rvalue::Repeat when the elem is neither a ZST nor optimizable using memset.
    fn write_operand_repeatedly(
        self,
        elem: OperandRef<'tcx, Self::Value>,
        count: u64,
        dest: PlaceRef<'tcx, Self::Value>,
    ) -> Self;

    fn range_metadata(&mut self, load: Self::Value, range: Range<u128>);
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

    fn gep(&mut self, ptr: Self::Value, indices: &[Self::Value]) -> Self::Value;
    fn inbounds_gep(&mut self, ptr: Self::Value, indices: &[Self::Value]) -> Self::Value;
    fn struct_gep(&mut self, ptr: Self::Value, idx: u64) -> Self::Value;

    fn trunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn sext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn fptoui_sat(&mut self, val: Self::Value, dest_ty: Self::Type) -> Option<Self::Value>;
    fn fptosi_sat(&mut self, val: Self::Value, dest_ty: Self::Type) -> Option<Self::Value>;
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

    fn landing_pad(
        &mut self,
        ty: Self::Type,
        pers_fn: Self::Value,
        num_clauses: usize,
    ) -> Self::Value;
    fn set_cleanup(&mut self, landing_pad: Self::Value);
    fn cleanup_pad(&mut self, parent: Option<Self::Value>, args: &[Self::Value]) -> Self::Funclet;
    fn catch_pad(&mut self, parent: Self::Value, args: &[Self::Value]) -> Self::Funclet;
    fn set_personality_fn(&mut self, personality: Self::Value);

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
        llfn: Self::Value,
        args: &[Self::Value],
        funclet: Option<&Self::Funclet>,
        fn_abi_for_attrs: Option<&FnAbi<'tcx, Ty<'tcx>>>,
    ) -> Self::Value;
    fn zext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;

    unsafe fn delete_basic_block(&mut self, bb: Self::BasicBlock);
    fn do_not_inline(&mut self, llret: Self::Value);
}
