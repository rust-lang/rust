use std::assert_matches::assert_matches;
use std::ops::Deref;

use rustc_abi::{Align, Scalar, Size, WrappingRange};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::ty::layout::{FnAbiOf, LayoutOf, TyAndLayout};
use rustc_middle::ty::{AtomicOrdering, Instance, Ty};
use rustc_session::config::OptLevel;
use rustc_span::Span;
use rustc_target::callconv::FnAbi;

use super::abi::AbiBuilderMethods;
use super::asm::AsmBuilderMethods;
use super::consts::ConstCodegenMethods;
use super::coverageinfo::CoverageInfoBuilderMethods;
use super::debuginfo::DebugInfoBuilderMethods;
use super::intrinsic::IntrinsicCallBuilderMethods;
use super::misc::MiscCodegenMethods;
use super::type_::{ArgAbiBuilderMethods, BaseTypeCodegenMethods, LayoutTypeCodegenMethods};
use super::{CodegenMethods, StaticBuilderMethods};
use crate::MemFlags;
use crate::common::{AtomicRmwBinOp, IntPredicate, RealPredicate, SynchronizationScope, TypeKind};
use crate::mir::operand::{OperandRef, OperandValue};
use crate::mir::place::{PlaceRef, PlaceValue};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OverflowOp {
    Add,
    Sub,
    Mul,
}

pub trait BuilderMethods<'a, 'tcx>:
    Sized
    + LayoutOf<'tcx, LayoutOfResult = TyAndLayout<'tcx>>
    + FnAbiOf<'tcx, FnAbiOfResult = &'tcx FnAbi<'tcx, Ty<'tcx>>>
    + Deref<Target = Self::CodegenCx>
    + CoverageInfoBuilderMethods<'tcx>
    + DebugInfoBuilderMethods
    + ArgAbiBuilderMethods<'tcx>
    + AbiBuilderMethods
    + IntrinsicCallBuilderMethods<'tcx>
    + AsmBuilderMethods<'tcx>
    + StaticBuilderMethods
{
    // `BackendTypes` is a supertrait of both `CodegenMethods` and
    // `BuilderMethods`. This bound ensures all impls agree on the associated
    // types within.
    type CodegenCx: CodegenMethods<
            'tcx,
            Value = Self::Value,
            Metadata = Self::Metadata,
            Function = Self::Function,
            BasicBlock = Self::BasicBlock,
            Type = Self::Type,
            Funclet = Self::Funclet,
            DIScope = Self::DIScope,
            DILocation = Self::DILocation,
            DIVariable = Self::DIVariable,
        >;

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

    // Conditional with expectation.
    //
    // This function is opt-in for back ends.
    //
    // The default implementation calls `self.expect()` before emiting the branch
    // by calling `self.cond_br()`
    fn cond_br_with_expect(
        &mut self,
        mut cond: Self::Value,
        then_llbb: Self::BasicBlock,
        else_llbb: Self::BasicBlock,
        expect: Option<bool>,
    ) {
        if let Some(expect) = expect {
            cond = self.expect(cond, expect);
        }
        self.cond_br(cond, then_llbb, else_llbb)
    }

    fn switch(
        &mut self,
        v: Self::Value,
        else_llbb: Self::BasicBlock,
        cases: impl ExactSizeIterator<Item = (u128, Self::BasicBlock)>,
    );

    // This is like `switch()`, but every case has a bool flag indicating whether it's cold.
    //
    // Default implementation throws away the cold flags and calls `switch()`.
    fn switch_with_weights(
        &mut self,
        v: Self::Value,
        else_llbb: Self::BasicBlock,
        _else_is_cold: bool,
        cases: impl ExactSizeIterator<Item = (u128, Self::BasicBlock, bool)>,
    ) {
        self.switch(v, else_llbb, cases.map(|(val, bb, _)| (val, bb)))
    }

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
        instance: Option<Instance<'tcx>>,
    ) -> Self::Value;
    fn unreachable(&mut self);

    fn add(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fadd_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fadd_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn sub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fsub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fsub_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fsub_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn mul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fmul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fmul_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fmul_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn udiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn exactudiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn sdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn exactsdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fdiv_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fdiv_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn urem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn srem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn frem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn frem_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn frem_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    /// Generate a left-shift. Both operands must have the same size. The right operand must be
    /// interpreted as unsigned and can be assumed to be less than the size of the left operand.
    fn shl(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    /// Generate a logical right-shift. Both operands must have the same size. The right operand
    /// must be interpreted as unsigned and can be assumed to be less than the size of the left
    /// operand.
    fn lshr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    /// Generate an arithmetic right-shift. Both operands must have the same size. The right operand
    /// must be interpreted as unsigned and can be assumed to be less than the size of the left
    /// operand.
    fn ashr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn unchecked_sadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.add(lhs, rhs)
    }
    fn unchecked_uadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.add(lhs, rhs)
    }
    fn unchecked_suadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unchecked_sadd(lhs, rhs)
    }
    fn unchecked_ssub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.sub(lhs, rhs)
    }
    fn unchecked_usub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.sub(lhs, rhs)
    }
    fn unchecked_susub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.unchecked_ssub(lhs, rhs)
    }
    fn unchecked_smul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.mul(lhs, rhs)
    }
    fn unchecked_umul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.mul(lhs, rhs)
    }
    fn unchecked_sumul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        // Which to default to is a fairly arbitrary choice,
        // but this is what slice layout was using before.
        self.unchecked_smul(lhs, rhs)
    }
    fn and(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn or(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    /// Defaults to [`Self::or`], but guarantees `(lhs & rhs) == 0` so some backends
    /// can emit something more helpful for optimizations.
    fn or_disjoint(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.or(lhs, rhs)
    }
    fn xor(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn neg(&mut self, v: Self::Value) -> Self::Value;
    fn fneg(&mut self, v: Self::Value) -> Self::Value;
    fn not(&mut self, v: Self::Value) -> Self::Value;

    fn checked_binop(
        &mut self,
        oop: OverflowOp,
        ty: Ty<'tcx>,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> (Self::Value, Self::Value);

    fn from_immediate(&mut self, val: Self::Value) -> Self::Value;
    fn to_immediate_scalar(&mut self, val: Self::Value, scalar: Scalar) -> Self::Value;

    fn alloca(&mut self, size: Size, align: Align) -> Self::Value;
    fn dynamic_alloca(&mut self, size: Self::Value, align: Align) -> Self::Value;

    fn load(&mut self, ty: Self::Type, ptr: Self::Value, align: Align) -> Self::Value;
    fn volatile_load(&mut self, ty: Self::Type, ptr: Self::Value) -> Self::Value;
    fn atomic_load(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        order: AtomicOrdering,
        size: Size,
    ) -> Self::Value;
    fn load_from_place(&mut self, ty: Self::Type, place: PlaceValue<Self::Value>) -> Self::Value {
        assert_eq!(place.llextra, None);
        self.load(ty, place.llval, place.align)
    }
    fn load_operand(&mut self, place: PlaceRef<'tcx, Self::Value>)
    -> OperandRef<'tcx, Self::Value>;

    /// Called for Rvalue::Repeat when the elem is neither a ZST nor optimizable using memset.
    fn write_operand_repeatedly(
        &mut self,
        elem: OperandRef<'tcx, Self::Value>,
        count: u64,
        dest: PlaceRef<'tcx, Self::Value>,
    );

    /// Emits an `assume` that the integer value `imm` of type `ty` is contained in `range`.
    ///
    /// This *always* emits the assumption, so you probably want to check the
    /// optimization level and `Scalar::is_always_valid` before calling it.
    fn assume_integer_range(&mut self, imm: Self::Value, ty: Self::Type, range: WrappingRange) {
        let WrappingRange { start, end } = range;

        // Perhaps one day we'll be able to use assume operand bundles for this,
        // but for now this encoding with a single icmp+assume is best per
        // <https://github.com/llvm/llvm-project/issues/123278#issuecomment-2597440158>
        let shifted = if start == 0 {
            imm
        } else {
            let low = self.const_uint_big(ty, start);
            self.sub(imm, low)
        };
        let width = self.const_uint_big(ty, u128::wrapping_sub(end, start));
        let cmp = self.icmp(IntPredicate::IntULE, shifted, width);
        self.assume(cmp);
    }

    /// Emits an `assume` that the `val` of pointer type is non-null.
    ///
    /// You may want to check the optimization level before bothering calling this.
    fn assume_nonnull(&mut self, val: Self::Value) {
        // Arguably in LLVM it'd be better to emit an assume operand bundle instead
        // <https://llvm.org/docs/LangRef.html#assume-operand-bundles>
        // but this works fine for all backends.

        let null = self.const_null(self.type_ptr());
        let is_null = self.icmp(IntPredicate::IntNE, val, null);
        self.assume(is_null);
    }

    fn range_metadata(&mut self, load: Self::Value, range: WrappingRange);
    fn nonnull_metadata(&mut self, load: Self::Value);

    fn store(&mut self, val: Self::Value, ptr: Self::Value, align: Align) -> Self::Value;
    fn store_to_place(&mut self, val: Self::Value, place: PlaceValue<Self::Value>) -> Self::Value {
        assert_eq!(place.llextra, None);
        self.store(val, place.llval, place.align)
    }
    fn store_with_flags(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        align: Align,
        flags: MemFlags,
    ) -> Self::Value;
    fn store_to_place_with_flags(
        &mut self,
        val: Self::Value,
        place: PlaceValue<Self::Value>,
        flags: MemFlags,
    ) -> Self::Value {
        assert_eq!(place.llextra, None);
        self.store_with_flags(val, place.llval, place.align, flags)
    }
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
    fn inbounds_nuw_gep(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        indices: &[Self::Value],
    ) -> Self::Value {
        self.inbounds_gep(ty, ptr, indices)
    }
    fn ptradd(&mut self, ptr: Self::Value, offset: Self::Value) -> Self::Value {
        self.gep(self.cx().type_i8(), ptr, &[offset])
    }
    fn inbounds_ptradd(&mut self, ptr: Self::Value, offset: Self::Value) -> Self::Value {
        self.inbounds_gep(self.cx().type_i8(), ptr, &[offset])
    }

    fn trunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    /// Produces the same value as [`Self::trunc`] (and defaults to that),
    /// but is UB unless the *zero*-extending the result can reproduce `val`.
    fn unchecked_utrunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.trunc(val, dest_ty)
    }
    /// Produces the same value as [`Self::trunc`] (and defaults to that),
    /// but is UB unless the *sign*-extending the result can reproduce `val`.
    fn unchecked_strunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.trunc(val, dest_ty)
    }

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
        assert_matches!(
            self.cx().type_kind(float_ty),
            TypeKind::Half | TypeKind::Float | TypeKind::Double | TypeKind::FP128
        );
        assert_eq!(self.cx().type_kind(int_ty), TypeKind::Integer);

        if let Some(false) = self.cx().sess().opts.unstable_opts.saturating_float_casts {
            return if signed { self.fptosi(x, dest_ty) } else { self.fptoui(x, dest_ty) };
        }

        if signed { self.fptosi_sat(x, dest_ty) } else { self.fptoui_sat(x, dest_ty) }
    }

    fn icmp(&mut self, op: IntPredicate, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fcmp(&mut self, op: RealPredicate, lhs: Self::Value, rhs: Self::Value) -> Self::Value;

    /// Returns `-1` if `lhs < rhs`, `0` if `lhs == rhs`, and `1` if `lhs > rhs`.
    // FIXME: Move the default implementation from `codegen_scalar_binop` into this method and
    // remove the `Option` return once LLVM 20 is the minimum version.
    fn three_way_compare(
        &mut self,
        _ty: Ty<'tcx>,
        _lhs: Self::Value,
        _rhs: Self::Value,
    ) -> Option<Self::Value> {
        None
    }

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

    /// *Typed* copy for non-overlapping places.
    ///
    /// Has a default implementation in terms of `memcpy`, but specific backends
    /// can override to do something smarter if possible.
    ///
    /// (For example, typed load-stores with alias metadata.)
    fn typed_place_copy(
        &mut self,
        dst: PlaceValue<Self::Value>,
        src: PlaceValue<Self::Value>,
        layout: TyAndLayout<'tcx>,
    ) {
        self.typed_place_copy_with_flags(dst, src, layout, MemFlags::empty());
    }

    fn typed_place_copy_with_flags(
        &mut self,
        dst: PlaceValue<Self::Value>,
        src: PlaceValue<Self::Value>,
        layout: TyAndLayout<'tcx>,
        flags: MemFlags,
    ) {
        assert!(layout.is_sized(), "cannot typed-copy an unsigned type");
        assert!(src.llextra.is_none(), "cannot directly copy from unsized values");
        assert!(dst.llextra.is_none(), "cannot directly copy into unsized values");
        if flags.contains(MemFlags::NONTEMPORAL) {
            // HACK(nox): This is inefficient but there is no nontemporal memcpy.
            let ty = self.backend_type(layout);
            let val = self.load_from_place(ty, src);
            self.store_to_place_with_flags(val, dst, flags);
        } else if self.sess().opts.optimize == OptLevel::No && self.is_backend_immediate(layout) {
            // If we're not optimizing, the aliasing information from `memcpy`
            // isn't useful, so just load-store the value for smaller code.
            let temp = self.load_operand(src.with_type(layout));
            temp.val.store_with_flags(self, dst.with_type(layout), flags);
        } else if !layout.is_zst() {
            let bytes = self.const_usize(layout.size.bytes());
            self.memcpy(dst.llval, dst.align, src.llval, src.align, bytes, flags);
        }
    }

    /// *Typed* swap for non-overlapping places.
    ///
    /// Avoids `alloca`s for Immediates and ScalarPairs.
    ///
    /// FIXME: Maybe do something smarter for Ref types too?
    /// For now, the `typed_swap_nonoverlapping` intrinsic just doesn't call this for those
    /// cases (in non-debug), preferring the fallback body instead.
    fn typed_place_swap(
        &mut self,
        left: PlaceValue<Self::Value>,
        right: PlaceValue<Self::Value>,
        layout: TyAndLayout<'tcx>,
    ) {
        let mut temp = self.load_operand(left.with_type(layout));
        if let OperandValue::Ref(..) = temp.val {
            // The SSA value isn't stand-alone, so we need to copy it elsewhere
            let alloca = PlaceRef::alloca(self, layout);
            self.typed_place_copy(alloca.val, left, layout);
            temp = self.load_operand(alloca);
        }
        self.typed_place_copy(left, right, layout);
        temp.val.store(self, right.with_type(layout));
    }

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

    fn set_personality_fn(&mut self, personality: Self::Function);

    // These are used by everyone except msvc
    fn cleanup_landing_pad(&mut self, pers_fn: Self::Function) -> (Self::Value, Self::Value);
    fn filter_landing_pad(&mut self, pers_fn: Self::Function);
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
    ) -> (Self::Value, Self::Value);
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

    fn call(
        &mut self,
        llty: Self::Type,
        fn_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        llfn: Self::Value,
        args: &[Self::Value],
        funclet: Option<&Self::Funclet>,
        instance: Option<Instance<'tcx>>,
    ) -> Self::Value;
    fn zext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;

    fn apply_attrs_to_cleanup_callsite(&mut self, llret: Self::Value);
}
