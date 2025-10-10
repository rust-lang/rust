//! A lint that checks for known panics like overflows, division by zero,
//! out-of-bound access etc. Uses const propagation to determine the values of
//! operands during checks.

use std::fmt::Debug;

use rustc_abi::{BackendRepr, FieldIdx, HasDataLayout, Size, TargetDataLayout, VariantIdx};
use rustc_const_eval::const_eval::DummyMachine;
use rustc_const_eval::interpret::{
    ImmTy, InterpCx, InterpResult, Projectable, Scalar, format_interp_error, interp_ok,
};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::HirId;
use rustc_hir::def::DefKind;
use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::bug;
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::layout::{LayoutError, LayoutOf, LayoutOfHelpers, TyAndLayout};
use rustc_middle::ty::{self, ConstInt, ScalarInt, Ty, TyCtxt, TypeVisitableExt};
use rustc_span::Span;
use tracing::{debug, instrument, trace};

use crate::errors::{AssertLint, AssertLintKind};

pub(super) struct KnownPanicsLint;

impl<'tcx> crate::MirLint<'tcx> for KnownPanicsLint {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        if body.tainted_by_errors.is_some() {
            return;
        }

        let def_id = body.source.def_id().expect_local();
        let def_kind = tcx.def_kind(def_id);
        let is_fn_like = def_kind.is_fn_like();
        let is_assoc_const = def_kind == DefKind::AssocConst;

        // Only run const prop on functions, methods, closures and associated constants
        if !is_fn_like && !is_assoc_const {
            // skip anon_const/statics/consts because they'll be evaluated by miri anyway
            trace!("KnownPanicsLint skipped for {:?}", def_id);
            return;
        }

        // FIXME(welseywiser) const prop doesn't work on coroutines because of query cycles
        // computing their layout.
        if tcx.is_coroutine(def_id.to_def_id()) {
            trace!("KnownPanicsLint skipped for coroutine {:?}", def_id);
            return;
        }

        trace!("KnownPanicsLint starting for {:?}", def_id);

        let mut linter = ConstPropagator::new(body, tcx);
        linter.visit_body(body);

        trace!("KnownPanicsLint done for {:?}", def_id);
    }
}

/// Visits MIR nodes, performs const propagation
/// and runs lint checks as it goes
struct ConstPropagator<'mir, 'tcx> {
    ecx: InterpCx<'tcx, DummyMachine>,
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    worklist: Vec<BasicBlock>,
    visited_blocks: DenseBitSet<BasicBlock>,
    locals: IndexVec<Local, Value<'tcx>>,
    body: &'mir Body<'tcx>,
    written_only_inside_own_block_locals: FxHashSet<Local>,
    can_const_prop: IndexVec<Local, ConstPropMode>,
}

#[derive(Debug, Clone)]
enum Value<'tcx> {
    Immediate(ImmTy<'tcx>),
    Aggregate { variant: VariantIdx, fields: IndexVec<FieldIdx, Value<'tcx>> },
    Uninit,
}

impl<'tcx> From<ImmTy<'tcx>> for Value<'tcx> {
    fn from(v: ImmTy<'tcx>) -> Self {
        Self::Immediate(v)
    }
}

impl<'tcx> Value<'tcx> {
    fn project(
        &self,
        proj: &[PlaceElem<'tcx>],
        prop: &ConstPropagator<'_, 'tcx>,
    ) -> Option<&Value<'tcx>> {
        let mut this = self;
        for proj in proj {
            this = match (*proj, this) {
                (PlaceElem::Field(idx, _), Value::Aggregate { fields, .. }) => {
                    fields.get(idx).unwrap_or(&Value::Uninit)
                }
                (PlaceElem::Index(idx), Value::Aggregate { fields, .. }) => {
                    let idx = prop.get_const(idx.into())?.immediate()?;
                    let idx = prop.ecx.read_target_usize(idx).discard_err()?.try_into().ok()?;
                    if idx <= FieldIdx::MAX_AS_U32 {
                        fields.get(FieldIdx::from_u32(idx)).unwrap_or(&Value::Uninit)
                    } else {
                        return None;
                    }
                }
                (
                    PlaceElem::ConstantIndex { offset, min_length: _, from_end: false },
                    Value::Aggregate { fields, .. },
                ) => fields
                    .get(FieldIdx::from_u32(offset.try_into().ok()?))
                    .unwrap_or(&Value::Uninit),
                _ => return None,
            };
        }
        Some(this)
    }

    fn project_mut(&mut self, proj: &[PlaceElem<'_>]) -> Option<&mut Value<'tcx>> {
        let mut this = self;
        for proj in proj {
            this = match (proj, this) {
                (PlaceElem::Field(idx, _), Value::Aggregate { fields, .. }) => {
                    fields.ensure_contains_elem(*idx, || Value::Uninit)
                }
                (PlaceElem::Field(..), val @ Value::Uninit) => {
                    *val =
                        Value::Aggregate { variant: VariantIdx::ZERO, fields: Default::default() };
                    val.project_mut(&[*proj])?
                }
                _ => return None,
            };
        }
        Some(this)
    }

    fn immediate(&self) -> Option<&ImmTy<'tcx>> {
        match self {
            Value::Immediate(op) => Some(op),
            _ => None,
        }
    }
}

impl<'tcx> LayoutOfHelpers<'tcx> for ConstPropagator<'_, 'tcx> {
    type LayoutOfResult = Result<TyAndLayout<'tcx>, LayoutError<'tcx>>;

    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, _: Span, _: Ty<'tcx>) -> LayoutError<'tcx> {
        err
    }
}

impl HasDataLayout for ConstPropagator<'_, '_> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'tcx> ty::layout::HasTyCtxt<'tcx> for ConstPropagator<'_, 'tcx> {
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx> ty::layout::HasTypingEnv<'tcx> for ConstPropagator<'_, 'tcx> {
    #[inline]
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        self.typing_env
    }
}

impl<'mir, 'tcx> ConstPropagator<'mir, 'tcx> {
    fn new(body: &'mir Body<'tcx>, tcx: TyCtxt<'tcx>) -> ConstPropagator<'mir, 'tcx> {
        let def_id = body.source.def_id();
        // FIXME(#132279): This is used during the phase transition from analysis
        // to runtime, so we have to manually specify the correct typing mode.
        let typing_env = ty::TypingEnv::post_analysis(tcx, body.source.def_id());
        let can_const_prop = CanConstProp::check(tcx, typing_env, body);
        let ecx = InterpCx::new(tcx, tcx.def_span(def_id), typing_env, DummyMachine);

        ConstPropagator {
            ecx,
            tcx,
            typing_env,
            worklist: vec![START_BLOCK],
            visited_blocks: DenseBitSet::new_empty(body.basic_blocks.len()),
            locals: IndexVec::from_elem_n(Value::Uninit, body.local_decls.len()),
            body,
            can_const_prop,
            written_only_inside_own_block_locals: Default::default(),
        }
    }

    fn local_decls(&self) -> &'mir LocalDecls<'tcx> {
        &self.body.local_decls
    }

    fn get_const(&self, place: Place<'tcx>) -> Option<&Value<'tcx>> {
        self.locals[place.local].project(&place.projection, self)
    }

    /// Remove `local` from the pool of `Locals`. Allows writing to them,
    /// but not reading from them anymore.
    fn remove_const(&mut self, local: Local) {
        self.locals[local] = Value::Uninit;
        self.written_only_inside_own_block_locals.remove(&local);
    }

    fn access_mut(&mut self, place: &Place<'_>) -> Option<&mut Value<'tcx>> {
        match self.can_const_prop[place.local] {
            ConstPropMode::NoPropagation => return None,
            ConstPropMode::OnlyInsideOwnBlock => {
                self.written_only_inside_own_block_locals.insert(place.local);
            }
            ConstPropMode::FullConstProp => {}
        }
        self.locals[place.local].project_mut(place.projection)
    }

    fn lint_root(&self, source_info: SourceInfo) -> Option<HirId> {
        source_info.scope.lint_root(&self.body.source_scopes)
    }

    fn use_ecx<F, T>(&mut self, f: F) -> Option<T>
    where
        F: FnOnce(&mut Self) -> InterpResult<'tcx, T>,
    {
        f(self)
            .map_err_info(|err| {
                trace!("InterpCx operation failed: {:?}", err);
                // Some errors shouldn't come up because creating them causes
                // an allocation, which we should avoid. When that happens,
                // dedicated error variants should be introduced instead.
                assert!(
                    !err.kind().formatted_string(),
                    "known panics lint encountered formatting error: {}",
                    format_interp_error(self.ecx.tcx.dcx(), err),
                );
                err
            })
            .discard_err()
    }

    /// Returns the value, if any, of evaluating `c`.
    fn eval_constant(&mut self, c: &ConstOperand<'tcx>) -> Option<ImmTy<'tcx>> {
        // FIXME we need to revisit this for #67176
        if c.has_param() {
            return None;
        }

        // Normalization needed b/c known panics lint runs in
        // `mir_drops_elaborated_and_const_checked`, which happens before
        // optimized MIR. Only after optimizing the MIR can we guarantee
        // that the `PostAnalysisNormalize` pass has happened and that the body's consts
        // are normalized, so any call to resolve before that needs to be
        // manually normalized.
        let val = self.tcx.try_normalize_erasing_regions(self.typing_env, c.const_).ok()?;

        self.use_ecx(|this| this.ecx.eval_mir_constant(&val, c.span, None))?
            .as_mplace_or_imm()
            .right()
    }

    /// Returns the value, if any, of evaluating `place`.
    #[instrument(level = "trace", skip(self), ret)]
    fn eval_place(&mut self, place: Place<'tcx>) -> Option<ImmTy<'tcx>> {
        match self.get_const(place)? {
            Value::Immediate(imm) => Some(imm.clone()),
            Value::Aggregate { .. } => None,
            Value::Uninit => None,
        }
    }

    /// Returns the value, if any, of evaluating `op`. Calls upon `eval_constant`
    /// or `eval_place`, depending on the variant of `Operand` used.
    fn eval_operand(&mut self, op: &Operand<'tcx>) -> Option<ImmTy<'tcx>> {
        match *op {
            Operand::Constant(ref c) => self.eval_constant(c),
            Operand::Move(place) | Operand::Copy(place) => self.eval_place(place),
        }
    }

    fn report_assert_as_lint(
        &self,
        location: Location,
        lint_kind: AssertLintKind,
        assert_kind: AssertKind<impl Debug>,
    ) {
        let source_info = self.body.source_info(location);
        if let Some(lint_root) = self.lint_root(*source_info) {
            let span = source_info.span;
            self.tcx.emit_node_span_lint(
                lint_kind.lint(),
                lint_root,
                span,
                AssertLint { span, assert_kind, lint_kind },
            );
        }
    }

    fn check_unary_op(&mut self, op: UnOp, arg: &Operand<'tcx>, location: Location) -> Option<()> {
        let arg = self.eval_operand(arg)?;
        // The only operator that can overflow is `Neg`.
        if op == UnOp::Neg && arg.layout.ty.is_integral() {
            // Compute this as `0 - arg` so we can use `SubWithOverflow` to check for overflow.
            let (arg, overflow) = self.use_ecx(|this| {
                let arg = this.ecx.read_immediate(&arg)?;
                let (_res, overflow) = this
                    .ecx
                    .binary_op(BinOp::SubWithOverflow, &ImmTy::from_int(0, arg.layout), &arg)?
                    .to_scalar_pair();
                interp_ok((arg, overflow.to_bool()?))
            })?;
            if overflow {
                self.report_assert_as_lint(
                    location,
                    AssertLintKind::ArithmeticOverflow,
                    AssertKind::OverflowNeg(arg.to_const_int()),
                );
                return None;
            }
        }

        Some(())
    }

    fn check_binary_op(
        &mut self,
        op: BinOp,
        left: &Operand<'tcx>,
        right: &Operand<'tcx>,
        location: Location,
    ) -> Option<()> {
        let r =
            self.eval_operand(right).and_then(|r| self.use_ecx(|this| this.ecx.read_immediate(&r)));
        let l =
            self.eval_operand(left).and_then(|l| self.use_ecx(|this| this.ecx.read_immediate(&l)));
        // Check for exceeding shifts *even if* we cannot evaluate the LHS.
        if matches!(op, BinOp::Shr | BinOp::Shl) {
            let r = r.clone()?;
            // We need the type of the LHS. We cannot use `place_layout` as that is the type
            // of the result, which for checked binops is not the same!
            let left_ty = left.ty(self.local_decls(), self.tcx);
            let left_size = self.ecx.layout_of(left_ty).ok()?.size;
            let right_size = r.layout.size;
            let r_bits = r.to_scalar().to_bits(right_size).discard_err();
            if r_bits.is_some_and(|b| b >= left_size.bits() as u128) {
                debug!("check_binary_op: reporting assert for {:?}", location);
                let panic = AssertKind::Overflow(
                    op,
                    // Invent a dummy value, the diagnostic ignores it anyway
                    ConstInt::new(
                        ScalarInt::try_from_uint(1_u8, left_size).unwrap(),
                        left_ty.is_signed(),
                        left_ty.is_ptr_sized_integral(),
                    ),
                    r.to_const_int(),
                );
                self.report_assert_as_lint(location, AssertLintKind::ArithmeticOverflow, panic);
                return None;
            }
        }

        // Div/Rem are handled via the assertions they trigger.
        // But for Add/Sub/Mul, those assertions only exist in debug builds, and we want to
        // lint in release builds as well, so we check on the operation instead.
        // So normalize to the "overflowing" operator, and then ensure that it
        // actually is an overflowing operator.
        let op = op.wrapping_to_overflowing().unwrap_or(op);
        // The remaining operators are handled through `wrapping_to_overflowing`.
        if let (Some(l), Some(r)) = (l, r)
            && l.layout.ty.is_integral()
            && op.is_overflowing()
            && self.use_ecx(|this| {
                let (_res, overflow) = this.ecx.binary_op(op, &l, &r)?.to_scalar_pair();
                overflow.to_bool()
            })?
        {
            self.report_assert_as_lint(
                location,
                AssertLintKind::ArithmeticOverflow,
                AssertKind::Overflow(op, l.to_const_int(), r.to_const_int()),
            );
            return None;
        }

        Some(())
    }

    fn check_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) -> Option<()> {
        // Perform any special handling for specific Rvalue types.
        // Generally, checks here fall into one of two categories:
        //   1. Additional checking to provide useful lints to the user
        //        - In this case, we will do some validation and then fall through to the
        //          end of the function which evals the assignment.
        //   2. Working around bugs in other parts of the compiler
        //        - In this case, we'll return `None` from this function to stop evaluation.
        match rvalue {
            // Additional checking: give lints to the user if an overflow would occur.
            // We do this here and not in the `Assert` terminator as that terminator is
            // only sometimes emitted (overflow checks can be disabled), but we want to always
            // lint.
            Rvalue::UnaryOp(op, arg) => {
                trace!("checking UnaryOp(op = {:?}, arg = {:?})", op, arg);
                self.check_unary_op(*op, arg, location)?;
            }
            Rvalue::BinaryOp(op, box (left, right)) => {
                trace!("checking BinaryOp(op = {:?}, left = {:?}, right = {:?})", op, left, right);
                self.check_binary_op(*op, left, right, location)?;
            }

            // Do not try creating references (#67862)
            Rvalue::RawPtr(_, place) | Rvalue::Ref(_, _, place) => {
                trace!("skipping RawPtr | Ref for {:?}", place);

                // This may be creating mutable references or immutable references to cells.
                // If that happens, the pointed to value could be mutated via that reference.
                // Since we aren't tracking references, the const propagator loses track of what
                // value the local has right now.
                // Thus, all locals that have their reference taken
                // must not take part in propagation.
                self.remove_const(place.local);

                return None;
            }
            Rvalue::ThreadLocalRef(def_id) => {
                trace!("skipping ThreadLocalRef({:?})", def_id);

                return None;
            }

            // There's no other checking to do at this time.
            Rvalue::Aggregate(..)
            | Rvalue::Use(..)
            | Rvalue::CopyForDeref(..)
            | Rvalue::Repeat(..)
            | Rvalue::Cast(..)
            | Rvalue::ShallowInitBox(..)
            | Rvalue::Discriminant(..)
            | Rvalue::NullaryOp(..)
            | Rvalue::WrapUnsafeBinder(..) => {}
        }

        // FIXME we need to revisit this for #67176
        if rvalue.has_param() {
            return None;
        }
        if !rvalue.ty(self.local_decls(), self.tcx).is_sized(self.tcx, self.typing_env) {
            // the interpreter doesn't support unsized locals (only unsized arguments),
            // but rustc does (in a kinda broken way), so we have to skip them here
            return None;
        }

        Some(())
    }

    fn check_assertion(
        &mut self,
        expected: bool,
        msg: &AssertKind<Operand<'tcx>>,
        cond: &Operand<'tcx>,
        location: Location,
    ) {
        let Some(value) = &self.eval_operand(cond) else { return };
        trace!("assertion on {:?} should be {:?}", value, expected);

        let expected = Scalar::from_bool(expected);
        let Some(value_const) = self.use_ecx(|this| this.ecx.read_scalar(value)) else { return };

        if expected != value_const {
            // Poison all places this operand references so that further code
            // doesn't use the invalid value
            if let Some(place) = cond.place() {
                self.remove_const(place.local);
            }

            enum DbgVal<T> {
                Val(T),
                Underscore,
            }
            impl<T: std::fmt::Debug> std::fmt::Debug for DbgVal<T> {
                fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    match self {
                        Self::Val(val) => val.fmt(fmt),
                        Self::Underscore => fmt.write_str("_"),
                    }
                }
            }
            let mut eval_to_int = |op| {
                // This can be `None` if the lhs wasn't const propagated and we just
                // triggered the assert on the value of the rhs.
                self.eval_operand(op)
                    .and_then(|op| self.ecx.read_immediate(&op).discard_err())
                    .map_or(DbgVal::Underscore, |op| DbgVal::Val(op.to_const_int()))
            };
            let msg = match msg {
                AssertKind::DivisionByZero(op) => AssertKind::DivisionByZero(eval_to_int(op)),
                AssertKind::RemainderByZero(op) => AssertKind::RemainderByZero(eval_to_int(op)),
                AssertKind::Overflow(bin_op @ (BinOp::Div | BinOp::Rem), op1, op2) => {
                    // Division overflow is *UB* in the MIR, and different than the
                    // other overflow checks.
                    AssertKind::Overflow(*bin_op, eval_to_int(op1), eval_to_int(op2))
                }
                AssertKind::BoundsCheck { len, index } => {
                    let len = eval_to_int(len);
                    let index = eval_to_int(index);
                    AssertKind::BoundsCheck { len, index }
                }
                // Remaining overflow errors are already covered by checks on the binary operators.
                AssertKind::Overflow(..) | AssertKind::OverflowNeg(_) => return,
                // Need proper const propagator for these.
                _ => return,
            };
            self.report_assert_as_lint(location, AssertLintKind::UnconditionalPanic, msg);
        }
    }

    fn ensure_not_propagated(&self, local: Local) {
        if cfg!(debug_assertions) {
            let val = self.get_const(local.into());
            assert!(
                matches!(val, Some(Value::Uninit))
                    || self
                        .layout_of(self.local_decls()[local].ty)
                        .map_or(true, |layout| layout.is_zst()),
                "failed to remove values for `{local:?}`, value={val:?}",
            )
        }
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn eval_rvalue(&mut self, rvalue: &Rvalue<'tcx>, dest: &Place<'tcx>) -> Option<()> {
        if !dest.projection.is_empty() {
            return None;
        }
        use rustc_middle::mir::Rvalue::*;
        let layout = self.ecx.layout_of(dest.ty(self.body, self.tcx).ty).ok()?;
        trace!(?layout);

        let val: Value<'_> = match *rvalue {
            ThreadLocalRef(_) => return None,

            Use(ref operand) | WrapUnsafeBinder(ref operand, _) => {
                self.eval_operand(operand)?.into()
            }

            CopyForDeref(place) => self.eval_place(place)?.into(),

            BinaryOp(bin_op, box (ref left, ref right)) => {
                let left = self.eval_operand(left)?;
                let left = self.use_ecx(|this| this.ecx.read_immediate(&left))?;

                let right = self.eval_operand(right)?;
                let right = self.use_ecx(|this| this.ecx.read_immediate(&right))?;

                let val = self.use_ecx(|this| this.ecx.binary_op(bin_op, &left, &right))?;
                if matches!(val.layout.backend_repr, BackendRepr::ScalarPair(..)) {
                    // FIXME `Value` should properly support pairs in `Immediate`... but currently
                    // it does not.
                    let (val, overflow) = val.to_pair(&self.ecx);
                    Value::Aggregate {
                        variant: VariantIdx::ZERO,
                        fields: [val.into(), overflow.into()].into_iter().collect(),
                    }
                } else {
                    val.into()
                }
            }

            UnaryOp(un_op, ref operand) => {
                let operand = self.eval_operand(operand)?;
                let val = self.use_ecx(|this| this.ecx.read_immediate(&operand))?;

                let val = self.use_ecx(|this| this.ecx.unary_op(un_op, &val))?;
                val.into()
            }

            Aggregate(ref kind, ref fields) => Value::Aggregate {
                fields: fields
                    .iter()
                    .map(|field| self.eval_operand(field).map_or(Value::Uninit, Value::Immediate))
                    .collect(),
                variant: match **kind {
                    AggregateKind::Adt(_, variant, _, _, _) => variant,
                    AggregateKind::Array(_)
                    | AggregateKind::Tuple
                    | AggregateKind::RawPtr(_, _)
                    | AggregateKind::Closure(_, _)
                    | AggregateKind::Coroutine(_, _)
                    | AggregateKind::CoroutineClosure(_, _) => VariantIdx::ZERO,
                },
            },

            Repeat(ref op, n) => {
                trace!(?op, ?n);
                return None;
            }

            Ref(..) | RawPtr(..) => return None,

            NullaryOp(ref null_op, ty) => {
                let op_layout = self.ecx.layout_of(ty).ok()?;
                let val = match null_op {
                    NullOp::SizeOf => op_layout.size.bytes(),
                    NullOp::AlignOf => op_layout.align.bytes(),
                    NullOp::OffsetOf(fields) => self
                        .tcx
                        .offset_of_subfield(self.typing_env, op_layout, fields.iter())
                        .bytes(),
                    NullOp::UbChecks => return None,
                    NullOp::ContractChecks => return None,
                };
                ImmTy::from_scalar(Scalar::from_target_usize(val, self), layout).into()
            }

            ShallowInitBox(..) => return None,

            Cast(ref kind, ref value, to) => match kind {
                CastKind::IntToInt | CastKind::IntToFloat => {
                    let value = self.eval_operand(value)?;
                    let value = self.ecx.read_immediate(&value).discard_err()?;
                    let to = self.ecx.layout_of(to).ok()?;
                    let res = self.ecx.int_to_int_or_float(&value, to).discard_err()?;
                    res.into()
                }
                CastKind::FloatToFloat | CastKind::FloatToInt => {
                    let value = self.eval_operand(value)?;
                    let value = self.ecx.read_immediate(&value).discard_err()?;
                    let to = self.ecx.layout_of(to).ok()?;
                    let res = self.ecx.float_to_float_or_int(&value, to).discard_err()?;
                    res.into()
                }
                CastKind::Transmute | CastKind::Subtype => {
                    let value = self.eval_operand(value)?;
                    let to = self.ecx.layout_of(to).ok()?;
                    // `offset` for immediates only supports scalar/scalar-pair ABIs,
                    // so bail out if the target is not one.
                    match (value.layout.backend_repr, to.backend_repr) {
                        (BackendRepr::Scalar(..), BackendRepr::Scalar(..)) => {}
                        (BackendRepr::ScalarPair(..), BackendRepr::ScalarPair(..)) => {}
                        _ => return None,
                    }

                    value.offset(Size::ZERO, to, &self.ecx).discard_err()?.into()
                }
                _ => return None,
            },

            Discriminant(place) => {
                let variant = match self.get_const(place)? {
                    Value::Immediate(op) => {
                        let op = op.clone();
                        self.use_ecx(|this| this.ecx.read_discriminant(&op))?
                    }
                    Value::Aggregate { variant, .. } => *variant,
                    Value::Uninit => return None,
                };
                let imm = self.use_ecx(|this| {
                    this.ecx.discriminant_for_variant(
                        place.ty(this.local_decls(), this.tcx).ty,
                        variant,
                    )
                })?;
                imm.into()
            }
        };
        trace!(?val);

        *self.access_mut(dest)? = val;

        Some(())
    }
}

impl<'tcx> Visitor<'tcx> for ConstPropagator<'_, 'tcx> {
    fn visit_body(&mut self, body: &Body<'tcx>) {
        while let Some(bb) = self.worklist.pop() {
            if !self.visited_blocks.insert(bb) {
                continue;
            }

            let data = &body.basic_blocks[bb];
            self.visit_basic_block_data(bb, data);
        }
    }

    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        self.super_operand(operand, location);
    }

    fn visit_const_operand(&mut self, constant: &ConstOperand<'tcx>, location: Location) {
        trace!("visit_const_operand: {:?}", constant);
        self.super_const_operand(constant, location);
        self.eval_constant(constant);
    }

    fn visit_assign(&mut self, place: &Place<'tcx>, rvalue: &Rvalue<'tcx>, location: Location) {
        self.super_assign(place, rvalue, location);

        let Some(()) = self.check_rvalue(rvalue, location) else { return };

        match self.can_const_prop[place.local] {
            // Do nothing if the place is indirect.
            _ if place.is_indirect() => {}
            ConstPropMode::NoPropagation => self.ensure_not_propagated(place.local),
            ConstPropMode::OnlyInsideOwnBlock | ConstPropMode::FullConstProp => {
                if self.eval_rvalue(rvalue, place).is_none() {
                    // Const prop failed, so erase the destination, ensuring that whatever happens
                    // from here on, does not know about the previous value.
                    // This is important in case we have
                    // ```rust
                    // let mut x = 42;
                    // x = SOME_MUTABLE_STATIC;
                    // // x must now be uninit
                    // ```
                    // FIXME: we overzealously erase the entire local, because that's easier to
                    // implement.
                    trace!(
                        "propagation into {:?} failed.
                        Nuking the entire site from orbit, it's the only way to be sure",
                        place,
                    );
                    self.remove_const(place.local);
                }
            }
        }
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        trace!("visit_statement: {:?}", statement);

        // We want to evaluate operands before any change to the assigned-to value,
        // so we recurse first.
        self.super_statement(statement, location);

        match statement.kind {
            StatementKind::SetDiscriminant { ref place, variant_index } => {
                match self.can_const_prop[place.local] {
                    // Do nothing if the place is indirect.
                    _ if place.is_indirect() => {}
                    ConstPropMode::NoPropagation => self.ensure_not_propagated(place.local),
                    ConstPropMode::FullConstProp | ConstPropMode::OnlyInsideOwnBlock => {
                        match self.access_mut(place) {
                            Some(Value::Aggregate { variant, .. }) => *variant = variant_index,
                            _ => self.remove_const(place.local),
                        }
                    }
                }
            }
            StatementKind::StorageLive(local) => {
                self.remove_const(local);
            }
            StatementKind::StorageDead(local) => {
                self.remove_const(local);
            }
            _ => {}
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        self.super_terminator(terminator, location);
        match &terminator.kind {
            TerminatorKind::Assert { expected, msg, cond, .. } => {
                self.check_assertion(*expected, msg, cond, location);
            }
            TerminatorKind::SwitchInt { discr, targets } => {
                if let Some(ref value) = self.eval_operand(discr)
                    && let Some(value_const) = self.use_ecx(|this| this.ecx.read_scalar(value))
                    && let Some(constant) = value_const.to_bits(value_const.size()).discard_err()
                {
                    // We managed to evaluate the discriminant, so we know we only need to visit
                    // one target.
                    let target = targets.target_for_value(constant);
                    self.worklist.push(target);
                    return;
                }
                // We failed to evaluate the discriminant, fallback to visiting all successors.
            }
            // None of these have Operands to const-propagate.
            TerminatorKind::Goto { .. }
            | TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Return
            | TerminatorKind::TailCall { .. }
            | TerminatorKind::Unreachable
            | TerminatorKind::Drop { .. }
            | TerminatorKind::Yield { .. }
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::Call { .. }
            | TerminatorKind::InlineAsm { .. } => {}
        }

        self.worklist.extend(terminator.successors());
    }

    fn visit_basic_block_data(&mut self, block: BasicBlock, data: &BasicBlockData<'tcx>) {
        self.super_basic_block_data(block, data);

        // We remove all Locals which are restricted in propagation to their containing blocks and
        // which were modified in the current block.
        // Take it out of the ecx so we can get a mutable reference to the ecx for `remove_const`.
        let mut written_only_inside_own_block_locals =
            std::mem::take(&mut self.written_only_inside_own_block_locals);

        // This loop can get very hot for some bodies: it check each local in each bb.
        // To avoid this quadratic behaviour, we only clear the locals that were modified inside
        // the current block.
        // The order in which we remove consts does not matter.
        #[allow(rustc::potential_query_instability)]
        for local in written_only_inside_own_block_locals.drain() {
            debug_assert_eq!(self.can_const_prop[local], ConstPropMode::OnlyInsideOwnBlock);
            self.remove_const(local);
        }
        self.written_only_inside_own_block_locals = written_only_inside_own_block_locals;

        if cfg!(debug_assertions) {
            for (local, &mode) in self.can_const_prop.iter_enumerated() {
                match mode {
                    ConstPropMode::FullConstProp => {}
                    ConstPropMode::NoPropagation | ConstPropMode::OnlyInsideOwnBlock => {
                        self.ensure_not_propagated(local);
                    }
                }
            }
        }
    }
}

/// The maximum number of bytes that we'll allocate space for a local or the return value.
/// Needed for #66397, because otherwise we eval into large places and that can cause OOM or just
/// Severely regress performance.
const MAX_ALLOC_LIMIT: u64 = 1024;

/// The mode that `ConstProp` is allowed to run in for a given `Local`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ConstPropMode {
    /// The `Local` can be propagated into and reads of this `Local` can also be propagated.
    FullConstProp,
    /// The `Local` can only be propagated into and from its own block.
    OnlyInsideOwnBlock,
    /// The `Local` cannot be part of propagation at all. Any statement
    /// referencing it either for reading or writing will not get propagated.
    NoPropagation,
}

/// A visitor that determines locals in a MIR body
/// that can be const propagated
struct CanConstProp {
    can_const_prop: IndexVec<Local, ConstPropMode>,
    // False at the beginning. Once set, no more assignments are allowed to that local.
    found_assignment: DenseBitSet<Local>,
}

impl CanConstProp {
    /// Returns true if `local` can be propagated
    fn check<'tcx>(
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        body: &Body<'tcx>,
    ) -> IndexVec<Local, ConstPropMode> {
        let mut cpv = CanConstProp {
            can_const_prop: IndexVec::from_elem(ConstPropMode::FullConstProp, &body.local_decls),
            found_assignment: DenseBitSet::new_empty(body.local_decls.len()),
        };
        for (local, val) in cpv.can_const_prop.iter_enumerated_mut() {
            let ty = body.local_decls[local].ty;
            if ty.is_async_drop_in_place_coroutine(tcx) {
                // No const propagation for async drop coroutine (AsyncDropGlue).
                // Otherwise, tcx.layout_of(typing_env.as_query_input(ty)) will be called
                // (early layout request for async drop coroutine) to calculate layout size.
                // Layout for `async_drop_in_place<T>::{closure}` may only be known with known T.
                *val = ConstPropMode::NoPropagation;
                continue;
            } else if ty.is_union() {
                // Unions are incompatible with the current implementation of
                // const prop because Rust has no concept of an active
                // variant of a union
                *val = ConstPropMode::NoPropagation;
            } else {
                match tcx.layout_of(typing_env.as_query_input(ty)) {
                    Ok(layout) if layout.size < Size::from_bytes(MAX_ALLOC_LIMIT) => {}
                    // Either the layout fails to compute, then we can't use this local anyway
                    // or the local is too large, then we don't want to.
                    _ => {
                        *val = ConstPropMode::NoPropagation;
                        continue;
                    }
                }
            }
        }
        // Consider that arguments are assigned on entry.
        for arg in body.args_iter() {
            cpv.found_assignment.insert(arg);
        }
        cpv.visit_body(body);
        cpv.can_const_prop
    }
}

impl<'tcx> Visitor<'tcx> for CanConstProp {
    fn visit_place(&mut self, place: &Place<'tcx>, mut context: PlaceContext, loc: Location) {
        use rustc_middle::mir::visit::PlaceContext::*;

        // Dereferencing just read the address of `place.local`.
        if place.projection.first() == Some(&PlaceElem::Deref) {
            context = NonMutatingUse(NonMutatingUseContext::Copy);
        }

        self.visit_local(place.local, context, loc);
        self.visit_projection(place.as_ref(), context, loc);
    }

    fn visit_local(&mut self, local: Local, context: PlaceContext, _: Location) {
        use rustc_middle::mir::visit::PlaceContext::*;
        match context {
            // These are just stores, where the storing is not propagatable, but there may be later
            // mutations of the same local via `Store`
            | MutatingUse(MutatingUseContext::Call)
            | MutatingUse(MutatingUseContext::AsmOutput)
            | MutatingUse(MutatingUseContext::Deinit)
            // Actual store that can possibly even propagate a value
            | MutatingUse(MutatingUseContext::Store)
            | MutatingUse(MutatingUseContext::SetDiscriminant) => {
                if !self.found_assignment.insert(local) {
                    match &mut self.can_const_prop[local] {
                        // If the local can only get propagated in its own block, then we don't have
                        // to worry about multiple assignments, as we'll nuke the const state at the
                        // end of the block anyway, and inside the block we overwrite previous
                        // states as applicable.
                        ConstPropMode::OnlyInsideOwnBlock => {}
                        ConstPropMode::NoPropagation => {}
                        other @ ConstPropMode::FullConstProp => {
                            trace!(
                                "local {:?} can't be propagated because of multiple assignments. Previous state: {:?}",
                                local, other,
                            );
                            *other = ConstPropMode::OnlyInsideOwnBlock;
                        }
                    }
                }
            }
            // Reading constants is allowed an arbitrary number of times
            NonMutatingUse(NonMutatingUseContext::Copy)
            | NonMutatingUse(NonMutatingUseContext::Move)
            | NonMutatingUse(NonMutatingUseContext::Inspect)
            | NonMutatingUse(NonMutatingUseContext::PlaceMention)
            | NonUse(_) => {}

            // These could be propagated with a smarter analysis or just some careful thinking about
            // whether they'd be fine right now.
            MutatingUse(MutatingUseContext::Yield)
            | MutatingUse(MutatingUseContext::Drop)
            | MutatingUse(MutatingUseContext::Retag)
            // These can't ever be propagated under any scheme, as we can't reason about indirect
            // mutation.
            | NonMutatingUse(NonMutatingUseContext::SharedBorrow)
            | NonMutatingUse(NonMutatingUseContext::FakeBorrow)
            | NonMutatingUse(NonMutatingUseContext::RawBorrow)
            | MutatingUse(MutatingUseContext::Borrow)
            | MutatingUse(MutatingUseContext::RawBorrow) => {
                trace!("local {:?} can't be propagated because it's used: {:?}", local, context);
                self.can_const_prop[local] = ConstPropMode::NoPropagation;
            }
            MutatingUse(MutatingUseContext::Projection)
            | NonMutatingUse(NonMutatingUseContext::Projection) => bug!("visit_place should not pass {context:?} for {local:?}"),
        }
    }
}
