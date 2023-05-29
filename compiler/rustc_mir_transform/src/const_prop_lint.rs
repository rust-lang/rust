//! Propagates constants for early reporting of statically known
//! assertion failures

use std::fmt::Debug;

use either::Left;

use rustc_const_eval::interpret::Immediate;
use rustc_const_eval::interpret::{
    self, InterpCx, InterpResult, LocalValue, MemoryKind, OpTy, Scalar, StackPopCleanup,
};
use rustc_const_eval::ReportErrorExt;
use rustc_hir::def::DefKind;
use rustc_hir::HirId;
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::ty::layout::{LayoutError, LayoutOf, LayoutOfHelpers, TyAndLayout};
use rustc_middle::ty::InternalSubsts;
use rustc_middle::ty::{
    self, ConstInt, Instance, ParamEnv, ScalarInt, Ty, TyCtxt, TypeVisitableExt,
};
use rustc_span::Span;
use rustc_target::abi::{HasDataLayout, Size, TargetDataLayout};
use rustc_trait_selection::traits;

use crate::const_prop::CanConstProp;
use crate::const_prop::ConstPropMachine;
use crate::const_prop::ConstPropMode;
use crate::errors::AssertLint;
use crate::MirLint;

/// The maximum number of bytes that we'll allocate space for a local or the return value.
/// Needed for #66397, because otherwise we eval into large places and that can cause OOM or just
/// Severely regress performance.
const MAX_ALLOC_LIMIT: u64 = 1024;

pub struct ConstProp;

impl<'tcx> MirLint<'tcx> for ConstProp {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        // will be evaluated by miri and produce its errors there
        if body.source.promoted.is_some() {
            return;
        }

        let def_id = body.source.def_id().expect_local();
        let is_fn_like = tcx.def_kind(def_id).is_fn_like();
        let is_assoc_const = tcx.def_kind(def_id) == DefKind::AssocConst;

        // Only run const prop on functions, methods, closures and associated constants
        if !is_fn_like && !is_assoc_const {
            // skip anon_const/statics/consts because they'll be evaluated by miri anyway
            trace!("ConstProp skipped for {:?}", def_id);
            return;
        }

        let is_generator = tcx.type_of(def_id.to_def_id()).subst_identity().is_generator();
        // FIXME(welseywiser) const prop doesn't work on generators because of query cycles
        // computing their layout.
        if is_generator {
            trace!("ConstProp skipped for generator {:?}", def_id);
            return;
        }

        // Check if it's even possible to satisfy the 'where' clauses
        // for this item.
        // This branch will never be taken for any normal function.
        // However, it's possible to `#!feature(trivial_bounds)]` to write
        // a function with impossible to satisfy clauses, e.g.:
        // `fn foo() where String: Copy {}`
        //
        // We don't usually need to worry about this kind of case,
        // since we would get a compilation error if the user tried
        // to call it. However, since we can do const propagation
        // even without any calls to the function, we need to make
        // sure that it even makes sense to try to evaluate the body.
        // If there are unsatisfiable where clauses, then all bets are
        // off, and we just give up.
        //
        // We manually filter the predicates, skipping anything that's not
        // "global". We are in a potentially generic context
        // (e.g. we are evaluating a function without substituting generic
        // parameters, so this filtering serves two purposes:
        //
        // 1. We skip evaluating any predicates that we would
        // never be able prove are unsatisfiable (e.g. `<T as Foo>`
        // 2. We avoid trying to normalize predicates involving generic
        // parameters (e.g. `<T as Foo>::MyItem`). This can confuse
        // the normalization code (leading to cycle errors), since
        // it's usually never invoked in this way.
        let predicates = tcx
            .predicates_of(def_id.to_def_id())
            .predicates
            .iter()
            .filter_map(|(p, _)| if p.is_global() { Some(*p) } else { None });
        if traits::impossible_predicates(tcx, traits::elaborate(tcx, predicates).collect()) {
            trace!("ConstProp skipped for {:?}: found unsatisfiable predicates", def_id);
            return;
        }

        trace!("ConstProp starting for {:?}", def_id);

        let dummy_body = &Body::new(
            body.source,
            (*body.basic_blocks).to_owned(),
            body.source_scopes.clone(),
            body.local_decls.clone(),
            Default::default(),
            body.arg_count,
            Default::default(),
            body.span,
            body.generator_kind(),
            body.tainted_by_errors,
        );

        // FIXME(oli-obk, eddyb) Optimize locals (or even local paths) to hold
        // constants, instead of just checking for const-folding succeeding.
        // That would require a uniform one-def no-mutation analysis
        // and RPO (or recursing when needing the value of a local).
        let mut optimization_finder = ConstPropagator::new(body, dummy_body, tcx);
        optimization_finder.visit_body(body);

        trace!("ConstProp done for {:?}", def_id);
    }
}

/// Finds optimization opportunities on the MIR.
struct ConstPropagator<'mir, 'tcx> {
    ecx: InterpCx<'mir, 'tcx, ConstPropMachine<'mir, 'tcx>>,
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    worklist: Vec<BasicBlock>,
    visited_blocks: BitSet<BasicBlock>,
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

impl<'tcx> ty::layout::HasParamEnv<'tcx> for ConstPropagator<'_, 'tcx> {
    #[inline]
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.param_env
    }
}

impl<'mir, 'tcx> ConstPropagator<'mir, 'tcx> {
    fn new(
        body: &Body<'tcx>,
        dummy_body: &'mir Body<'tcx>,
        tcx: TyCtxt<'tcx>,
    ) -> ConstPropagator<'mir, 'tcx> {
        let def_id = body.source.def_id();
        let substs = &InternalSubsts::identity_for_item(tcx, def_id);
        let param_env = tcx.param_env_reveal_all_normalized(def_id);

        let can_const_prop = CanConstProp::check(tcx, param_env, body);
        let mut ecx = InterpCx::new(
            tcx,
            tcx.def_span(def_id),
            param_env,
            ConstPropMachine::new(can_const_prop),
        );

        let ret_layout = ecx
            .layout_of(body.bound_return_ty().subst(tcx, substs))
            .ok()
            // Don't bother allocating memory for large values.
            // I don't know how return types can seem to be unsized but this happens in the
            // `type/type-unsatisfiable.rs` test.
            .filter(|ret_layout| {
                ret_layout.is_sized() && ret_layout.size < Size::from_bytes(MAX_ALLOC_LIMIT)
            })
            .unwrap_or_else(|| ecx.layout_of(tcx.types.unit).unwrap());

        let ret = ecx
            .allocate(ret_layout, MemoryKind::Stack)
            .expect("couldn't perform small allocation")
            .into();

        ecx.push_stack_frame(
            Instance::new(def_id, substs),
            dummy_body,
            &ret,
            StackPopCleanup::Root { cleanup: false },
        )
        .expect("failed to push initial stack frame");

        ConstPropagator {
            ecx,
            tcx,
            param_env,
            worklist: vec![START_BLOCK],
            visited_blocks: BitSet::new_empty(body.basic_blocks.len()),
        }
    }

    fn body(&self) -> &'mir Body<'tcx> {
        self.ecx.frame().body
    }

    fn local_decls(&self) -> &'mir LocalDecls<'tcx> {
        &self.body().local_decls
    }

    fn get_const(&self, place: Place<'tcx>) -> Option<OpTy<'tcx>> {
        let op = match self.ecx.eval_place_to_op(place, None) {
            Ok(op) => {
                if matches!(*op, interpret::Operand::Immediate(Immediate::Uninit)) {
                    // Make sure nobody accidentally uses this value.
                    return None;
                }
                op
            }
            Err(e) => {
                trace!("get_const failed: {:?}", e.into_kind().debug());
                return None;
            }
        };

        // Try to read the local as an immediate so that if it is representable as a scalar, we can
        // handle it as such, but otherwise, just return the value as is.
        Some(match self.ecx.read_immediate_raw(&op) {
            Ok(Left(imm)) => imm.into(),
            _ => op,
        })
    }

    /// Remove `local` from the pool of `Locals`. Allows writing to them,
    /// but not reading from them anymore.
    fn remove_const(ecx: &mut InterpCx<'mir, 'tcx, ConstPropMachine<'mir, 'tcx>>, local: Local) {
        ecx.frame_mut().locals[local].value =
            LocalValue::Live(interpret::Operand::Immediate(interpret::Immediate::Uninit));
        ecx.machine.written_only_inside_own_block_locals.remove(&local);
    }

    fn lint_root(&self, source_info: SourceInfo) -> Option<HirId> {
        source_info.scope.lint_root(&self.body().source_scopes)
    }

    fn use_ecx<F, T>(&mut self, location: Location, f: F) -> Option<T>
    where
        F: FnOnce(&mut Self) -> InterpResult<'tcx, T>,
    {
        // Overwrite the PC -- whatever the interpreter does to it does not make any sense anyway.
        self.ecx.frame_mut().loc = Left(location);
        match f(self) {
            Ok(val) => Some(val),
            Err(error) => {
                trace!("InterpCx operation failed: {:?}", error);
                // Some errors shouldn't come up because creating them causes
                // an allocation, which we should avoid. When that happens,
                // dedicated error variants should be introduced instead.
                assert!(
                    !error.kind().formatted_string(),
                    "const-prop encountered formatting error: {error:?}",
                );
                None
            }
        }
    }

    /// Returns the value, if any, of evaluating `c`.
    fn eval_constant(&mut self, c: &Constant<'tcx>, location: Location) -> Option<OpTy<'tcx>> {
        // FIXME we need to revisit this for #67176
        if c.has_param() {
            return None;
        }

        // Normalization needed b/c const prop lint runs in
        // `mir_drops_elaborated_and_const_checked`, which happens before
        // optimized MIR. Only after optimizing the MIR can we guarantee
        // that the `RevealAll` pass has happened and that the body's consts
        // are normalized, so any call to resolve before that needs to be
        // manually normalized.
        let val = self.tcx.try_normalize_erasing_regions(self.param_env, c.literal).ok()?;

        self.use_ecx(location, |this| this.ecx.eval_mir_constant(&val, Some(c.span), None))
    }

    /// Returns the value, if any, of evaluating `place`.
    fn eval_place(&mut self, place: Place<'tcx>, location: Location) -> Option<OpTy<'tcx>> {
        trace!("eval_place(place={:?})", place);
        self.use_ecx(location, |this| this.ecx.eval_place_to_op(place, None))
    }

    /// Returns the value, if any, of evaluating `op`. Calls upon `eval_constant`
    /// or `eval_place`, depending on the variant of `Operand` used.
    fn eval_operand(&mut self, op: &Operand<'tcx>, location: Location) -> Option<OpTy<'tcx>> {
        match *op {
            Operand::Constant(ref c) => self.eval_constant(c, location),
            Operand::Move(place) | Operand::Copy(place) => self.eval_place(place, location),
        }
    }

    fn report_assert_as_lint(&self, source_info: &SourceInfo, lint: AssertLint<impl Debug>) {
        if let Some(lint_root) = self.lint_root(*source_info) {
            self.tcx.emit_spanned_lint(lint.lint(), lint_root, source_info.span, lint);
        }
    }

    fn check_unary_op(&mut self, op: UnOp, arg: &Operand<'tcx>, location: Location) -> Option<()> {
        if let (val, true) = self.use_ecx(location, |this| {
            let val = this.ecx.read_immediate(&this.ecx.eval_operand(arg, None)?)?;
            let (_res, overflow, _ty) = this.ecx.overflowing_unary_op(op, &val)?;
            Ok((val, overflow))
        })? {
            // `AssertKind` only has an `OverflowNeg` variant, so make sure that is
            // appropriate to use.
            assert_eq!(op, UnOp::Neg, "Neg is the only UnOp that can overflow");
            let source_info = self.body().source_info(location);
            self.report_assert_as_lint(
                source_info,
                AssertLint::ArithmeticOverflow(
                    source_info.span,
                    AssertKind::OverflowNeg(val.to_const_int()),
                ),
            );
            return None;
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
        let r = self.use_ecx(location, |this| {
            this.ecx.read_immediate(&this.ecx.eval_operand(right, None)?)
        });
        let l = self
            .use_ecx(location, |this| this.ecx.read_immediate(&this.ecx.eval_operand(left, None)?));
        // Check for exceeding shifts *even if* we cannot evaluate the LHS.
        if matches!(op, BinOp::Shr | BinOp::Shl) {
            let r = r.clone()?;
            // We need the type of the LHS. We cannot use `place_layout` as that is the type
            // of the result, which for checked binops is not the same!
            let left_ty = left.ty(self.local_decls(), self.tcx);
            let left_size = self.ecx.layout_of(left_ty).ok()?.size;
            let right_size = r.layout.size;
            let r_bits = r.to_scalar().to_bits(right_size).ok();
            if r_bits.is_some_and(|b| b >= left_size.bits() as u128) {
                debug!("check_binary_op: reporting assert for {:?}", location);
                let source_info = self.body().source_info(location);
                let panic = AssertKind::Overflow(
                    op,
                    match l {
                        Some(l) => l.to_const_int(),
                        // Invent a dummy value, the diagnostic ignores it anyway
                        None => ConstInt::new(
                            ScalarInt::try_from_uint(1_u8, left_size).unwrap(),
                            left_ty.is_signed(),
                            left_ty.is_ptr_sized_integral(),
                        ),
                    },
                    r.to_const_int(),
                );
                self.report_assert_as_lint(
                    source_info,
                    AssertLint::ArithmeticOverflow(source_info.span, panic),
                );
                return None;
            }
        }

        if let (Some(l), Some(r)) = (l, r) {
            // The remaining operators are handled through `overflowing_binary_op`.
            if self.use_ecx(location, |this| {
                let (_res, overflow, _ty) = this.ecx.overflowing_binary_op(op, &l, &r)?;
                Ok(overflow)
            })? {
                let source_info = self.body().source_info(location);
                self.report_assert_as_lint(
                    source_info,
                    AssertLint::ArithmeticOverflow(
                        source_info.span,
                        AssertKind::Overflow(op, l.to_const_int(), r.to_const_int()),
                    ),
                );
                return None;
            }
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
            Rvalue::CheckedBinaryOp(op, box (left, right)) => {
                trace!(
                    "checking CheckedBinaryOp(op = {:?}, left = {:?}, right = {:?})",
                    op,
                    left,
                    right
                );
                self.check_binary_op(*op, left, right, location)?;
            }

            // Do not try creating references (#67862)
            Rvalue::AddressOf(_, place) | Rvalue::Ref(_, _, place) => {
                trace!("skipping AddressOf | Ref for {:?}", place);

                // This may be creating mutable references or immutable references to cells.
                // If that happens, the pointed to value could be mutated via that reference.
                // Since we aren't tracking references, the const propagator loses track of what
                // value the local has right now.
                // Thus, all locals that have their reference taken
                // must not take part in propagation.
                Self::remove_const(&mut self.ecx, place.local);

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
            | Rvalue::Len(..)
            | Rvalue::Cast(..)
            | Rvalue::ShallowInitBox(..)
            | Rvalue::Discriminant(..)
            | Rvalue::NullaryOp(..) => {}
        }

        // FIXME we need to revisit this for #67176
        if rvalue.has_param() {
            return None;
        }
        if !rvalue.ty(self.local_decls(), self.tcx).is_sized(self.tcx, self.param_env) {
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
    ) -> Option<!> {
        let value = &self.eval_operand(&cond, location)?;
        trace!("assertion on {:?} should be {:?}", value, expected);

        let expected = Scalar::from_bool(expected);
        let value_const = self.use_ecx(location, |this| this.ecx.read_scalar(&value))?;

        if expected != value_const {
            // Poison all places this operand references so that further code
            // doesn't use the invalid value
            if let Some(place) = cond.place() {
                Self::remove_const(&mut self.ecx, place.local);
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
                self.eval_operand(op, location)
                    .and_then(|op| self.ecx.read_immediate(&op).ok())
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
                AssertKind::BoundsCheck { ref len, ref index } => {
                    let len = eval_to_int(len);
                    let index = eval_to_int(index);
                    AssertKind::BoundsCheck { len, index }
                }
                // Remaining overflow errors are already covered by checks on the binary operators.
                AssertKind::Overflow(..) | AssertKind::OverflowNeg(_) => return None,
                // Need proper const propagator for these.
                _ => return None,
            };
            let source_info = self.body().source_info(location);
            self.report_assert_as_lint(
                source_info,
                AssertLint::UnconditionalPanic(source_info.span, msg),
            );
        }

        None
    }

    fn ensure_not_propagated(&self, local: Local) {
        if cfg!(debug_assertions) {
            assert!(
                self.get_const(local.into()).is_none()
                    || self
                        .layout_of(self.local_decls()[local].ty)
                        .map_or(true, |layout| layout.is_zst()),
                "failed to remove values for `{local:?}`, value={:?}",
                self.get_const(local.into()),
            )
        }
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

    fn visit_constant(&mut self, constant: &Constant<'tcx>, location: Location) {
        trace!("visit_constant: {:?}", constant);
        self.super_constant(constant, location);
        self.eval_constant(constant, location);
    }

    fn visit_assign(&mut self, place: &Place<'tcx>, rvalue: &Rvalue<'tcx>, location: Location) {
        self.super_assign(place, rvalue, location);

        let Some(()) = self.check_rvalue(rvalue, location) else { return };

        match self.ecx.machine.can_const_prop[place.local] {
            // Do nothing if the place is indirect.
            _ if place.is_indirect() => {}
            ConstPropMode::NoPropagation => self.ensure_not_propagated(place.local),
            ConstPropMode::OnlyInsideOwnBlock | ConstPropMode::FullConstProp => {
                if self
                    .use_ecx(location, |this| this.ecx.eval_rvalue_into_place(rvalue, *place))
                    .is_none()
                {
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
                    Self::remove_const(&mut self.ecx, place.local);
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
            StatementKind::SetDiscriminant { ref place, .. } => {
                match self.ecx.machine.can_const_prop[place.local] {
                    // Do nothing if the place is indirect.
                    _ if place.is_indirect() => {}
                    ConstPropMode::NoPropagation => self.ensure_not_propagated(place.local),
                    ConstPropMode::FullConstProp | ConstPropMode::OnlyInsideOwnBlock => {
                        if self.use_ecx(location, |this| this.ecx.statement(statement)).is_some() {
                            trace!("propped discriminant into {:?}", place);
                        } else {
                            Self::remove_const(&mut self.ecx, place.local);
                        }
                    }
                }
            }
            StatementKind::StorageLive(local) => {
                let frame = self.ecx.frame_mut();
                frame.locals[local].value =
                    LocalValue::Live(interpret::Operand::Immediate(interpret::Immediate::Uninit));
            }
            StatementKind::StorageDead(local) => {
                let frame = self.ecx.frame_mut();
                frame.locals[local].value = LocalValue::Dead;
            }
            _ => {}
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        self.super_terminator(terminator, location);
        match &terminator.kind {
            TerminatorKind::Assert { expected, ref msg, ref cond, .. } => {
                self.check_assertion(*expected, msg, cond, location);
            }
            TerminatorKind::SwitchInt { ref discr, ref targets } => {
                if let Some(ref value) = self.eval_operand(&discr, location)
                  && let Some(value_const) = self.use_ecx(location, |this| this.ecx.read_scalar(&value))
                  && let Ok(constant) = value_const.try_to_int()
                  && let Ok(constant) = constant.to_bits(constant.size())
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
            | TerminatorKind::Resume
            | TerminatorKind::Terminate
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::Drop { .. }
            | TerminatorKind::Yield { .. }
            | TerminatorKind::GeneratorDrop
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
            std::mem::take(&mut self.ecx.machine.written_only_inside_own_block_locals);

        // This loop can get very hot for some bodies: it check each local in each bb.
        // To avoid this quadratic behaviour, we only clear the locals that were modified inside
        // the current block.
        for local in written_only_inside_own_block_locals.drain() {
            debug_assert_eq!(
                self.ecx.machine.can_const_prop[local],
                ConstPropMode::OnlyInsideOwnBlock
            );
            Self::remove_const(&mut self.ecx, local);
        }
        self.ecx.machine.written_only_inside_own_block_locals =
            written_only_inside_own_block_locals;

        if cfg!(debug_assertions) {
            for (local, &mode) in self.ecx.machine.can_const_prop.iter_enumerated() {
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
