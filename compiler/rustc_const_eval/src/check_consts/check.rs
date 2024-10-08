//! The `Visitor` responsible for actually checking a `mir::Body` for invalid operations.

use std::assert_matches::assert_matches;
use std::borrow::Cow;
use std::mem;
use std::ops::Deref;

use rustc_errors::{Diag, ErrorGuaranteed};
use rustc_hir::def_id::DefId;
use rustc_hir::{self as hir, LangItem};
use rustc_index::bit_set::BitSet;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::ObligationCause;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::span_bug;
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::{self, Instance, InstanceKind, Ty, TypeVisitableExt};
use rustc_mir_dataflow::Analysis;
use rustc_mir_dataflow::impls::MaybeStorageLive;
use rustc_mir_dataflow::storage::always_storage_live_locals;
use rustc_span::{DUMMY_SP, Span, Symbol, sym};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::traits::{self, ObligationCauseCode, ObligationCtxt};
use tracing::{debug, instrument, trace};

use super::ops::{self, NonConstOp, Status};
use super::qualifs::{self, HasMutInterior, NeedsDrop, NeedsNonConstDrop};
use super::resolver::FlowSensitiveAnalysis;
use super::{ConstCx, Qualif};
use crate::const_eval::is_unstable_const_fn;
use crate::errors::UnstableInStable;

type QualifResults<'mir, 'tcx, Q> =
    rustc_mir_dataflow::ResultsCursor<'mir, 'tcx, FlowSensitiveAnalysis<'mir, 'mir, 'tcx, Q>>;

#[derive(Default)]
pub(crate) struct Qualifs<'mir, 'tcx> {
    has_mut_interior: Option<QualifResults<'mir, 'tcx, HasMutInterior>>,
    needs_drop: Option<QualifResults<'mir, 'tcx, NeedsDrop>>,
    needs_non_const_drop: Option<QualifResults<'mir, 'tcx, NeedsNonConstDrop>>,
}

impl<'mir, 'tcx> Qualifs<'mir, 'tcx> {
    /// Returns `true` if `local` is `NeedsDrop` at the given `Location`.
    ///
    /// Only updates the cursor if absolutely necessary
    fn needs_drop(
        &mut self,
        ccx: &'mir ConstCx<'mir, 'tcx>,
        local: Local,
        location: Location,
    ) -> bool {
        let ty = ccx.body.local_decls[local].ty;
        // Peeking into opaque types causes cycles if the current function declares said opaque
        // type. Thus we avoid short circuiting on the type and instead run the more expensive
        // analysis that looks at the actual usage within this function
        if !ty.has_opaque_types() && !NeedsDrop::in_any_value_of_ty(ccx, ty) {
            return false;
        }

        let needs_drop = self.needs_drop.get_or_insert_with(|| {
            let ConstCx { tcx, body, .. } = *ccx;

            FlowSensitiveAnalysis::new(NeedsDrop, ccx)
                .into_engine(tcx, body)
                .iterate_to_fixpoint()
                .into_results_cursor(body)
        });

        needs_drop.seek_before_primary_effect(location);
        needs_drop.get().contains(local)
    }

    /// Returns `true` if `local` is `NeedsNonConstDrop` at the given `Location`.
    ///
    /// Only updates the cursor if absolutely necessary
    pub(crate) fn needs_non_const_drop(
        &mut self,
        ccx: &'mir ConstCx<'mir, 'tcx>,
        local: Local,
        location: Location,
    ) -> bool {
        let ty = ccx.body.local_decls[local].ty;
        // Peeking into opaque types causes cycles if the current function declares said opaque
        // type. Thus we avoid short circuiting on the type and instead run the more expensive
        // analysis that looks at the actual usage within this function
        if !ty.has_opaque_types() && !NeedsNonConstDrop::in_any_value_of_ty(ccx, ty) {
            return false;
        }

        let needs_non_const_drop = self.needs_non_const_drop.get_or_insert_with(|| {
            let ConstCx { tcx, body, .. } = *ccx;

            FlowSensitiveAnalysis::new(NeedsNonConstDrop, ccx)
                .into_engine(tcx, body)
                .iterate_to_fixpoint()
                .into_results_cursor(body)
        });

        needs_non_const_drop.seek_before_primary_effect(location);
        needs_non_const_drop.get().contains(local)
    }

    /// Returns `true` if `local` is `HasMutInterior` at the given `Location`.
    ///
    /// Only updates the cursor if absolutely necessary.
    fn has_mut_interior(
        &mut self,
        ccx: &'mir ConstCx<'mir, 'tcx>,
        local: Local,
        location: Location,
    ) -> bool {
        let ty = ccx.body.local_decls[local].ty;
        // Peeking into opaque types causes cycles if the current function declares said opaque
        // type. Thus we avoid short circuiting on the type and instead run the more expensive
        // analysis that looks at the actual usage within this function
        if !ty.has_opaque_types() && !HasMutInterior::in_any_value_of_ty(ccx, ty) {
            return false;
        }

        let has_mut_interior = self.has_mut_interior.get_or_insert_with(|| {
            let ConstCx { tcx, body, .. } = *ccx;

            FlowSensitiveAnalysis::new(HasMutInterior, ccx)
                .into_engine(tcx, body)
                .iterate_to_fixpoint()
                .into_results_cursor(body)
        });

        has_mut_interior.seek_before_primary_effect(location);
        has_mut_interior.get().contains(local)
    }

    fn in_return_place(
        &mut self,
        ccx: &'mir ConstCx<'mir, 'tcx>,
        tainted_by_errors: Option<ErrorGuaranteed>,
    ) -> ConstQualifs {
        // FIXME(explicit_tail_calls): uhhhh I think we can return without return now, does it change anything

        // Find the `Return` terminator if one exists.
        //
        // If no `Return` terminator exists, this MIR is divergent. Just return the conservative
        // qualifs for the return type.
        let return_block = ccx
            .body
            .basic_blocks
            .iter_enumerated()
            .find(|(_, block)| matches!(block.terminator().kind, TerminatorKind::Return))
            .map(|(bb, _)| bb);

        let Some(return_block) = return_block else {
            return qualifs::in_any_value_of_ty(ccx, ccx.body.return_ty(), tainted_by_errors);
        };

        let return_loc = ccx.body.terminator_loc(return_block);

        ConstQualifs {
            needs_drop: self.needs_drop(ccx, RETURN_PLACE, return_loc),
            needs_non_const_drop: self.needs_non_const_drop(ccx, RETURN_PLACE, return_loc),
            has_mut_interior: self.has_mut_interior(ccx, RETURN_PLACE, return_loc),
            tainted_by_errors,
        }
    }
}

pub struct Checker<'mir, 'tcx> {
    ccx: &'mir ConstCx<'mir, 'tcx>,
    qualifs: Qualifs<'mir, 'tcx>,

    /// The span of the current statement.
    span: Span,

    /// A set that stores for each local whether it is "transient", i.e. guaranteed to be dead
    /// when this MIR body returns.
    transient_locals: Option<BitSet<Local>>,

    error_emitted: Option<ErrorGuaranteed>,
    secondary_errors: Vec<Diag<'tcx>>,
}

impl<'mir, 'tcx> Deref for Checker<'mir, 'tcx> {
    type Target = ConstCx<'mir, 'tcx>;

    fn deref(&self) -> &Self::Target {
        self.ccx
    }
}

impl<'mir, 'tcx> Checker<'mir, 'tcx> {
    pub fn new(ccx: &'mir ConstCx<'mir, 'tcx>) -> Self {
        Checker {
            span: ccx.body.span,
            ccx,
            qualifs: Default::default(),
            transient_locals: None,
            error_emitted: None,
            secondary_errors: Vec::new(),
        }
    }

    pub fn check_body(&mut self) {
        let ConstCx { tcx, body, .. } = *self.ccx;
        let def_id = self.ccx.def_id();

        // `async` functions cannot be `const fn`. This is checked during AST lowering, so there's
        // no need to emit duplicate errors here.
        if self.ccx.is_async() || body.coroutine.is_some() {
            tcx.dcx().span_delayed_bug(body.span, "`async` functions cannot be `const fn`");
            return;
        }

        if !tcx.has_attr(def_id, sym::rustc_do_not_const_check) {
            self.visit_body(body);
        }

        // If we got through const-checking without emitting any "primary" errors, emit any
        // "secondary" errors if they occurred. Otherwise, cancel the "secondary" errors.
        let secondary_errors = mem::take(&mut self.secondary_errors);
        if self.error_emitted.is_none() {
            for error in secondary_errors {
                self.error_emitted = Some(error.emit());
            }
        } else {
            assert!(self.tcx.dcx().has_errors().is_some());
            for error in secondary_errors {
                error.cancel();
            }
        }
    }

    fn local_is_transient(&mut self, local: Local) -> bool {
        let ccx = self.ccx;
        self.transient_locals
            .get_or_insert_with(|| {
                // A local is "transient" if it is guaranteed dead at all `Return`.
                // So first compute the say of "maybe live" locals at each program point.
                let always_live_locals = &always_storage_live_locals(&ccx.body);
                let mut maybe_storage_live =
                    MaybeStorageLive::new(Cow::Borrowed(always_live_locals))
                        .into_engine(ccx.tcx, &ccx.body)
                        .iterate_to_fixpoint()
                        .into_results_cursor(&ccx.body);

                // And then check all `Return` in the MIR, and if a local is "maybe live" at a
                // `Return` then it is definitely not transient.
                let mut transient = BitSet::new_filled(ccx.body.local_decls.len());
                // Make sure to only visit reachable blocks, the dataflow engine can ICE otherwise.
                for (bb, data) in traversal::reachable(&ccx.body) {
                    if matches!(data.terminator().kind, TerminatorKind::Return) {
                        let location = ccx.body.terminator_loc(bb);
                        maybe_storage_live.seek_after_primary_effect(location);
                        // If a local may be live here, it is definitely not transient.
                        transient.subtract(maybe_storage_live.get());
                    }
                }

                transient
            })
            .contains(local)
    }

    pub fn qualifs_in_return_place(&mut self) -> ConstQualifs {
        self.qualifs.in_return_place(self.ccx, self.error_emitted)
    }

    /// Emits an error if an expression cannot be evaluated in the current context.
    pub fn check_op(&mut self, op: impl NonConstOp<'tcx>) {
        self.check_op_spanned(op, self.span);
    }

    /// Emits an error at the given `span` if an expression cannot be evaluated in the current
    /// context.
    pub fn check_op_spanned<O: NonConstOp<'tcx>>(&mut self, op: O, span: Span) {
        let gate = match op.status_in_item(self.ccx) {
            Status::Allowed => return,

            Status::Unstable(gate) if self.tcx.features().active(gate) => {
                let unstable_in_stable = self.ccx.is_const_stable_const_fn()
                    && !super::rustc_allow_const_fn_unstable(self.tcx, self.def_id(), gate);
                if unstable_in_stable {
                    emit_unstable_in_stable_error(self.ccx, span, gate);
                }

                return;
            }

            Status::Unstable(gate) => Some(gate),
            Status::Forbidden => None,
        };

        if self.tcx.sess.opts.unstable_opts.unleash_the_miri_inside_of_you {
            self.tcx.sess.miri_unleashed_feature(span, gate);
            return;
        }

        let err = op.build_error(self.ccx, span);
        assert!(err.is_error());

        match op.importance() {
            ops::DiagImportance::Primary => {
                let reported = err.emit();
                self.error_emitted = Some(reported);
            }

            ops::DiagImportance::Secondary => self.secondary_errors.push(err),
        }
    }

    fn check_static(&mut self, def_id: DefId, span: Span) {
        if self.tcx.is_thread_local_static(def_id) {
            self.tcx.dcx().span_bug(span, "tls access is checked in `Rvalue::ThreadLocalRef`");
        }
        if let Some(def_id) = def_id.as_local()
            && let Err(guar) = self.tcx.at(span).check_well_formed(hir::OwnerId { def_id })
        {
            self.error_emitted = Some(guar);
        }
    }

    /// Returns whether this place can possibly escape the evaluation of the current const/static
    /// initializer. The check assumes that all already existing pointers and references point to
    /// non-escaping places.
    fn place_may_escape(&mut self, place: &Place<'_>) -> bool {
        let is_transient = match self.const_kind() {
            // In a const fn all borrows are transient or point to the places given via
            // references in the arguments (so we already checked them with
            // TransientMutBorrow/MutBorrow as appropriate).
            // The borrow checker guarantees that no new non-transient borrows are created.
            // NOTE: Once we have heap allocations during CTFE we need to figure out
            // how to prevent `const fn` to create long-lived allocations that point
            // to mutable memory.
            hir::ConstContext::ConstFn => true,
            _ => {
                // For indirect places, we are not creating a new permanent borrow, it's just as
                // transient as the already existing one. For reborrowing references this is handled
                // at the top of `visit_rvalue`, but for raw pointers we handle it here.
                // Pointers/references to `static mut` and cases where the `*` is not the first
                // projection also end up here.
                // Locals with StorageDead do not live beyond the evaluation and can
                // thus safely be borrowed without being able to be leaked to the final
                // value of the constant.
                // Note: This is only sound if every local that has a `StorageDead` has a
                // `StorageDead` in every control flow path leading to a `return` terminator.
                // If anything slips through, there's no safety net -- safe code can create
                // references to variants of `!Freeze` enums as long as that variant is `Freeze`, so
                // interning can't protect us here. (There *is* a safety net for mutable references
                // though, interning will ICE if we miss something here.)
                place.is_indirect() || self.local_is_transient(place.local)
            }
        };
        // Transient places cannot possibly escape because the place doesn't exist any more at the
        // end of evaluation.
        !is_transient
    }
}

impl<'tcx> Visitor<'tcx> for Checker<'_, 'tcx> {
    fn visit_basic_block_data(&mut self, bb: BasicBlock, block: &BasicBlockData<'tcx>) {
        trace!("visit_basic_block_data: bb={:?} is_cleanup={:?}", bb, block.is_cleanup);

        // We don't const-check basic blocks on the cleanup path since we never unwind during
        // const-eval: a panic causes an immediate compile error. In other words, cleanup blocks
        // are unreachable during const-eval.
        //
        // We can't be more conservative (e.g., by const-checking cleanup blocks anyways) because
        // locals that would never be dropped during normal execution are sometimes dropped during
        // unwinding, which means backwards-incompatible live-drop errors.
        if block.is_cleanup {
            return;
        }

        self.super_basic_block_data(bb, block);
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        trace!("visit_rvalue: rvalue={:?} location={:?}", rvalue, location);

        self.super_rvalue(rvalue, location);

        match rvalue {
            Rvalue::ThreadLocalRef(_) => self.check_op(ops::ThreadLocalAccess),

            Rvalue::Use(_)
            | Rvalue::CopyForDeref(..)
            | Rvalue::Repeat(..)
            | Rvalue::Discriminant(..)
            | Rvalue::Len(_) => {}

            Rvalue::Aggregate(kind, ..) => {
                if let AggregateKind::Coroutine(def_id, ..) = kind.as_ref()
                    && let Some(
                        coroutine_kind @ hir::CoroutineKind::Desugared(
                            hir::CoroutineDesugaring::Async,
                            _,
                        ),
                    ) = self.tcx.coroutine_kind(def_id)
                {
                    self.check_op(ops::Coroutine(coroutine_kind));
                }
            }

            Rvalue::Ref(_, BorrowKind::Mut { .. }, place)
            | Rvalue::RawPtr(Mutability::Mut, place) => {
                // Inside mutable statics, we allow arbitrary mutable references.
                // We've allowed `static mut FOO = &mut [elements];` for a long time (the exact
                // reasons why are lost to history), and there is no reason to restrict that to
                // arrays and slices.
                let is_allowed =
                    self.const_kind() == hir::ConstContext::Static(hir::Mutability::Mut);

                if !is_allowed && self.place_may_escape(place) {
                    self.check_op(ops::EscapingMutBorrow(if matches!(rvalue, Rvalue::Ref(..)) {
                        hir::BorrowKind::Ref
                    } else {
                        hir::BorrowKind::Raw
                    }));
                }
            }

            Rvalue::Ref(_, BorrowKind::Shared | BorrowKind::Fake(_), place)
            | Rvalue::RawPtr(Mutability::Not, place) => {
                let borrowed_place_has_mut_interior = qualifs::in_place::<HasMutInterior, _>(
                    self.ccx,
                    &mut |local| self.qualifs.has_mut_interior(self.ccx, local, location),
                    place.as_ref(),
                );

                if borrowed_place_has_mut_interior && self.place_may_escape(place) {
                    self.check_op(ops::EscapingCellBorrow);
                }
            }

            Rvalue::Cast(
                CastKind::PointerCoercion(
                    PointerCoercion::MutToConstPointer
                    | PointerCoercion::ArrayToPointer
                    | PointerCoercion::UnsafeFnPointer
                    | PointerCoercion::ClosureFnPointer(_)
                    | PointerCoercion::ReifyFnPointer,
                    _,
                ),
                _,
                _,
            ) => {
                // These are all okay; they only change the type, not the data.
            }

            Rvalue::Cast(
                CastKind::PointerCoercion(PointerCoercion::Unsize | PointerCoercion::DynStar, _),
                _,
                _,
            ) => {
                // Unsizing and `dyn*` coercions are implemented for CTFE.
            }

            Rvalue::Cast(CastKind::PointerExposeProvenance, _, _) => {
                self.check_op(ops::RawPtrToIntCast);
            }
            Rvalue::Cast(CastKind::PointerWithExposedProvenance, _, _) => {
                // Since no pointer can ever get exposed (rejected above), this is easy to support.
            }

            Rvalue::Cast(_, _, _) => {}

            Rvalue::NullaryOp(
                NullOp::SizeOf | NullOp::AlignOf | NullOp::OffsetOf(_) | NullOp::UbChecks,
                _,
            ) => {}
            Rvalue::ShallowInitBox(_, _) => {}

            Rvalue::UnaryOp(_, operand) => {
                let ty = operand.ty(self.body, self.tcx);
                if is_int_bool_float_or_char(ty) {
                    // Int, bool, float, and char operations are fine.
                } else {
                    span_bug!(self.span, "non-primitive type in `Rvalue::UnaryOp`: {:?}", ty);
                }
            }

            Rvalue::BinaryOp(op, box (lhs, rhs)) => {
                let lhs_ty = lhs.ty(self.body, self.tcx);
                let rhs_ty = rhs.ty(self.body, self.tcx);

                if is_int_bool_float_or_char(lhs_ty) && is_int_bool_float_or_char(rhs_ty) {
                    // Int, bool, float, and char operations are fine.
                } else if lhs_ty.is_fn_ptr() || lhs_ty.is_unsafe_ptr() {
                    assert_matches!(
                        op,
                        BinOp::Eq
                            | BinOp::Ne
                            | BinOp::Le
                            | BinOp::Lt
                            | BinOp::Ge
                            | BinOp::Gt
                            | BinOp::Offset
                    );

                    self.check_op(ops::RawPtrComparison);
                } else {
                    span_bug!(
                        self.span,
                        "non-primitive type in `Rvalue::BinaryOp`: {:?} ⚬ {:?}",
                        lhs_ty,
                        rhs_ty
                    );
                }
            }
        }
    }

    fn visit_operand(&mut self, op: &Operand<'tcx>, location: Location) {
        self.super_operand(op, location);
        if let Operand::Constant(c) = op {
            if let Some(def_id) = c.check_static_ptr(self.tcx) {
                self.check_static(def_id, self.span);
            }
        }
    }

    fn visit_source_info(&mut self, source_info: &SourceInfo) {
        trace!("visit_source_info: source_info={:?}", source_info);
        self.span = source_info.span;
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        trace!("visit_statement: statement={:?} location={:?}", statement, location);

        self.super_statement(statement, location);

        match statement.kind {
            StatementKind::Assign(..)
            | StatementKind::SetDiscriminant { .. }
            | StatementKind::Deinit(..)
            | StatementKind::FakeRead(..)
            | StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Retag { .. }
            | StatementKind::PlaceMention(..)
            | StatementKind::AscribeUserType(..)
            | StatementKind::Coverage(..)
            | StatementKind::Intrinsic(..)
            | StatementKind::ConstEvalCounter
            | StatementKind::Nop => {}
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        self.super_terminator(terminator, location);

        match &terminator.kind {
            TerminatorKind::Call { func, args, fn_span, .. }
            | TerminatorKind::TailCall { func, args, fn_span, .. } => {
                let call_source = match terminator.kind {
                    TerminatorKind::Call { call_source, .. } => call_source,
                    TerminatorKind::TailCall { .. } => CallSource::Normal,
                    _ => unreachable!(),
                };

                let ConstCx { tcx, body, param_env, .. } = *self.ccx;
                let caller = self.def_id();

                let fn_ty = func.ty(body, tcx);

                let (mut callee, mut fn_args) = match *fn_ty.kind() {
                    ty::FnDef(def_id, fn_args) => (def_id, fn_args),

                    ty::FnPtr(..) => {
                        self.check_op(ops::FnCallIndirect);
                        return;
                    }
                    _ => {
                        span_bug!(terminator.source_info.span, "invalid callee of type {:?}", fn_ty)
                    }
                };

                // Check that all trait bounds that are marked as `~const` can be satisfied.
                //
                // Typeck only does a "non-const" check since it operates on HIR and cannot distinguish
                // which path expressions are getting called on and which path expressions are only used
                // as function pointers. This is required for correctness.
                let infcx = tcx.infer_ctxt().build();
                let ocx = ObligationCtxt::new_with_diagnostics(&infcx);

                let predicates = tcx.predicates_of(callee).instantiate(tcx, fn_args);
                let cause = ObligationCause::new(
                    terminator.source_info.span,
                    self.body.source.def_id().expect_local(),
                    ObligationCauseCode::WhereClause(callee, DUMMY_SP),
                );
                let normalized_predicates = ocx.normalize(&cause, param_env, predicates);
                ocx.register_obligations(traits::predicates_for_generics(
                    |_, _| cause.clone(),
                    self.param_env,
                    normalized_predicates,
                ));

                let errors = ocx.select_all_or_error();
                if !errors.is_empty() {
                    infcx.err_ctxt().report_fulfillment_errors(errors);
                }

                let mut is_trait = false;
                // Attempting to call a trait method?
                if tcx.trait_of_item(callee).is_some() {
                    trace!("attempting to call a trait method");
                    // trait method calls are only permitted when `effects` is enabled.
                    // we don't error, since that is handled by typeck. We try to resolve
                    // the trait into the concrete method, and uses that for const stability
                    // checks.
                    // FIXME(effects) we might consider moving const stability checks to typeck as well.
                    if tcx.features().effects {
                        is_trait = true;

                        if let Ok(Some(instance)) =
                            Instance::try_resolve(tcx, param_env, callee, fn_args)
                            && let InstanceKind::Item(def) = instance.def
                        {
                            // Resolve a trait method call to its concrete implementation, which may be in a
                            // `const` trait impl. This is only used for the const stability check below, since
                            // we want to look at the concrete impl's stability.
                            fn_args = instance.args;
                            callee = def;
                        }
                    } else {
                        self.check_op(ops::FnCallNonConst {
                            caller,
                            callee,
                            args: fn_args,
                            span: *fn_span,
                            call_source,
                            feature: Some(if tcx.features().const_trait_impl {
                                sym::effects
                            } else {
                                sym::const_trait_impl
                            }),
                        });
                        return;
                    }
                }

                // At this point, we are calling a function, `callee`, whose `DefId` is known...

                // `begin_panic` and `#[rustc_const_panic_str]` functions accept generic
                // types other than str. Check to enforce that only str can be used in
                // const-eval.

                // const-eval of the `begin_panic` fn assumes the argument is `&str`
                if tcx.is_lang_item(callee, LangItem::BeginPanic) {
                    match args[0].node.ty(&self.ccx.body.local_decls, tcx).kind() {
                        ty::Ref(_, ty, _) if ty.is_str() => return,
                        _ => self.check_op(ops::PanicNonStr),
                    }
                }

                // const-eval of `#[rustc_const_panic_str]` functions assumes the argument is `&&str`
                if tcx.has_attr(callee, sym::rustc_const_panic_str) {
                    match args[0].node.ty(&self.ccx.body.local_decls, tcx).kind() {
                        ty::Ref(_, ty, _) if matches!(ty.kind(), ty::Ref(_, ty, _) if ty.is_str()) =>
                        {
                            return;
                        }
                        _ => self.check_op(ops::PanicNonStr),
                    }
                }

                if tcx.is_lang_item(callee, LangItem::ExchangeMalloc) {
                    self.check_op(ops::HeapAllocation);
                    return;
                }

                if !tcx.is_const_fn_raw(callee) && !is_trait {
                    self.check_op(ops::FnCallNonConst {
                        caller,
                        callee,
                        args: fn_args,
                        span: *fn_span,
                        call_source,
                        feature: None,
                    });
                    return;
                }

                // If the `const fn` we are trying to call is not const-stable, ensure that we have
                // the proper feature gate enabled.
                if let Some((gate, implied_by)) = is_unstable_const_fn(tcx, callee) {
                    trace!(?gate, "calling unstable const fn");
                    if self.span.allows_unstable(gate) {
                        return;
                    }
                    if let Some(implied_by_gate) = implied_by
                        && self.span.allows_unstable(implied_by_gate)
                    {
                        return;
                    }

                    // Calling an unstable function *always* requires that the corresponding gate
                    // (or implied gate) be enabled, even if the function has
                    // `#[rustc_allow_const_fn_unstable(the_gate)]`.
                    let gate_declared = |gate| tcx.features().declared(gate);
                    let feature_gate_declared = gate_declared(gate);
                    let implied_gate_declared = implied_by.is_some_and(gate_declared);
                    if !feature_gate_declared && !implied_gate_declared {
                        self.check_op(ops::FnCallUnstable(callee, Some(gate)));
                        return;
                    }

                    // If this crate is not using stability attributes, or the caller is not claiming to be a
                    // stable `const fn`, that is all that is required.
                    if !self.ccx.is_const_stable_const_fn() {
                        trace!("crate not using stability attributes or caller not stably const");
                        return;
                    }

                    // Otherwise, we are something const-stable calling a const-unstable fn.
                    if super::rustc_allow_const_fn_unstable(tcx, caller, gate) {
                        trace!("rustc_allow_const_fn_unstable gate active");
                        return;
                    }

                    self.check_op(ops::FnCallUnstable(callee, Some(gate)));
                    return;
                }

                // FIXME(ecstaticmorse); For compatibility, we consider `unstable` callees that
                // have no `rustc_const_stable` attributes to be const-unstable as well. This
                // should be fixed later.
                let callee_is_unstable_unmarked = tcx.lookup_const_stability(callee).is_none()
                    && tcx.lookup_stability(callee).is_some_and(|s| s.is_unstable());
                if callee_is_unstable_unmarked {
                    trace!("callee_is_unstable_unmarked");
                    // We do not use `const` modifiers for intrinsic "functions", as intrinsics are
                    // `extern` functions, and these have no way to get marked `const`. So instead we
                    // use `rustc_const_(un)stable` attributes to mean that the intrinsic is `const`
                    if self.ccx.is_const_stable_const_fn() || tcx.intrinsic(callee).is_some() {
                        self.check_op(ops::FnCallUnstable(callee, None));
                        return;
                    }
                }
                trace!("permitting call");
            }

            // Forbid all `Drop` terminators unless the place being dropped is a local with no
            // projections that cannot be `NeedsNonConstDrop`.
            TerminatorKind::Drop { place: dropped_place, .. } => {
                // If we are checking live drops after drop-elaboration, don't emit duplicate
                // errors here.
                if super::post_drop_elaboration::checking_enabled(self.ccx) {
                    return;
                }

                let mut err_span = self.span;
                let ty_of_dropped_place = dropped_place.ty(self.body, self.tcx).ty;

                let ty_needs_non_const_drop =
                    qualifs::NeedsNonConstDrop::in_any_value_of_ty(self.ccx, ty_of_dropped_place);

                debug!(?ty_of_dropped_place, ?ty_needs_non_const_drop);

                if !ty_needs_non_const_drop {
                    return;
                }

                let needs_non_const_drop = if let Some(local) = dropped_place.as_local() {
                    // Use the span where the local was declared as the span of the drop error.
                    err_span = self.body.local_decls[local].source_info.span;
                    self.qualifs.needs_non_const_drop(self.ccx, local, location)
                } else {
                    true
                };

                if needs_non_const_drop {
                    self.check_op_spanned(
                        ops::LiveDrop {
                            dropped_at: Some(terminator.source_info.span),
                            dropped_ty: ty_of_dropped_place,
                        },
                        err_span,
                    );
                }
            }

            TerminatorKind::InlineAsm { .. } => self.check_op(ops::InlineAsm),

            TerminatorKind::Yield { .. } => self.check_op(ops::Coroutine(
                self.tcx
                    .coroutine_kind(self.body.source.def_id())
                    .expect("Only expected to have a yield in a coroutine"),
            )),

            TerminatorKind::CoroutineDrop => {
                span_bug!(
                    self.body.source_info(location).span,
                    "We should not encounter TerminatorKind::CoroutineDrop after coroutine transform"
                );
            }

            TerminatorKind::UnwindTerminate(_) => {
                // Cleanup blocks are skipped for const checking (see `visit_basic_block_data`).
                span_bug!(self.span, "`Terminate` terminator outside of cleanup block")
            }

            TerminatorKind::Assert { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::Goto { .. }
            | TerminatorKind::UnwindResume
            | TerminatorKind::Return
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Unreachable => {}
        }
    }
}

fn is_int_bool_float_or_char(ty: Ty<'_>) -> bool {
    ty.is_bool() || ty.is_integral() || ty.is_char() || ty.is_floating_point()
}

fn emit_unstable_in_stable_error(ccx: &ConstCx<'_, '_>, span: Span, gate: Symbol) {
    let attr_span = ccx.tcx.def_span(ccx.def_id()).shrink_to_lo();

    ccx.dcx().emit_err(UnstableInStable { gate: gate.to_string(), span, attr_span });
}
