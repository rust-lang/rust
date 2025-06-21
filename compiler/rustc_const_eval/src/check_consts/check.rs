//! The `Visitor` responsible for actually checking a `mir::Body` for invalid operations.

use std::assert_matches::assert_matches;
use std::borrow::Cow;
use std::mem;
use std::num::NonZero;
use std::ops::Deref;

use rustc_attr_data_structures as attrs;
use rustc_errors::{Diag, ErrorGuaranteed};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::{self as hir, LangItem};
use rustc_index::bit_set::DenseBitSet;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::span_bug;
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::{self, Ty, TypeVisitableExt};
use rustc_mir_dataflow::Analysis;
use rustc_mir_dataflow::impls::{MaybeStorageLive, always_storage_live_locals};
use rustc_span::{Span, Symbol, sym};
use rustc_trait_selection::traits::{
    Obligation, ObligationCause, ObligationCauseCode, ObligationCtxt,
};
use tracing::{instrument, trace};

use super::ops::{self, NonConstOp, Status};
use super::qualifs::{self, HasMutInterior, NeedsDrop, NeedsNonConstDrop};
use super::resolver::FlowSensitiveAnalysis;
use super::{ConstCx, Qualif};
use crate::check_consts::is_fn_or_trait_safe_to_expose_on_stable;
use crate::errors;

type QualifResults<'mir, 'tcx, Q> =
    rustc_mir_dataflow::ResultsCursor<'mir, 'tcx, FlowSensitiveAnalysis<'mir, 'tcx, Q>>;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum ConstConditionsHold {
    Yes,
    No,
}

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
    pub(crate) fn needs_drop(
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
                .iterate_to_fixpoint(tcx, body, None)
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
                .iterate_to_fixpoint(tcx, body, None)
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
                .iterate_to_fixpoint(tcx, body, None)
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
    transient_locals: Option<DenseBitSet<Local>>,

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
                        .iterate_to_fixpoint(ccx.tcx, &ccx.body, None)
                        .into_results_cursor(&ccx.body);

                // And then check all `Return` in the MIR, and if a local is "maybe live" at a
                // `Return` then it is definitely not transient.
                let mut transient = DenseBitSet::new_filled(ccx.body.local_decls.len());
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
            Status::Unstable {
                gate,
                safe_to_expose_on_stable,
                is_function_call,
                gate_already_checked,
            } if gate_already_checked || self.tcx.features().enabled(gate) => {
                if gate_already_checked {
                    assert!(
                        !safe_to_expose_on_stable,
                        "setting `gate_already_checked` without `safe_to_expose_on_stable` makes no sense"
                    );
                }
                // Generally this is allowed since the feature gate is enabled -- except
                // if this function wants to be safe-to-expose-on-stable.
                if !safe_to_expose_on_stable
                    && self.enforce_recursive_const_stability()
                    && !super::rustc_allow_const_fn_unstable(self.tcx, self.def_id(), gate)
                {
                    emit_unstable_in_stable_exposed_error(self.ccx, span, gate, is_function_call);
                }

                return;
            }

            Status::Unstable { gate, .. } => Some(gate),
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

            ops::DiagImportance::Secondary => {
                self.secondary_errors.push(err);
                self.tcx.dcx().span_delayed_bug(
                    span,
                    "compilation must fail when there is a secondary const checker error",
                );
            }
        }
    }

    fn check_static(&mut self, def_id: DefId, span: Span) {
        if self.tcx.is_thread_local_static(def_id) {
            self.tcx.dcx().span_bug(span, "tls access is checked in `Rvalue::ThreadLocalRef`");
        }
        if let Some(def_id) = def_id.as_local()
            && let Err(guar) = self.tcx.ensure_ok().check_well_formed(hir::OwnerId { def_id })
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
                // transient as the already existing one.
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

    /// Returns whether there are const-conditions.
    fn revalidate_conditional_constness(
        &mut self,
        callee: DefId,
        callee_args: ty::GenericArgsRef<'tcx>,
        call_span: Span,
    ) -> Option<ConstConditionsHold> {
        let tcx = self.tcx;
        if !tcx.is_conditionally_const(callee) {
            return None;
        }

        let const_conditions = tcx.const_conditions(callee).instantiate(tcx, callee_args);
        if const_conditions.is_empty() {
            return None;
        }

        let (infcx, param_env) = tcx.infer_ctxt().build_with_typing_env(self.body.typing_env(tcx));
        let ocx = ObligationCtxt::new(&infcx);

        let body_id = self.body.source.def_id().expect_local();
        let host_polarity = match self.const_kind() {
            hir::ConstContext::ConstFn => ty::BoundConstness::Maybe,
            hir::ConstContext::Static(_) | hir::ConstContext::Const { .. } => {
                ty::BoundConstness::Const
            }
        };
        let const_conditions =
            ocx.normalize(&ObligationCause::misc(call_span, body_id), param_env, const_conditions);
        ocx.register_obligations(const_conditions.into_iter().map(|(trait_ref, span)| {
            Obligation::new(
                tcx,
                ObligationCause::new(
                    call_span,
                    body_id,
                    ObligationCauseCode::WhereClause(callee, span),
                ),
                param_env,
                trait_ref.to_host_effect_clause(tcx, host_polarity),
            )
        }));

        let errors = ocx.select_all_or_error();
        if errors.is_empty() {
            Some(ConstConditionsHold::Yes)
        } else {
            tcx.dcx()
                .span_delayed_bug(call_span, "this should have reported a ~const error in HIR");
            Some(ConstConditionsHold::No)
        }
    }

    pub fn check_drop_terminator(
        &mut self,
        dropped_place: Place<'tcx>,
        location: Location,
        terminator_span: Span,
    ) {
        let ty_of_dropped_place = dropped_place.ty(self.body, self.tcx).ty;

        let needs_drop = if let Some(local) = dropped_place.as_local() {
            self.qualifs.needs_drop(self.ccx, local, location)
        } else {
            qualifs::NeedsDrop::in_any_value_of_ty(self.ccx, ty_of_dropped_place)
        };
        // If this type doesn't need a drop at all, then there's nothing to enforce.
        if !needs_drop {
            return;
        }

        let mut err_span = self.span;
        let needs_non_const_drop = if let Some(local) = dropped_place.as_local() {
            // Use the span where the local was declared as the span of the drop error.
            err_span = self.body.local_decls[local].source_info.span;
            self.qualifs.needs_non_const_drop(self.ccx, local, location)
        } else {
            qualifs::NeedsNonConstDrop::in_any_value_of_ty(self.ccx, ty_of_dropped_place)
        };

        self.check_op_spanned(
            ops::LiveDrop {
                dropped_at: terminator_span,
                dropped_ty: ty_of_dropped_place,
                needs_non_const_drop,
            },
            err_span,
        );
    }

    /// Check the const stability of the given item (fn or trait).
    fn check_callee_stability(&mut self, def_id: DefId) {
        match self.tcx.lookup_const_stability(def_id) {
            Some(attrs::ConstStability { level: attrs::StabilityLevel::Stable { .. }, .. }) => {
                // All good.
            }
            None => {
                // This doesn't need a separate const-stability check -- const-stability equals
                // regular stability, and regular stability is checked separately.
                // However, we *do* have to worry about *recursive* const stability.
                if self.enforce_recursive_const_stability()
                    && !is_fn_or_trait_safe_to_expose_on_stable(self.tcx, def_id)
                {
                    self.dcx().emit_err(errors::UnmarkedConstItemExposed {
                        span: self.span,
                        def_path: self.tcx.def_path_str(def_id),
                    });
                }
            }
            Some(attrs::ConstStability {
                level: attrs::StabilityLevel::Unstable { implied_by: implied_feature, issue, .. },
                feature,
                ..
            }) => {
                // An unstable const fn/trait with a feature gate.
                let callee_safe_to_expose_on_stable =
                    is_fn_or_trait_safe_to_expose_on_stable(self.tcx, def_id);

                // We only honor `span.allows_unstable` aka `#[allow_internal_unstable]` if
                // the callee is safe to expose, to avoid bypassing recursive stability.
                // This is not ideal since it means the user sees an error, not the macro
                // author, but that's also the case if one forgets to set
                // `#[allow_internal_unstable]` in the first place. Note that this cannot be
                // integrated in the check below since we want to enforce
                // `callee_safe_to_expose_on_stable` even if
                // `!self.enforce_recursive_const_stability()`.
                if (self.span.allows_unstable(feature)
                    || implied_feature.is_some_and(|f| self.span.allows_unstable(f)))
                    && callee_safe_to_expose_on_stable
                {
                    return;
                }

                // We can't use `check_op` to check whether the feature is enabled because
                // the logic is a bit different than elsewhere: local functions don't need
                // the feature gate, and there might be an "implied" gate that also suffices
                // to allow this.
                let feature_enabled = def_id.is_local()
                    || self.tcx.features().enabled(feature)
                    || implied_feature.is_some_and(|f| self.tcx.features().enabled(f))
                    || {
                        // When we're compiling the compiler itself we may pull in
                        // crates from crates.io, but those crates may depend on other
                        // crates also pulled in from crates.io. We want to ideally be
                        // able to compile everything without requiring upstream
                        // modifications, so in the case that this looks like a
                        // `rustc_private` crate (e.g., a compiler crate) and we also have
                        // the `-Z force-unstable-if-unmarked` flag present (we're
                        // compiling a compiler crate), then let this missing feature
                        // annotation slide.
                        // This matches what we do in `eval_stability_allow_unstable` for
                        // regular stability.
                        feature == sym::rustc_private
                            && issue == NonZero::new(27812)
                            && self.tcx.sess.opts.unstable_opts.force_unstable_if_unmarked
                    };
                // Even if the feature is enabled, we still need check_op to double-check
                // this if the callee is not safe to expose on stable.
                if !feature_enabled || !callee_safe_to_expose_on_stable {
                    self.check_op(ops::CallUnstable {
                        def_id,
                        feature,
                        feature_enabled,
                        safe_to_expose_on_stable: callee_safe_to_expose_on_stable,
                        is_function_call: self.tcx.def_kind(def_id) != DefKind::Trait,
                    });
                }
            }
        }
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
                    && let Some(coroutine_kind) = self.tcx.coroutine_kind(def_id)
                {
                    self.check_op(ops::Coroutine(coroutine_kind));
                }
            }

            Rvalue::Ref(_, BorrowKind::Mut { .. }, place)
            | Rvalue::RawPtr(RawPtrKind::Mut, place) => {
                // Inside mutable statics, we allow arbitrary mutable references.
                // We've allowed `static mut FOO = &mut [elements];` for a long time (the exact
                // reasons why are lost to history), and there is no reason to restrict that to
                // arrays and slices.
                let is_allowed =
                    self.const_kind() == hir::ConstContext::Static(hir::Mutability::Mut);

                if !is_allowed && self.place_may_escape(place) {
                    self.check_op(ops::EscapingMutBorrow);
                }
            }

            Rvalue::Ref(_, BorrowKind::Shared | BorrowKind::Fake(_), place)
            | Rvalue::RawPtr(RawPtrKind::Const, place) => {
                let borrowed_place_has_mut_interior = qualifs::in_place::<HasMutInterior, _>(
                    self.ccx,
                    &mut |local| self.qualifs.has_mut_interior(self.ccx, local, location),
                    place.as_ref(),
                );

                if borrowed_place_has_mut_interior && self.place_may_escape(place) {
                    self.check_op(ops::EscapingCellBorrow);
                }
            }

            Rvalue::RawPtr(RawPtrKind::FakeForPtrMetadata, place) => {
                // These are only inserted for slice length, so the place must already be indirect.
                // This implies we do not have to worry about whether the borrow escapes.
                if !place.is_indirect() {
                    self.tcx.dcx().span_delayed_bug(
                        self.body.source_info(location).span,
                        "fake borrows are always indirect",
                    );
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
                NullOp::SizeOf
                | NullOp::AlignOf
                | NullOp::OffsetOf(_)
                | NullOp::UbChecks
                | NullOp::ContractChecks,
                _,
            ) => {}
            Rvalue::ShallowInitBox(_, _) => {}

            Rvalue::UnaryOp(op, operand) => {
                let ty = operand.ty(self.body, self.tcx);
                match op {
                    UnOp::Not | UnOp::Neg => {
                        if is_int_bool_float_or_char(ty) {
                            // Int, bool, float, and char operations are fine.
                        } else {
                            span_bug!(
                                self.span,
                                "non-primitive type in `Rvalue::UnaryOp{op:?}`: {ty:?}",
                            );
                        }
                    }
                    UnOp::PtrMetadata => {
                        // Getting the metadata from a pointer is always const.
                        // We already validated the type is valid in the validator.
                    }
                }
            }

            Rvalue::BinaryOp(op, box (lhs, rhs)) => {
                let lhs_ty = lhs.ty(self.body, self.tcx);
                let rhs_ty = rhs.ty(self.body, self.tcx);

                if is_int_bool_float_or_char(lhs_ty) && is_int_bool_float_or_char(rhs_ty) {
                    // Int, bool, float, and char operations are fine.
                } else if lhs_ty.is_fn_ptr() || lhs_ty.is_raw_ptr() {
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

            Rvalue::WrapUnsafeBinder(..) => {
                // Unsafe binders are always trivial to create.
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
            | StatementKind::BackwardIncompatibleDropHint { .. }
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

                let ConstCx { tcx, body, .. } = *self.ccx;

                let fn_ty = func.ty(body, tcx);

                let (callee, fn_args) = match *fn_ty.kind() {
                    ty::FnDef(def_id, fn_args) => (def_id, fn_args),

                    ty::FnPtr(..) => {
                        self.check_op(ops::FnCallIndirect);
                        // We can get here without an error in miri-unleashed mode... might as well
                        // skip the rest of the checks as well then.
                        return;
                    }
                    _ => {
                        span_bug!(terminator.source_info.span, "invalid callee of type {:?}", fn_ty)
                    }
                };

                let has_const_conditions =
                    self.revalidate_conditional_constness(callee, fn_args, *fn_span);

                // Attempting to call a trait method?
                if let Some(trait_did) = tcx.trait_of_item(callee) {
                    // We can't determine the actual callee here, so we have to do different checks
                    // than usual.

                    trace!("attempting to call a trait method");
                    let trait_is_const = tcx.is_const_trait(trait_did);

                    // Only consider a trait to be const if the const conditions hold.
                    // Otherwise, it's really misleading to call something "conditionally"
                    // const when it's very obviously not conditionally const.
                    if trait_is_const && has_const_conditions == Some(ConstConditionsHold::Yes) {
                        // Trait calls are always conditionally-const.
                        self.check_op(ops::ConditionallyConstCall {
                            callee,
                            args: fn_args,
                            span: *fn_span,
                            call_source,
                        });
                        self.check_callee_stability(trait_did);
                    } else {
                        // Not even a const trait.
                        self.check_op(ops::FnCallNonConst {
                            callee,
                            args: fn_args,
                            span: *fn_span,
                            call_source,
                        });
                    }
                    // That's all we can check here.
                    return;
                }

                // Even if we know the callee, ensure we can use conditionally-const calls.
                if has_const_conditions.is_some() {
                    self.check_op(ops::ConditionallyConstCall {
                        callee,
                        args: fn_args,
                        span: *fn_span,
                        call_source,
                    });
                }

                // At this point, we are calling a function, `callee`, whose `DefId` is known...

                // `begin_panic` and `#[rustc_const_panic_str]` functions accept generic
                // types other than str. Check to enforce that only str can be used in
                // const-eval.

                // const-eval of the `begin_panic` fn assumes the argument is `&str`
                if tcx.is_lang_item(callee, LangItem::BeginPanic) {
                    match args[0].node.ty(&self.ccx.body.local_decls, tcx).kind() {
                        ty::Ref(_, ty, _) if ty.is_str() => {}
                        _ => self.check_op(ops::PanicNonStr),
                    }
                    // Allow this call, skip all the checks below.
                    return;
                }

                // const-eval of `#[rustc_const_panic_str]` functions assumes the argument is `&&str`
                if tcx.has_attr(callee, sym::rustc_const_panic_str) {
                    match args[0].node.ty(&self.ccx.body.local_decls, tcx).kind() {
                        ty::Ref(_, ty, _) if matches!(ty.kind(), ty::Ref(_, ty, _) if ty.is_str()) =>
                            {}
                        _ => {
                            self.check_op(ops::PanicNonStr);
                        }
                    }
                    // Allow this call, skip all the checks below.
                    return;
                }

                // This can be called on stable via the `vec!` macro.
                if tcx.is_lang_item(callee, LangItem::ExchangeMalloc) {
                    self.check_op(ops::HeapAllocation);
                    // Allow this call, skip all the checks below.
                    return;
                }

                // Intrinsics are language primitives, not regular calls, so treat them separately.
                if let Some(intrinsic) = tcx.intrinsic(callee) {
                    if !tcx.is_const_fn(callee) {
                        // Non-const intrinsic.
                        self.check_op(ops::IntrinsicNonConst { name: intrinsic.name });
                        // If we allowed this, we're in miri-unleashed mode, so we might
                        // as well skip the remaining checks.
                        return;
                    }
                    // We use `intrinsic.const_stable` to determine if this can be safely exposed to
                    // stable code, rather than `const_stable_indirect`. This is to make
                    // `#[rustc_const_stable_indirect]` an attribute that is always safe to add.
                    // We also ask is_safe_to_expose_on_stable_const_fn; this determines whether the intrinsic
                    // fallback body is safe to expose on stable.
                    let is_const_stable = intrinsic.const_stable
                        || (!intrinsic.must_be_overridden
                            && is_fn_or_trait_safe_to_expose_on_stable(tcx, callee));
                    match tcx.lookup_const_stability(callee) {
                        None => {
                            // This doesn't need a separate const-stability check -- const-stability equals
                            // regular stability, and regular stability is checked separately.
                            // However, we *do* have to worry about *recursive* const stability.
                            if !is_const_stable && self.enforce_recursive_const_stability() {
                                self.dcx().emit_err(errors::UnmarkedIntrinsicExposed {
                                    span: self.span,
                                    def_path: self.tcx.def_path_str(callee),
                                });
                            }
                        }
                        Some(attrs::ConstStability {
                            level: attrs::StabilityLevel::Unstable { .. },
                            feature,
                            ..
                        }) => {
                            self.check_op(ops::IntrinsicUnstable {
                                name: intrinsic.name,
                                feature,
                                const_stable_indirect: is_const_stable,
                            });
                        }
                        Some(attrs::ConstStability {
                            level: attrs::StabilityLevel::Stable { .. },
                            ..
                        }) => {
                            // All good. Note that a `#[rustc_const_stable]` intrinsic (meaning it
                            // can be *directly* invoked from stable const code) does not always
                            // have the `#[rustc_intrinsic_const_stable_indirect]` attribute (which controls
                            // exposing an intrinsic indirectly); we accept this call anyway.
                        }
                    }
                    // This completes the checks for intrinsics.
                    return;
                }

                if !tcx.is_const_fn(callee) {
                    self.check_op(ops::FnCallNonConst {
                        callee,
                        args: fn_args,
                        span: *fn_span,
                        call_source,
                    });
                    // If we allowed this, we're in miri-unleashed mode, so we might
                    // as well skip the remaining checks.
                    return;
                }

                // Finally, stability for regular function calls -- this is the big one.
                self.check_callee_stability(callee);
            }

            // Forbid all `Drop` terminators unless the place being dropped is a local with no
            // projections that cannot be `NeedsNonConstDrop`.
            TerminatorKind::Drop { place: dropped_place, .. } => {
                // If we are checking live drops after drop-elaboration, don't emit duplicate
                // errors here.
                if super::post_drop_elaboration::checking_enabled(self.ccx) {
                    return;
                }

                self.check_drop_terminator(*dropped_place, location, terminator.source_info.span);
            }

            TerminatorKind::InlineAsm { .. } => self.check_op(ops::InlineAsm),

            TerminatorKind::Yield { .. } => {
                self.check_op(ops::Coroutine(
                    self.tcx
                        .coroutine_kind(self.body.source.def_id())
                        .expect("Only expected to have a yield in a coroutine"),
                ));
            }

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

fn emit_unstable_in_stable_exposed_error(
    ccx: &ConstCx<'_, '_>,
    span: Span,
    gate: Symbol,
    is_function_call: bool,
) -> ErrorGuaranteed {
    let attr_span = ccx.tcx.def_span(ccx.def_id()).shrink_to_lo();

    ccx.dcx().emit_err(errors::UnstableInStableExposed {
        gate: gate.to_string(),
        span,
        attr_span,
        is_function_call,
        is_function_call2: is_function_call,
    })
}
