use std::iter;

use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::{
    Applicability, Diag, E0309, E0310, E0311, E0803, Subdiagnostic, struct_span_code_err,
};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::Visitor;
use rustc_hir::{self as hir, ParamName};
use rustc_middle::bug;
use rustc_middle::traits::ObligationCauseCode;
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::{
    self, IsSuggestable, Region, Ty, TyCtxt, TypeVisitableExt as _, Upcast as _,
};
use rustc_span::{BytePos, ErrorGuaranteed, Span, Symbol, kw};
use tracing::{debug, instrument};

use super::ObligationCauseAsDiagArg;
use super::nice_region_error::find_anon_type;
use crate::error_reporting::TypeErrCtxt;
use crate::error_reporting::infer::ObligationCauseExt;
use crate::errors::{
    self, FulfillReqLifetime, LfBoundNotSatisfied, OutlivesBound, OutlivesContent,
    RefLongerThanData, RegionOriginNote, WhereClauseSuggestions, note_and_explain,
};
use crate::fluent_generated as fluent;
use crate::infer::region_constraints::GenericKind;
use crate::infer::{
    BoundRegionConversionTime, InferCtxt, RegionResolutionError, RegionVariableOrigin,
    SubregionOrigin,
};

impl<'a, 'tcx> TypeErrCtxt<'a, 'tcx> {
    pub fn report_region_errors(
        &self,
        generic_param_scope: LocalDefId,
        errors: &[RegionResolutionError<'tcx>],
    ) -> ErrorGuaranteed {
        assert!(!errors.is_empty());

        if let Some(guaranteed) = self.infcx.tainted_by_errors() {
            return guaranteed;
        }

        debug!("report_region_errors(): {} errors to start", errors.len());

        // try to pre-process the errors, which will group some of them
        // together into a `ProcessedErrors` group:
        let errors = self.process_errors(errors);

        debug!("report_region_errors: {} errors after preprocessing", errors.len());

        let mut guar = None;
        for error in errors {
            debug!("report_region_errors: error = {:?}", error);

            let e = if let Some(guar) =
                self.try_report_nice_region_error(generic_param_scope, &error)
            {
                guar
            } else {
                match error.clone() {
                    // These errors could indicate all manner of different
                    // problems with many different solutions. Rather
                    // than generate a "one size fits all" error, what we
                    // attempt to do is go through a number of specific
                    // scenarios and try to find the best way to present
                    // the error. If all of these fails, we fall back to a rather
                    // general bit of code that displays the error information
                    RegionResolutionError::ConcreteFailure(origin, sub, sup) => {
                        if sub.is_placeholder() || sup.is_placeholder() {
                            self.report_placeholder_failure(generic_param_scope, origin, sub, sup)
                                .emit()
                        } else {
                            self.report_concrete_failure(generic_param_scope, origin, sub, sup)
                                .emit()
                        }
                    }

                    RegionResolutionError::GenericBoundFailure(origin, param_ty, sub) => self
                        .report_generic_bound_failure(
                            generic_param_scope,
                            origin.span(),
                            Some(origin),
                            param_ty,
                            sub,
                        ),

                    RegionResolutionError::SubSupConflict(
                        _,
                        var_origin,
                        sub_origin,
                        sub_r,
                        sup_origin,
                        sup_r,
                        _,
                    ) => {
                        if sub_r.is_placeholder() {
                            self.report_placeholder_failure(
                                generic_param_scope,
                                sub_origin,
                                sub_r,
                                sup_r,
                            )
                            .emit()
                        } else if sup_r.is_placeholder() {
                            self.report_placeholder_failure(
                                generic_param_scope,
                                sup_origin,
                                sub_r,
                                sup_r,
                            )
                            .emit()
                        } else {
                            self.report_sub_sup_conflict(
                                generic_param_scope,
                                var_origin,
                                sub_origin,
                                sub_r,
                                sup_origin,
                                sup_r,
                            )
                        }
                    }

                    RegionResolutionError::UpperBoundUniverseConflict(
                        _,
                        _,
                        _,
                        sup_origin,
                        sup_r,
                    ) => {
                        assert!(sup_r.is_placeholder());

                        // Make a dummy value for the "sub region" --
                        // this is the initial value of the
                        // placeholder. In practice, we expect more
                        // tailored errors that don't really use this
                        // value.
                        let sub_r = self.tcx.lifetimes.re_erased;

                        self.report_placeholder_failure(
                            generic_param_scope,
                            sup_origin,
                            sub_r,
                            sup_r,
                        )
                        .emit()
                    }

                    RegionResolutionError::CannotNormalize(clause, origin) => {
                        let clause: ty::Clause<'tcx> =
                            clause.map_bound(ty::ClauseKind::TypeOutlives).upcast(self.tcx);
                        self.tcx
                            .dcx()
                            .struct_span_err(origin.span(), format!("cannot normalize `{clause}`"))
                            .emit()
                    }
                }
            };

            guar = Some(e)
        }

        guar.unwrap()
    }

    // This method goes through all the errors and try to group certain types
    // of error together, for the purpose of suggesting explicit lifetime
    // parameters to the user. This is done so that we can have a more
    // complete view of what lifetimes should be the same.
    // If the return value is an empty vector, it means that processing
    // failed (so the return value of this method should not be used).
    //
    // The method also attempts to weed out messages that seem like
    // duplicates that will be unhelpful to the end-user. But
    // obviously it never weeds out ALL errors.
    fn process_errors(
        &self,
        errors: &[RegionResolutionError<'tcx>],
    ) -> Vec<RegionResolutionError<'tcx>> {
        debug!("process_errors()");

        // We want to avoid reporting generic-bound failures if we can
        // avoid it: these have a very high rate of being unhelpful in
        // practice. This is because they are basically secondary
        // checks that test the state of the region graph after the
        // rest of inference is done, and the other kinds of errors
        // indicate that the region constraint graph is internally
        // inconsistent, so these test results are likely to be
        // meaningless.
        //
        // Therefore, we filter them out of the list unless they are
        // the only thing in the list.

        let is_bound_failure = |e: &RegionResolutionError<'tcx>| match *e {
            RegionResolutionError::GenericBoundFailure(..) => true,
            RegionResolutionError::ConcreteFailure(..)
            | RegionResolutionError::SubSupConflict(..)
            | RegionResolutionError::UpperBoundUniverseConflict(..)
            | RegionResolutionError::CannotNormalize(..) => false,
        };

        let mut errors = if errors.iter().all(|e| is_bound_failure(e)) {
            errors.to_owned()
        } else {
            errors.iter().filter(|&e| !is_bound_failure(e)).cloned().collect()
        };

        // sort the errors by span, for better error message stability.
        errors.sort_by_key(|u| match *u {
            RegionResolutionError::ConcreteFailure(ref sro, _, _) => sro.span(),
            RegionResolutionError::GenericBoundFailure(ref sro, _, _) => sro.span(),
            RegionResolutionError::SubSupConflict(_, ref rvo, _, _, _, _, _) => rvo.span(),
            RegionResolutionError::UpperBoundUniverseConflict(_, ref rvo, _, _, _) => rvo.span(),
            RegionResolutionError::CannotNormalize(_, ref sro) => sro.span(),
        });
        errors
    }

    pub(super) fn note_region_origin(&self, err: &mut Diag<'_>, origin: &SubregionOrigin<'tcx>) {
        match *origin {
            SubregionOrigin::Subtype(ref trace) => RegionOriginNote::WithRequirement {
                span: trace.cause.span,
                requirement: ObligationCauseAsDiagArg(trace.cause.clone()),
                expected_found: self.values_str(trace.values, &trace.cause, err.long_ty_path()),
            }
            .add_to_diag(err),
            SubregionOrigin::Reborrow(span) => {
                RegionOriginNote::Plain { span, msg: fluent::trait_selection_reborrow }
                    .add_to_diag(err)
            }
            SubregionOrigin::RelateObjectBound(span) => {
                RegionOriginNote::Plain { span, msg: fluent::trait_selection_relate_object_bound }
                    .add_to_diag(err);
            }
            SubregionOrigin::ReferenceOutlivesReferent(ty, span) => {
                RegionOriginNote::WithName {
                    span,
                    msg: fluent::trait_selection_reference_outlives_referent,
                    name: &self.ty_to_string(ty),
                    continues: false,
                }
                .add_to_diag(err);
            }
            SubregionOrigin::RelateParamBound(span, ty, opt_span) => {
                RegionOriginNote::WithName {
                    span,
                    msg: fluent::trait_selection_relate_param_bound,
                    name: &self.ty_to_string(ty),
                    continues: opt_span.is_some(),
                }
                .add_to_diag(err);
                if let Some(span) = opt_span {
                    RegionOriginNote::Plain {
                        span,
                        msg: fluent::trait_selection_relate_param_bound_2,
                    }
                    .add_to_diag(err);
                }
            }
            SubregionOrigin::RelateRegionParamBound(span, _) => {
                RegionOriginNote::Plain {
                    span,
                    msg: fluent::trait_selection_relate_region_param_bound,
                }
                .add_to_diag(err);
            }
            SubregionOrigin::CompareImplItemObligation { span, .. } => {
                RegionOriginNote::Plain {
                    span,
                    msg: fluent::trait_selection_compare_impl_item_obligation,
                }
                .add_to_diag(err);
            }
            SubregionOrigin::CheckAssociatedTypeBounds { ref parent, .. } => {
                self.note_region_origin(err, parent);
            }
            SubregionOrigin::AscribeUserTypeProvePredicate(span) => {
                RegionOriginNote::Plain {
                    span,
                    msg: fluent::trait_selection_ascribe_user_type_prove_predicate,
                }
                .add_to_diag(err);
            }
        }
    }

    pub(super) fn report_concrete_failure(
        &self,
        generic_param_scope: LocalDefId,
        origin: SubregionOrigin<'tcx>,
        sub: Region<'tcx>,
        sup: Region<'tcx>,
    ) -> Diag<'a> {
        let mut err = match origin {
            SubregionOrigin::Subtype(box trace) => {
                let terr = TypeError::RegionsDoesNotOutlive(sup, sub);
                let mut err = self.report_and_explain_type_error(
                    trace,
                    self.tcx.param_env(generic_param_scope),
                    terr,
                );
                match (sub.kind(), sup.kind()) {
                    (ty::RePlaceholder(_), ty::RePlaceholder(_)) => {}
                    (ty::RePlaceholder(_), _) => {
                        note_and_explain_region(
                            self.tcx,
                            &mut err,
                            generic_param_scope,
                            "",
                            sup,
                            " doesn't meet the lifetime requirements",
                            None,
                        );
                    }
                    (_, ty::RePlaceholder(_)) => {
                        note_and_explain_region(
                            self.tcx,
                            &mut err,
                            generic_param_scope,
                            "the required lifetime does not necessarily outlive ",
                            sub,
                            "",
                            None,
                        );
                    }
                    _ => {
                        note_and_explain_region(
                            self.tcx,
                            &mut err,
                            generic_param_scope,
                            "",
                            sup,
                            "...",
                            None,
                        );
                        note_and_explain_region(
                            self.tcx,
                            &mut err,
                            generic_param_scope,
                            "...does not necessarily outlive ",
                            sub,
                            "",
                            None,
                        );
                    }
                }
                err
            }
            SubregionOrigin::Reborrow(span) => {
                let reference_valid = note_and_explain::RegionExplanation::new(
                    self.tcx,
                    generic_param_scope,
                    sub,
                    None,
                    note_and_explain::PrefixKind::RefValidFor,
                    note_and_explain::SuffixKind::Continues,
                );
                let content_valid = note_and_explain::RegionExplanation::new(
                    self.tcx,
                    generic_param_scope,
                    sup,
                    None,
                    note_and_explain::PrefixKind::ContentValidFor,
                    note_and_explain::SuffixKind::Empty,
                );
                self.dcx().create_err(OutlivesContent {
                    span,
                    notes: reference_valid.into_iter().chain(content_valid).collect(),
                })
            }
            SubregionOrigin::RelateObjectBound(span) => {
                let object_valid = note_and_explain::RegionExplanation::new(
                    self.tcx,
                    generic_param_scope,
                    sub,
                    None,
                    note_and_explain::PrefixKind::TypeObjValidFor,
                    note_and_explain::SuffixKind::Empty,
                );
                let pointer_valid = note_and_explain::RegionExplanation::new(
                    self.tcx,
                    generic_param_scope,
                    sup,
                    None,
                    note_and_explain::PrefixKind::SourcePointerValidFor,
                    note_and_explain::SuffixKind::Empty,
                );
                self.dcx().create_err(OutlivesBound {
                    span,
                    notes: object_valid.into_iter().chain(pointer_valid).collect(),
                })
            }
            SubregionOrigin::RelateParamBound(span, ty, opt_span) => {
                let prefix = match sub.kind() {
                    ty::ReStatic => note_and_explain::PrefixKind::TypeSatisfy,
                    _ => note_and_explain::PrefixKind::TypeOutlive,
                };
                let suffix = if opt_span.is_some() {
                    note_and_explain::SuffixKind::ReqByBinding
                } else {
                    note_and_explain::SuffixKind::Empty
                };
                let note = note_and_explain::RegionExplanation::new(
                    self.tcx,
                    generic_param_scope,
                    sub,
                    opt_span,
                    prefix,
                    suffix,
                );
                self.dcx().create_err(FulfillReqLifetime {
                    span,
                    ty: self.resolve_vars_if_possible(ty),
                    note,
                })
            }
            SubregionOrigin::RelateRegionParamBound(span, ty) => {
                let param_instantiated = note_and_explain::RegionExplanation::new(
                    self.tcx,
                    generic_param_scope,
                    sup,
                    None,
                    note_and_explain::PrefixKind::LfParamInstantiatedWith,
                    note_and_explain::SuffixKind::Empty,
                );
                let mut alt_span = None;
                if let Some(ty) = ty
                    && sub.is_static()
                    && let ty::Dynamic(preds, _, ty::DynKind::Dyn) = ty.kind()
                    && let Some(def_id) = preds.principal_def_id()
                {
                    for (clause, span) in
                        self.tcx.predicates_of(def_id).instantiate_identity(self.tcx)
                    {
                        if let ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(a, b)) =
                            clause.kind().skip_binder()
                            && let ty::Param(param) = a.kind()
                            && param.name == kw::SelfUpper
                            && b.is_static()
                        {
                            // Point at explicit `'static` bound on the trait (`trait T: 'static`).
                            alt_span = Some(span);
                        }
                    }
                }
                let param_must_outlive = note_and_explain::RegionExplanation::new(
                    self.tcx,
                    generic_param_scope,
                    sub,
                    alt_span,
                    note_and_explain::PrefixKind::LfParamMustOutlive,
                    note_and_explain::SuffixKind::Empty,
                );
                self.dcx().create_err(LfBoundNotSatisfied {
                    span,
                    notes: param_instantiated.into_iter().chain(param_must_outlive).collect(),
                })
            }
            SubregionOrigin::ReferenceOutlivesReferent(ty, span) => {
                let pointer_valid = note_and_explain::RegionExplanation::new(
                    self.tcx,
                    generic_param_scope,
                    sub,
                    None,
                    note_and_explain::PrefixKind::PointerValidFor,
                    note_and_explain::SuffixKind::Empty,
                );
                let data_valid = note_and_explain::RegionExplanation::new(
                    self.tcx,
                    generic_param_scope,
                    sup,
                    None,
                    note_and_explain::PrefixKind::DataValidFor,
                    note_and_explain::SuffixKind::Empty,
                );
                self.dcx().create_err(RefLongerThanData {
                    span,
                    ty: self.resolve_vars_if_possible(ty),
                    notes: pointer_valid.into_iter().chain(data_valid).collect(),
                })
            }
            SubregionOrigin::CompareImplItemObligation {
                span,
                impl_item_def_id,
                trait_item_def_id,
            } => {
                let mut err = self.report_extra_impl_obligation(
                    span,
                    impl_item_def_id,
                    trait_item_def_id,
                    &format!("`{sup}: {sub}`"),
                );
                // We should only suggest rewriting the `where` clause if the predicate is within that `where` clause
                if let Some(generics) = self.tcx.hir_get_generics(impl_item_def_id)
                    && generics.where_clause_span.contains(span)
                {
                    self.suggest_copy_trait_method_bounds(
                        trait_item_def_id,
                        impl_item_def_id,
                        &mut err,
                    );
                }
                err
            }
            SubregionOrigin::CheckAssociatedTypeBounds {
                impl_item_def_id,
                trait_item_def_id,
                parent,
            } => {
                let mut err = self.report_concrete_failure(generic_param_scope, *parent, sub, sup);

                // Don't mention the item name if it's an RPITIT, since that'll just confuse
                // folks.
                if !self.tcx.is_impl_trait_in_trait(impl_item_def_id.to_def_id()) {
                    let trait_item_span = self.tcx.def_span(trait_item_def_id);
                    let item_name = self.tcx.item_name(impl_item_def_id.to_def_id());
                    err.span_label(
                        trait_item_span,
                        format!("definition of `{item_name}` from trait"),
                    );
                }

                self.suggest_copy_trait_method_bounds(
                    trait_item_def_id,
                    impl_item_def_id,
                    &mut err,
                );
                err
            }
            SubregionOrigin::AscribeUserTypeProvePredicate(span) => {
                let instantiated = note_and_explain::RegionExplanation::new(
                    self.tcx,
                    generic_param_scope,
                    sup,
                    None,
                    note_and_explain::PrefixKind::LfInstantiatedWith,
                    note_and_explain::SuffixKind::Empty,
                );
                let must_outlive = note_and_explain::RegionExplanation::new(
                    self.tcx,
                    generic_param_scope,
                    sub,
                    None,
                    note_and_explain::PrefixKind::LfMustOutlive,
                    note_and_explain::SuffixKind::Empty,
                );
                self.dcx().create_err(LfBoundNotSatisfied {
                    span,
                    notes: instantiated.into_iter().chain(must_outlive).collect(),
                })
            }
        };
        if sub.is_error() || sup.is_error() {
            err.downgrade_to_delayed_bug();
        }
        err
    }

    pub fn suggest_copy_trait_method_bounds(
        &self,
        trait_item_def_id: DefId,
        impl_item_def_id: LocalDefId,
        err: &mut Diag<'_>,
    ) {
        // FIXME(compiler-errors): Right now this is only being used for region
        // predicate mismatches. Ideally, we'd use it for *all* predicate mismatches,
        // but right now it's not really very smart when it comes to implicit `Sized`
        // predicates and bounds on the trait itself.

        let Some(impl_def_id) = self.tcx.associated_item(impl_item_def_id).impl_container(self.tcx)
        else {
            return;
        };
        let Some(trait_ref) = self.tcx.impl_trait_ref(impl_def_id) else {
            return;
        };
        let trait_args = trait_ref
            .instantiate_identity()
            // Replace the explicit self type with `Self` for better suggestion rendering
            .with_self_ty(self.tcx, Ty::new_param(self.tcx, 0, kw::SelfUpper))
            .args;
        let trait_item_args = ty::GenericArgs::identity_for_item(self.tcx, impl_item_def_id)
            .rebase_onto(self.tcx, impl_def_id, trait_args);

        let Ok(trait_predicates) =
            self.tcx
                .explicit_predicates_of(trait_item_def_id)
                .instantiate_own(self.tcx, trait_item_args)
                .map(|(pred, _)| {
                    if pred.is_suggestable(self.tcx, false) {
                        Ok(pred.to_string())
                    } else {
                        Err(())
                    }
                })
                .collect::<Result<Vec<_>, ()>>()
        else {
            return;
        };

        let Some(generics) = self.tcx.hir_get_generics(impl_item_def_id) else {
            return;
        };

        let suggestion = if trait_predicates.is_empty() {
            WhereClauseSuggestions::Remove { span: generics.where_clause_span }
        } else {
            let space = if generics.where_clause_span.is_empty() { " " } else { "" };
            WhereClauseSuggestions::CopyPredicates {
                span: generics.where_clause_span,
                space,
                trait_predicates: trait_predicates.join(", "),
            }
        };
        err.subdiagnostic(suggestion);
    }

    pub(super) fn report_placeholder_failure(
        &self,
        generic_param_scope: LocalDefId,
        placeholder_origin: SubregionOrigin<'tcx>,
        sub: Region<'tcx>,
        sup: Region<'tcx>,
    ) -> Diag<'a> {
        // I can't think how to do better than this right now. -nikomatsakis
        debug!(?placeholder_origin, ?sub, ?sup, "report_placeholder_failure");
        match placeholder_origin {
            SubregionOrigin::Subtype(box ref trace)
                if matches!(
                    &trace.cause.code().peel_derives(),
                    ObligationCauseCode::WhereClause(..)
                        | ObligationCauseCode::WhereClauseInExpr(..)
                ) =>
            {
                // Hack to get around the borrow checker because trace.cause has an `Rc`.
                if let ObligationCauseCode::WhereClause(_, span)
                | ObligationCauseCode::WhereClauseInExpr(_, span, ..) =
                    &trace.cause.code().peel_derives()
                {
                    let span = *span;
                    let mut err = self.report_concrete_failure(
                        generic_param_scope,
                        placeholder_origin,
                        sub,
                        sup,
                    );
                    if !span.is_dummy() {
                        err =
                            err.with_span_note(span, "the lifetime requirement is introduced here");
                    }
                    err
                } else {
                    unreachable!(
                        "control flow ensures we have a `BindingObligation` or `WhereClauseInExpr` here..."
                    )
                }
            }
            SubregionOrigin::Subtype(box trace) => {
                let terr = TypeError::RegionsPlaceholderMismatch;
                return self.report_and_explain_type_error(
                    trace,
                    self.tcx.param_env(generic_param_scope),
                    terr,
                );
            }
            _ => {
                return self.report_concrete_failure(
                    generic_param_scope,
                    placeholder_origin,
                    sub,
                    sup,
                );
            }
        }
    }

    pub fn report_generic_bound_failure(
        &self,
        generic_param_scope: LocalDefId,
        span: Span,
        origin: Option<SubregionOrigin<'tcx>>,
        bound_kind: GenericKind<'tcx>,
        sub: Region<'tcx>,
    ) -> ErrorGuaranteed {
        self.construct_generic_bound_failure(generic_param_scope, span, origin, bound_kind, sub)
            .emit()
    }

    pub fn construct_generic_bound_failure(
        &self,
        generic_param_scope: LocalDefId,
        span: Span,
        origin: Option<SubregionOrigin<'tcx>>,
        bound_kind: GenericKind<'tcx>,
        sub: Region<'tcx>,
    ) -> Diag<'a> {
        if let Some(SubregionOrigin::CompareImplItemObligation {
            span,
            impl_item_def_id,
            trait_item_def_id,
        }) = origin
        {
            return self.report_extra_impl_obligation(
                span,
                impl_item_def_id,
                trait_item_def_id,
                &format!("`{bound_kind}: {sub}`"),
            );
        }

        let labeled_user_string = match bound_kind {
            GenericKind::Param(ref p) => format!("the parameter type `{p}`"),
            GenericKind::Placeholder(ref p) => format!("the placeholder type `{p:?}`"),
            GenericKind::Alias(ref p) => match p.kind(self.tcx) {
                ty::Projection | ty::Inherent => {
                    format!("the associated type `{p}`")
                }
                ty::Free => format!("the type alias `{p}`"),
                ty::Opaque => format!("the opaque type `{p}`"),
            },
        };

        let mut err = self
            .tcx
            .dcx()
            .struct_span_err(span, format!("{labeled_user_string} may not live long enough"));
        err.code(match sub.kind() {
            ty::ReEarlyParam(_) | ty::ReLateParam(_) if sub.has_name() => E0309,
            ty::ReStatic => E0310,
            _ => E0311,
        });

        '_explain: {
            let (description, span) = match sub.kind() {
                ty::ReEarlyParam(_) | ty::ReLateParam(_) | ty::ReStatic => {
                    msg_span_from_named_region(self.tcx, generic_param_scope, sub, Some(span))
                }
                _ => (format!("lifetime `{sub}`"), Some(span)),
            };
            let prefix = format!("{labeled_user_string} must be valid for ");
            label_msg_span(&mut err, &prefix, description, span, "...");
            if let Some(origin) = origin {
                self.note_region_origin(&mut err, &origin);
            }
        }

        'suggestion: {
            let msg = "consider adding an explicit lifetime bound";

            if (bound_kind, sub).has_infer_regions()
                || (bound_kind, sub).has_placeholders()
                || !bound_kind.is_suggestable(self.tcx, false)
            {
                let lt_name = sub.get_name_or_anon().to_string();
                err.help(format!("{msg} `{bound_kind}: {lt_name}`..."));
                break 'suggestion;
            }

            let mut generic_param_scope = generic_param_scope;
            while self.tcx.def_kind(generic_param_scope) == DefKind::OpaqueTy {
                generic_param_scope = self.tcx.local_parent(generic_param_scope);
            }

            // type_param_sugg_span is (span, has_bounds, needs_parentheses)
            let (type_scope, type_param_sugg_span) = match bound_kind {
                GenericKind::Param(param) => {
                    let generics = self.tcx.generics_of(generic_param_scope);
                    let type_param = generics.type_param(param, self.tcx);
                    let def_id = type_param.def_id.expect_local();
                    let scope = self.tcx.local_def_id_to_hir_id(def_id).owner.def_id;
                    // Get the `hir::Param` to verify whether it already has any bounds.
                    // We do this to avoid suggesting code that ends up as `T: 'a'b`,
                    // instead we suggest `T: 'a + 'b` in that case.
                    let hir_generics = self.tcx.hir_get_generics(scope).unwrap();
                    let sugg_span = match hir_generics.bounds_span_for_suggestions(def_id) {
                        Some((span, open_paren_sp)) => Some((span, true, open_paren_sp)),
                        // If `param` corresponds to `Self`, no usable suggestion span.
                        None if generics.has_self && param.index == 0 => None,
                        None => {
                            let span = if let Some(param) =
                                hir_generics.params.iter().find(|param| param.def_id == def_id)
                                && let ParamName::Plain(ident) = param.name
                            {
                                ident.span.shrink_to_hi()
                            } else {
                                let span = self.tcx.def_span(def_id);
                                span.shrink_to_hi()
                            };
                            Some((span, false, None))
                        }
                    };
                    (scope, sugg_span)
                }
                _ => (generic_param_scope, None),
            };
            let suggestion_scope = {
                let lifetime_scope = match sub.kind() {
                    ty::ReStatic => hir::def_id::CRATE_DEF_ID,
                    _ => match self.tcx.is_suitable_region(generic_param_scope, sub) {
                        Some(info) => info.scope,
                        None => generic_param_scope,
                    },
                };
                match self.tcx.is_descendant_of(type_scope.into(), lifetime_scope.into()) {
                    true => type_scope,
                    false => lifetime_scope,
                }
            };

            let mut suggs = vec![];
            let lt_name = self.suggest_name_region(generic_param_scope, sub, &mut suggs);

            if let Some((sp, has_lifetimes, open_paren_sp)) = type_param_sugg_span
                && suggestion_scope == type_scope
            {
                let suggestion =
                    if has_lifetimes { format!(" + {lt_name}") } else { format!(": {lt_name}") };

                if let Some(open_paren_sp) = open_paren_sp {
                    suggs.push((open_paren_sp, "(".to_string()));
                    suggs.push((sp, format!("){suggestion}")));
                } else {
                    suggs.push((sp, suggestion))
                }
            } else if let GenericKind::Alias(ref p) = bound_kind
                && let ty::Projection = p.kind(self.tcx)
                && let DefKind::AssocTy = self.tcx.def_kind(p.def_id)
                && let Some(ty::ImplTraitInTraitData::Trait { .. }) =
                    self.tcx.opt_rpitit_info(p.def_id)
            {
                // The lifetime found in the `impl` is longer than the one on the RPITIT.
                // Do not suggest `<Type as Trait>::{opaque}: 'static`.
            } else if let Some(generics) = self.tcx.hir_get_generics(suggestion_scope) {
                let pred = format!("{bound_kind}: {lt_name}");
                let suggestion = format!("{} {}", generics.add_where_or_trailing_comma(), pred);
                suggs.push((generics.tail_span_for_predicate_suggestion(), suggestion))
            } else {
                let consider = format!("{msg} `{bound_kind}: {sub}`...");
                err.help(consider);
            }

            if !suggs.is_empty() {
                err.multipart_suggestion_verbose(
                    msg,
                    suggs,
                    Applicability::MaybeIncorrect, // Issue #41966
                );
            }
        }

        err
    }

    pub fn suggest_name_region(
        &self,
        generic_param_scope: LocalDefId,
        lifetime: Region<'tcx>,
        add_lt_suggs: &mut Vec<(Span, String)>,
    ) -> String {
        struct LifetimeReplaceVisitor<'a> {
            needle: hir::LifetimeKind,
            new_lt: &'a str,
            add_lt_suggs: &'a mut Vec<(Span, String)>,
        }

        impl<'hir> hir::intravisit::Visitor<'hir> for LifetimeReplaceVisitor<'_> {
            fn visit_lifetime(&mut self, lt: &'hir hir::Lifetime) {
                if lt.kind == self.needle {
                    self.add_lt_suggs.push(lt.suggestion(self.new_lt));
                }
            }
        }

        let (lifetime_def_id, lifetime_scope) = match self
            .tcx
            .is_suitable_region(generic_param_scope, lifetime)
        {
            Some(info) if !lifetime.has_name() => (info.region_def_id.expect_local(), info.scope),
            _ => return lifetime.get_name_or_anon().to_string(),
        };

        let new_lt = {
            let generics = self.tcx.generics_of(lifetime_scope);
            let mut used_names =
                iter::successors(Some(generics), |g| g.parent.map(|p| self.tcx.generics_of(p)))
                    .flat_map(|g| &g.own_params)
                    .filter(|p| matches!(p.kind, ty::GenericParamDefKind::Lifetime))
                    .map(|p| p.name)
                    .collect::<Vec<_>>();
            let hir_id = self.tcx.local_def_id_to_hir_id(lifetime_scope);
            // consider late-bound lifetimes ...
            used_names.extend(self.tcx.late_bound_vars(hir_id).into_iter().filter_map(
                |p| match p {
                    ty::BoundVariableKind::Region(lt) => lt.get_name(),
                    _ => None,
                },
            ));
            (b'a'..=b'z')
                .map(|c| format!("'{}", c as char))
                .find(|candidate| !used_names.iter().any(|e| e.as_str() == candidate))
                .unwrap_or_else(|| "'lt".to_string())
        };

        let mut visitor = LifetimeReplaceVisitor {
            needle: hir::LifetimeKind::Param(lifetime_def_id),
            add_lt_suggs,
            new_lt: &new_lt,
        };
        match self.tcx.expect_hir_owner_node(lifetime_scope) {
            hir::OwnerNode::Item(i) => visitor.visit_item(i),
            hir::OwnerNode::ForeignItem(i) => visitor.visit_foreign_item(i),
            hir::OwnerNode::ImplItem(i) => visitor.visit_impl_item(i),
            hir::OwnerNode::TraitItem(i) => visitor.visit_trait_item(i),
            hir::OwnerNode::Crate(_) => bug!("OwnerNode::Crate doesn't not have generics"),
            hir::OwnerNode::Synthetic => unreachable!(),
        }

        let ast_generics = self.tcx.hir_get_generics(lifetime_scope).unwrap();
        let sugg = ast_generics
            .span_for_lifetime_suggestion()
            .map(|span| (span, format!("{new_lt}, ")))
            .unwrap_or_else(|| (ast_generics.span, format!("<{new_lt}>")));
        add_lt_suggs.push(sugg);

        new_lt
    }

    fn report_sub_sup_conflict(
        &self,
        generic_param_scope: LocalDefId,
        var_origin: RegionVariableOrigin,
        sub_origin: SubregionOrigin<'tcx>,
        sub_region: Region<'tcx>,
        sup_origin: SubregionOrigin<'tcx>,
        sup_region: Region<'tcx>,
    ) -> ErrorGuaranteed {
        let mut err = self.report_inference_failure(var_origin);

        note_and_explain_region(
            self.tcx,
            &mut err,
            generic_param_scope,
            "first, the lifetime cannot outlive ",
            sup_region,
            "...",
            None,
        );

        debug!("report_sub_sup_conflict: var_origin={:?}", var_origin);
        debug!("report_sub_sup_conflict: sub_region={:?}", sub_region);
        debug!("report_sub_sup_conflict: sub_origin={:?}", sub_origin);
        debug!("report_sub_sup_conflict: sup_region={:?}", sup_region);
        debug!("report_sub_sup_conflict: sup_origin={:?}", sup_origin);

        if let SubregionOrigin::Subtype(ref sup_trace) = sup_origin
            && let SubregionOrigin::Subtype(ref sub_trace) = sub_origin
            && let Some((sup_expected, sup_found)) =
                self.values_str(sup_trace.values, &sup_trace.cause, err.long_ty_path())
            && let Some((sub_expected, sub_found)) =
                self.values_str(sub_trace.values, &sup_trace.cause, err.long_ty_path())
            && sub_expected == sup_expected
            && sub_found == sup_found
        {
            note_and_explain_region(
                self.tcx,
                &mut err,
                generic_param_scope,
                "...but the lifetime must also be valid for ",
                sub_region,
                "...",
                None,
            );
            err.span_note(
                sup_trace.cause.span,
                format!("...so that the {}", sup_trace.cause.as_requirement_str()),
            );

            err.note_expected_found("", sup_expected, "", sup_found);
            return if sub_region.is_error() | sup_region.is_error() {
                err.delay_as_bug()
            } else {
                err.emit()
            };
        }

        self.note_region_origin(&mut err, &sup_origin);

        note_and_explain_region(
            self.tcx,
            &mut err,
            generic_param_scope,
            "but, the lifetime must be valid for ",
            sub_region,
            "...",
            None,
        );

        self.note_region_origin(&mut err, &sub_origin);
        if sub_region.is_error() | sup_region.is_error() { err.delay_as_bug() } else { err.emit() }
    }

    fn report_inference_failure(&self, var_origin: RegionVariableOrigin) -> Diag<'_> {
        let br_string = |br: ty::BoundRegionKind| {
            let mut s = match br {
                ty::BoundRegionKind::Named(_, name) => name.to_string(),
                _ => String::new(),
            };
            if !s.is_empty() {
                s.push(' ');
            }
            s
        };
        let var_description = match var_origin {
            RegionVariableOrigin::Misc(_) => String::new(),
            RegionVariableOrigin::PatternRegion(_) => " for pattern".to_string(),
            RegionVariableOrigin::BorrowRegion(_) => " for borrow expression".to_string(),
            RegionVariableOrigin::Autoref(_) => " for autoref".to_string(),
            RegionVariableOrigin::Coercion(_) => " for automatic coercion".to_string(),
            RegionVariableOrigin::BoundRegion(_, br, BoundRegionConversionTime::FnCall) => {
                format!(" for lifetime parameter {}in function call", br_string(br))
            }
            RegionVariableOrigin::BoundRegion(
                _,
                br,
                BoundRegionConversionTime::HigherRankedType,
            ) => {
                format!(" for lifetime parameter {}in generic type", br_string(br))
            }
            RegionVariableOrigin::BoundRegion(
                _,
                br,
                BoundRegionConversionTime::AssocTypeProjection(def_id),
            ) => format!(
                " for lifetime parameter {}in trait containing associated type `{}`",
                br_string(br),
                self.tcx.associated_item(def_id).name()
            ),
            RegionVariableOrigin::RegionParameterDefinition(_, name) => {
                format!(" for lifetime parameter `{name}`")
            }
            RegionVariableOrigin::UpvarRegion(ref upvar_id, _) => {
                let var_name = self.tcx.hir_name(upvar_id.var_path.hir_id);
                format!(" for capture of `{var_name}` by closure")
            }
            RegionVariableOrigin::Nll(..) => bug!("NLL variable found in lexical phase"),
        };

        struct_span_code_err!(
            self.dcx(),
            var_origin.span(),
            E0803,
            "cannot infer an appropriate lifetime{} due to conflicting requirements",
            var_description
        )
    }
}

pub(super) fn note_and_explain_region<'tcx>(
    tcx: TyCtxt<'tcx>,
    err: &mut Diag<'_>,
    generic_param_scope: LocalDefId,
    prefix: &str,
    region: ty::Region<'tcx>,
    suffix: &str,
    alt_span: Option<Span>,
) {
    let (description, span) = match region.kind() {
        ty::ReEarlyParam(_) | ty::ReLateParam(_) | ty::RePlaceholder(_) | ty::ReStatic => {
            msg_span_from_named_region(tcx, generic_param_scope, region, alt_span)
        }

        ty::ReError(_) => return,

        // FIXME(#125431): `ReVar` shouldn't reach here.
        ty::ReVar(_) => (format!("lifetime `{region}`"), alt_span),

        ty::ReBound(..) | ty::ReErased => {
            bug!("unexpected region for note_and_explain_region: {:?}", region);
        }
    };

    emit_msg_span(err, prefix, description, span, suffix);
}

fn explain_free_region<'tcx>(
    tcx: TyCtxt<'tcx>,
    err: &mut Diag<'_>,
    generic_param_scope: LocalDefId,
    prefix: &str,
    region: ty::Region<'tcx>,
    suffix: &str,
) {
    let (description, span) = msg_span_from_named_region(tcx, generic_param_scope, region, None);

    label_msg_span(err, prefix, description, span, suffix);
}

fn msg_span_from_named_region<'tcx>(
    tcx: TyCtxt<'tcx>,
    generic_param_scope: LocalDefId,
    region: ty::Region<'tcx>,
    alt_span: Option<Span>,
) -> (String, Option<Span>) {
    match region.kind() {
        ty::ReEarlyParam(br) => {
            let param_def_id = tcx.generics_of(generic_param_scope).region_param(br, tcx).def_id;
            let span = tcx.def_span(param_def_id);
            let text = if br.has_name() {
                format!("the lifetime `{}` as defined here", br.name)
            } else {
                "the anonymous lifetime as defined here".to_string()
            };
            (text, Some(span))
        }
        ty::ReLateParam(ref fr) => {
            if !fr.kind.is_named()
                && let Some((ty, _)) = find_anon_type(tcx, generic_param_scope, region)
            {
                ("the anonymous lifetime defined here".to_string(), Some(ty.span))
            } else {
                match fr.kind {
                    ty::LateParamRegionKind::Named(param_def_id, name) => {
                        let span = tcx.def_span(param_def_id);
                        let text = if name == kw::UnderscoreLifetime {
                            "the anonymous lifetime as defined here".to_string()
                        } else {
                            format!("the lifetime `{name}` as defined here")
                        };
                        (text, Some(span))
                    }
                    ty::LateParamRegionKind::Anon(_) => (
                        "the anonymous lifetime as defined here".to_string(),
                        Some(tcx.def_span(generic_param_scope)),
                    ),
                    _ => (
                        format!("the lifetime `{region}` as defined here"),
                        Some(tcx.def_span(generic_param_scope)),
                    ),
                }
            }
        }
        ty::ReStatic => ("the static lifetime".to_owned(), alt_span),
        ty::RePlaceholder(ty::PlaceholderRegion {
            bound: ty::BoundRegion { kind: ty::BoundRegionKind::Named(def_id, name), .. },
            ..
        }) => (format!("the lifetime `{name}` as defined here"), Some(tcx.def_span(def_id))),
        ty::RePlaceholder(ty::PlaceholderRegion {
            bound: ty::BoundRegion { kind: ty::BoundRegionKind::Anon, .. },
            ..
        }) => ("an anonymous lifetime".to_owned(), None),
        _ => bug!("{:?}", region),
    }
}

fn emit_msg_span(
    err: &mut Diag<'_>,
    prefix: &str,
    description: String,
    span: Option<Span>,
    suffix: &str,
) {
    let message = format!("{prefix}{description}{suffix}");

    if let Some(span) = span {
        err.span_note(span, message);
    } else {
        err.note(message);
    }
}

fn label_msg_span(
    err: &mut Diag<'_>,
    prefix: &str,
    description: String,
    span: Option<Span>,
    suffix: &str,
) {
    let message = format!("{prefix}{description}{suffix}");

    if let Some(span) = span {
        err.span_label(span, message);
    } else {
        err.note(message);
    }
}

#[instrument(level = "trace", skip(infcx))]
pub fn unexpected_hidden_region_diagnostic<'a, 'tcx>(
    infcx: &'a InferCtxt<'tcx>,
    generic_param_scope: LocalDefId,
    span: Span,
    hidden_ty: Ty<'tcx>,
    hidden_region: ty::Region<'tcx>,
    opaque_ty_key: ty::OpaqueTypeKey<'tcx>,
) -> Diag<'a> {
    let tcx = infcx.tcx;
    let mut err = infcx.dcx().create_err(errors::OpaqueCapturesLifetime {
        span,
        opaque_ty: Ty::new_opaque(tcx, opaque_ty_key.def_id.to_def_id(), opaque_ty_key.args),
        opaque_ty_span: tcx.def_span(opaque_ty_key.def_id),
    });

    // Explain the region we are capturing.
    match hidden_region.kind() {
        ty::ReEarlyParam(_) | ty::ReLateParam(_) | ty::ReStatic => {
            // Assuming regionck succeeded (*), we ought to always be
            // capturing *some* region from the fn header, and hence it
            // ought to be free. So under normal circumstances, we will go
            // down this path which gives a decent human readable
            // explanation.
            //
            // (*) if not, the `tainted_by_errors` field would be set to
            // `Some(ErrorGuaranteed)` in any case, so we wouldn't be here at all.
            explain_free_region(
                tcx,
                &mut err,
                generic_param_scope,
                &format!("hidden type `{hidden_ty}` captures "),
                hidden_region,
                "",
            );
            if let Some(_) = tcx.is_suitable_region(generic_param_scope, hidden_region) {
                suggest_precise_capturing(tcx, opaque_ty_key.def_id, hidden_region, &mut err);
            }
        }
        ty::RePlaceholder(_) => {
            explain_free_region(
                tcx,
                &mut err,
                generic_param_scope,
                &format!("hidden type `{}` captures ", hidden_ty),
                hidden_region,
                "",
            );
        }
        ty::ReError(_) => {
            err.downgrade_to_delayed_bug();
        }
        _ => {
            // Ugh. This is a painful case: the hidden region is not one
            // that we can easily summarize or explain. This can happen
            // in a case like
            // `tests/ui/multiple-lifetimes/ordinary-bounds-unsuited.rs`:
            //
            // ```
            // fn upper_bounds<'a, 'b>(a: Ordinary<'a>, b: Ordinary<'b>) -> impl Trait<'a, 'b> {
            //   if condition() { a } else { b }
            // }
            // ```
            //
            // Here the captured lifetime is the intersection of `'a` and
            // `'b`, which we can't quite express.

            // We can at least report a really cryptic error for now.
            note_and_explain_region(
                tcx,
                &mut err,
                generic_param_scope,
                &format!("hidden type `{hidden_ty}` captures "),
                hidden_region,
                "",
                None,
            );
        }
    }

    err
}

fn suggest_precise_capturing<'tcx>(
    tcx: TyCtxt<'tcx>,
    opaque_def_id: LocalDefId,
    captured_lifetime: ty::Region<'tcx>,
    diag: &mut Diag<'_>,
) {
    let hir::OpaqueTy { bounds, origin, .. } =
        tcx.hir_node_by_def_id(opaque_def_id).expect_opaque_ty();

    let hir::OpaqueTyOrigin::FnReturn { parent: fn_def_id, .. } = *origin else {
        return;
    };

    let new_lifetime = Symbol::intern(&captured_lifetime.to_string());

    if let Some((args, span)) = bounds.iter().find_map(|bound| match bound {
        hir::GenericBound::Use(args, span) => Some((args, span)),
        _ => None,
    }) {
        let last_lifetime_span = args.iter().rev().find_map(|arg| match arg {
            hir::PreciseCapturingArg::Lifetime(lt) => Some(lt.ident.span),
            _ => None,
        });

        let first_param_span = args.iter().find_map(|arg| match arg {
            hir::PreciseCapturingArg::Param(p) => Some(p.ident.span),
            _ => None,
        });

        let (span, pre, post) = if let Some(last_lifetime_span) = last_lifetime_span {
            (last_lifetime_span.shrink_to_hi(), ", ", "")
        } else if let Some(first_param_span) = first_param_span {
            (first_param_span.shrink_to_lo(), "", ", ")
        } else {
            // If we have no args, then have `use<>` and need to fall back to using
            // span math. This sucks, but should be reliable due to the construction
            // of the `use<>` span.
            (span.with_hi(span.hi() - BytePos(1)).shrink_to_hi(), "", "")
        };

        diag.subdiagnostic(errors::AddPreciseCapturing::Existing { span, new_lifetime, pre, post });
    } else {
        let mut captured_lifetimes = FxIndexSet::default();
        let mut captured_non_lifetimes = FxIndexSet::default();

        let variances = tcx.variances_of(opaque_def_id);
        let mut generics = tcx.generics_of(opaque_def_id);
        let mut synthetics = vec![];
        loop {
            for param in &generics.own_params {
                if variances[param.index as usize] == ty::Bivariant {
                    continue;
                }

                match param.kind {
                    ty::GenericParamDefKind::Lifetime => {
                        captured_lifetimes.insert(param.name);
                    }
                    ty::GenericParamDefKind::Type { synthetic: true, .. } => {
                        synthetics.push((tcx.def_span(param.def_id), param.name));
                    }
                    ty::GenericParamDefKind::Type { .. }
                    | ty::GenericParamDefKind::Const { .. } => {
                        captured_non_lifetimes.insert(param.name);
                    }
                }
            }

            if let Some(parent) = generics.parent {
                generics = tcx.generics_of(parent);
            } else {
                break;
            }
        }

        if !captured_lifetimes.insert(new_lifetime) {
            // Uh, strange. This lifetime appears to already be captured...
            return;
        }

        if synthetics.is_empty() {
            let concatenated_bounds = captured_lifetimes
                .into_iter()
                .chain(captured_non_lifetimes)
                .map(|sym| sym.to_string())
                .collect::<Vec<_>>()
                .join(", ");

            diag.subdiagnostic(errors::AddPreciseCapturing::New {
                span: tcx.def_span(opaque_def_id).shrink_to_hi(),
                new_lifetime,
                concatenated_bounds,
            });
        } else {
            let mut next_fresh_param = || {
                ["T", "U", "V", "W", "X", "Y", "A", "B", "C"]
                    .into_iter()
                    .map(Symbol::intern)
                    .chain((0..).map(|i| Symbol::intern(&format!("T{i}"))))
                    .find(|s| captured_non_lifetimes.insert(*s))
                    .unwrap()
            };

            let mut new_params = String::new();
            let mut suggs = vec![];
            let mut apit_spans = vec![];

            for (i, (span, name)) in synthetics.into_iter().enumerate() {
                apit_spans.push(span);

                let fresh_param = next_fresh_param();

                // Suggest renaming.
                suggs.push((span, fresh_param.to_string()));

                // Super jank. Turn `impl Trait` into `T: Trait`.
                //
                // This currently involves stripping the `impl` from the name of
                // the parameter, since APITs are always named after how they are
                // rendered in the AST. This sucks! But to recreate the bound list
                // from the APIT itself would be miserable, so we're stuck with
                // this for now!
                if i > 0 {
                    new_params += ", ";
                }
                let name_as_bounds = name.as_str().trim_start_matches("impl").trim_start();
                new_params += fresh_param.as_str();
                new_params += ": ";
                new_params += name_as_bounds;
            }

            let Some(generics) = tcx.hir_get_generics(fn_def_id) else {
                // This shouldn't happen, but don't ICE.
                return;
            };

            // Add generics or concatenate to the end of the list.
            suggs.push(if let Some(params_span) = generics.span_for_param_suggestion() {
                (params_span, format!(", {new_params}"))
            } else {
                (generics.span, format!("<{new_params}>"))
            });

            let concatenated_bounds = captured_lifetimes
                .into_iter()
                .chain(captured_non_lifetimes)
                .map(|sym| sym.to_string())
                .collect::<Vec<_>>()
                .join(", ");

            suggs.push((
                tcx.def_span(opaque_def_id).shrink_to_hi(),
                format!(" + use<{concatenated_bounds}>"),
            ));

            diag.subdiagnostic(errors::AddPreciseCapturingAndParams {
                suggs,
                new_lifetime,
                apit_spans,
            });
        }
    }
}
