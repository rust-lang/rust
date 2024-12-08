use rustc_errors::{Diag, Subdiagnostic};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::traits::ObligationCauseCode;
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::{self, IsSuggestable, Region, Ty};
use rustc_span::symbol::kw;
use tracing::debug;

use super::ObligationCauseAsDiagArg;
use crate::error_reporting::infer::{TypeErrCtxt, note_and_explain_region};
use crate::errors::{
    FulfillReqLifetime, LfBoundNotSatisfied, OutlivesBound, OutlivesContent, RefLongerThanData,
    RegionOriginNote, WhereClauseSuggestions, note_and_explain,
};
use crate::fluent_generated as fluent;
use crate::infer::{self, SubregionOrigin};

impl<'a, 'tcx> TypeErrCtxt<'a, 'tcx> {
    pub(super) fn note_region_origin(&self, err: &mut Diag<'_>, origin: &SubregionOrigin<'tcx>) {
        match *origin {
            infer::Subtype(ref trace) => RegionOriginNote::WithRequirement {
                span: trace.cause.span,
                requirement: ObligationCauseAsDiagArg(trace.cause.clone()),
                expected_found: self.values_str(trace.values).map(|(e, f, _)| (e, f)),
            }
            .add_to_diag(err),
            infer::Reborrow(span) => {
                RegionOriginNote::Plain { span, msg: fluent::infer_reborrow }.add_to_diag(err)
            }
            infer::RelateObjectBound(span) => {
                RegionOriginNote::Plain { span, msg: fluent::infer_relate_object_bound }
                    .add_to_diag(err);
            }
            infer::ReferenceOutlivesReferent(ty, span) => {
                RegionOriginNote::WithName {
                    span,
                    msg: fluent::infer_reference_outlives_referent,
                    name: &self.ty_to_string(ty),
                    continues: false,
                }
                .add_to_diag(err);
            }
            infer::RelateParamBound(span, ty, opt_span) => {
                RegionOriginNote::WithName {
                    span,
                    msg: fluent::infer_relate_param_bound,
                    name: &self.ty_to_string(ty),
                    continues: opt_span.is_some(),
                }
                .add_to_diag(err);
                if let Some(span) = opt_span {
                    RegionOriginNote::Plain { span, msg: fluent::infer_relate_param_bound_2 }
                        .add_to_diag(err);
                }
            }
            infer::RelateRegionParamBound(span, _) => {
                RegionOriginNote::Plain { span, msg: fluent::infer_relate_region_param_bound }
                    .add_to_diag(err);
            }
            infer::CompareImplItemObligation { span, .. } => {
                RegionOriginNote::Plain { span, msg: fluent::infer_compare_impl_item_obligation }
                    .add_to_diag(err);
            }
            infer::CheckAssociatedTypeBounds { ref parent, .. } => {
                self.note_region_origin(err, parent);
            }
            infer::AscribeUserTypeProvePredicate(span) => {
                RegionOriginNote::Plain {
                    span,
                    msg: fluent::infer_ascribe_user_type_prove_predicate,
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
            infer::Subtype(box trace) => {
                let terr = TypeError::RegionsDoesNotOutlive(sup, sub);
                let mut err = self.report_and_explain_type_error(trace, terr);
                match (*sub, *sup) {
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
            infer::Reborrow(span) => {
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
            infer::RelateObjectBound(span) => {
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
            infer::RelateParamBound(span, ty, opt_span) => {
                let prefix = match *sub {
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
            infer::RelateRegionParamBound(span, _) => {
                let param_instantiated = note_and_explain::RegionExplanation::new(
                    self.tcx,
                    generic_param_scope,
                    sup,
                    None,
                    note_and_explain::PrefixKind::LfParamInstantiatedWith,
                    note_and_explain::SuffixKind::Empty,
                );
                let param_must_outlive = note_and_explain::RegionExplanation::new(
                    self.tcx,
                    generic_param_scope,
                    sub,
                    None,
                    note_and_explain::PrefixKind::LfParamMustOutlive,
                    note_and_explain::SuffixKind::Empty,
                );
                self.dcx().create_err(LfBoundNotSatisfied {
                    span,
                    notes: param_instantiated.into_iter().chain(param_must_outlive).collect(),
                })
            }
            infer::ReferenceOutlivesReferent(ty, span) => {
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
            infer::CompareImplItemObligation { span, impl_item_def_id, trait_item_def_id } => {
                let mut err = self.infcx.report_extra_impl_obligation(
                    span,
                    impl_item_def_id,
                    trait_item_def_id,
                    &format!("`{sup}: {sub}`"),
                );
                // We should only suggest rewriting the `where` clause if the predicate is within that `where` clause
                if let Some(generics) = self.tcx.hir().get_generics(impl_item_def_id)
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
            infer::CheckAssociatedTypeBounds { impl_item_def_id, trait_item_def_id, parent } => {
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
            infer::AscribeUserTypeProvePredicate(span) => {
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

        let Some(generics) = self.tcx.hir().get_generics(impl_item_def_id) else {
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
            infer::Subtype(box ref trace)
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
                    && !span.is_dummy()
                {
                    let span = *span;
                    self.report_concrete_failure(generic_param_scope, placeholder_origin, sub, sup)
                        .with_span_note(span, "the lifetime requirement is introduced here")
                } else {
                    unreachable!(
                        "control flow ensures we have a `BindingObligation` or `WhereClauseInExpr` here..."
                    )
                }
            }
            infer::Subtype(box trace) => {
                let terr = TypeError::RegionsPlaceholderMismatch;
                return self.report_and_explain_type_error(trace, terr);
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
}
