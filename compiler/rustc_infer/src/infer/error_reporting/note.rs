use crate::errors::RegionOriginNote;
use crate::infer::error_reporting::{note_and_explain_region, TypeErrCtxt};
use crate::infer::{self, SubregionOrigin};
use rustc_errors::{
    fluent, struct_span_err, AddToDiagnostic, Applicability, Diagnostic, DiagnosticBuilder,
    ErrorGuaranteed,
};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::traits::ObligationCauseCode;
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::{self, IsSuggestable, Region};
use rustc_span::symbol::kw;

use super::ObligationCauseAsDiagArg;

impl<'tcx> TypeErrCtxt<'_, 'tcx> {
    pub(super) fn note_region_origin(&self, err: &mut Diagnostic, origin: &SubregionOrigin<'tcx>) {
        match *origin {
            infer::Subtype(ref trace) => RegionOriginNote::WithRequirement {
                span: trace.cause.span,
                requirement: ObligationCauseAsDiagArg(trace.cause.clone()),
                expected_found: self.values_str(trace.values).map(|(e, f, _, _)| (e, f)),
            }
            .add_to_diagnostic(err),
            infer::Reborrow(span) => {
                RegionOriginNote::Plain { span, msg: fluent::infer_reborrow }.add_to_diagnostic(err)
            }
            infer::RelateObjectBound(span) => {
                RegionOriginNote::Plain { span, msg: fluent::infer_relate_object_bound }
                    .add_to_diagnostic(err);
            }
            infer::ReferenceOutlivesReferent(ty, span) => {
                RegionOriginNote::WithName {
                    span,
                    msg: fluent::infer_reference_outlives_referent,
                    name: &self.ty_to_string(ty),
                    continues: false,
                }
                .add_to_diagnostic(err);
            }
            infer::RelateParamBound(span, ty, opt_span) => {
                RegionOriginNote::WithName {
                    span,
                    msg: fluent::infer_relate_param_bound,
                    name: &self.ty_to_string(ty),
                    continues: opt_span.is_some(),
                }
                .add_to_diagnostic(err);
                if let Some(span) = opt_span {
                    RegionOriginNote::Plain { span, msg: fluent::infer_relate_param_bound_2 }
                        .add_to_diagnostic(err);
                }
            }
            infer::RelateRegionParamBound(span) => {
                RegionOriginNote::Plain { span, msg: fluent::infer_relate_region_param_bound }
                    .add_to_diagnostic(err);
            }
            infer::CompareImplItemObligation { span, .. } => {
                RegionOriginNote::Plain { span, msg: fluent::infer_compare_impl_item_obligation }
                    .add_to_diagnostic(err);
            }
            infer::CheckAssociatedTypeBounds { ref parent, .. } => {
                self.note_region_origin(err, &parent);
            }
            infer::AscribeUserTypeProvePredicate(span) => {
                RegionOriginNote::Plain {
                    span,
                    msg: fluent::infer_ascribe_user_type_prove_predicate,
                }
                .add_to_diagnostic(err);
            }
        }
    }

    pub(super) fn report_concrete_failure(
        &self,
        origin: SubregionOrigin<'tcx>,
        sub: Region<'tcx>,
        sup: Region<'tcx>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
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
                            "the required lifetime does not necessarily outlive ",
                            sub,
                            "",
                            None,
                        );
                    }
                    _ => {
                        note_and_explain_region(self.tcx, &mut err, "", sup, "...", None);
                        note_and_explain_region(
                            self.tcx,
                            &mut err,
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
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0312,
                    "lifetime of reference outlives lifetime of borrowed content..."
                );
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "...the reference is valid for ",
                    sub,
                    "...",
                    None,
                );
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "...but the borrowed content is only valid for ",
                    sup,
                    "",
                    None,
                );
                err
            }
            infer::RelateObjectBound(span) => {
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0476,
                    "lifetime of the source pointer does not outlive lifetime bound of the \
                     object type"
                );
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "object type is valid for ",
                    sub,
                    "",
                    None,
                );
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "source pointer is only valid for ",
                    sup,
                    "",
                    None,
                );
                err
            }
            infer::RelateParamBound(span, ty, opt_span) => {
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0477,
                    "the type `{}` does not fulfill the required lifetime",
                    self.ty_to_string(ty)
                );
                match *sub {
                    ty::ReStatic => note_and_explain_region(
                        self.tcx,
                        &mut err,
                        "type must satisfy ",
                        sub,
                        if opt_span.is_some() { " as required by this binding" } else { "" },
                        opt_span,
                    ),
                    _ => note_and_explain_region(
                        self.tcx,
                        &mut err,
                        "type must outlive ",
                        sub,
                        if opt_span.is_some() { " as required by this binding" } else { "" },
                        opt_span,
                    ),
                }
                err
            }
            infer::RelateRegionParamBound(span) => {
                let mut err =
                    struct_span_err!(self.tcx.sess, span, E0478, "lifetime bound not satisfied");
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "lifetime parameter instantiated with ",
                    sup,
                    "",
                    None,
                );
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "but lifetime parameter must outlive ",
                    sub,
                    "",
                    None,
                );
                err
            }
            infer::ReferenceOutlivesReferent(ty, span) => {
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0491,
                    "in type `{}`, reference has a longer lifetime than the data it references",
                    self.ty_to_string(ty)
                );
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "the pointer is valid for ",
                    sub,
                    "",
                    None,
                );
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "but the referenced data is only valid for ",
                    sup,
                    "",
                    None,
                );
                err
            }
            infer::CompareImplItemObligation { span, impl_item_def_id, trait_item_def_id } => {
                let mut err = self.report_extra_impl_obligation(
                    span,
                    impl_item_def_id,
                    trait_item_def_id,
                    &format!("`{}: {}`", sup, sub),
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
                let mut err = self.report_concrete_failure(*parent, sub, sup);
                let trait_item_span = self.tcx.def_span(trait_item_def_id);
                let item_name = self.tcx.item_name(impl_item_def_id.to_def_id());
                err.span_label(
                    trait_item_span,
                    format!("definition of `{}` from trait", item_name),
                );
                self.suggest_copy_trait_method_bounds(
                    trait_item_def_id,
                    impl_item_def_id,
                    &mut err,
                );
                err
            }
            infer::AscribeUserTypeProvePredicate(span) => {
                let mut err =
                    struct_span_err!(self.tcx.sess, span, E0478, "lifetime bound not satisfied");
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "lifetime instantiated with ",
                    sup,
                    "",
                    None,
                );
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "but lifetime must outlive ",
                    sub,
                    "",
                    None,
                );
                err
            }
        };
        if sub.is_error() || sup.is_error() {
            err.delay_as_bug();
        }
        err
    }

    pub fn suggest_copy_trait_method_bounds(
        &self,
        trait_item_def_id: DefId,
        impl_item_def_id: LocalDefId,
        err: &mut Diagnostic,
    ) {
        // FIXME(compiler-errors): Right now this is only being used for region
        // predicate mismatches. Ideally, we'd use it for *all* predicate mismatches,
        // but right now it's not really very smart when it comes to implicit `Sized`
        // predicates and bounds on the trait itself.

        let Some(impl_def_id) =
            self.tcx.associated_item(impl_item_def_id).impl_container(self.tcx) else { return; };
        let Some(trait_ref) = self
            .tcx
            .impl_trait_ref(impl_def_id)
            else { return; };
        let trait_substs = trait_ref
            .subst_identity()
            // Replace the explicit self type with `Self` for better suggestion rendering
            .with_self_ty(self.tcx, self.tcx.mk_ty_param(0, kw::SelfUpper))
            .substs;
        let trait_item_substs =
            ty::InternalSubsts::identity_for_item(self.tcx, impl_item_def_id.to_def_id())
                .rebase_onto(self.tcx, impl_def_id, trait_substs);

        let Ok(trait_predicates) = self
            .tcx
            .explicit_predicates_of(trait_item_def_id)
            .instantiate_own(self.tcx, trait_item_substs)
            .map(|(pred, _)| {
                if pred.is_suggestable(self.tcx, false) {
                    Ok(pred.to_string())
                } else {
                    Err(())
                }
            })
            .collect::<Result<Vec<_>, ()>>() else { return; };

        let Some(generics) = self.tcx.hir().get_generics(impl_item_def_id) else { return; };

        if trait_predicates.is_empty() {
            err.span_suggestion_verbose(
                generics.where_clause_span,
                "remove the `where` clause",
                String::new(),
                Applicability::MachineApplicable,
            );
        } else {
            let space = if generics.where_clause_span.is_empty() { " " } else { "" };
            err.span_suggestion_verbose(
                generics.where_clause_span,
                "copy the `where` clause predicates from the trait",
                format!("{space}where {}", trait_predicates.join(", ")),
                Applicability::MachineApplicable,
            );
        }
    }

    pub(super) fn report_placeholder_failure(
        &self,
        placeholder_origin: SubregionOrigin<'tcx>,
        sub: Region<'tcx>,
        sup: Region<'tcx>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        // I can't think how to do better than this right now. -nikomatsakis
        debug!(?placeholder_origin, ?sub, ?sup, "report_placeholder_failure");
        match placeholder_origin {
            infer::Subtype(box ref trace)
                if matches!(
                    &trace.cause.code().peel_derives(),
                    ObligationCauseCode::BindingObligation(..)
                        | ObligationCauseCode::ExprBindingObligation(..)
                ) =>
            {
                // Hack to get around the borrow checker because trace.cause has an `Rc`.
                if let ObligationCauseCode::BindingObligation(_, span)
                | ObligationCauseCode::ExprBindingObligation(_, span, ..) =
                    &trace.cause.code().peel_derives()
                {
                    let span = *span;
                    let mut err = self.report_concrete_failure(placeholder_origin, sub, sup);
                    err.span_note(span, "the lifetime requirement is introduced here");
                    err
                } else {
                    unreachable!()
                }
            }
            infer::Subtype(box trace) => {
                let terr = TypeError::RegionsPlaceholderMismatch;
                return self.report_and_explain_type_error(trace, terr);
            }
            _ => return self.report_concrete_failure(placeholder_origin, sub, sup),
        }
    }
}
