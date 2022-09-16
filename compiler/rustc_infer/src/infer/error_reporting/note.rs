use crate::errors::RegionOriginNote;
use crate::infer::error_reporting::note_and_explain_region;
use crate::infer::{self, InferCtxt, SubregionOrigin};
use rustc_errors::{
    fluent, struct_span_err, AddSubdiagnostic, Diagnostic, DiagnosticBuilder, ErrorGuaranteed,
};
use rustc_middle::traits::ObligationCauseCode;
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::{self, Region};

use super::ObligationCauseAsDiagArg;

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    pub(super) fn note_region_origin(&self, err: &mut Diagnostic, origin: &SubregionOrigin<'tcx>) {
        match *origin {
            infer::Subtype(ref trace) => RegionOriginNote::WithRequirement {
                span: trace.cause.span,
                requirement: ObligationCauseAsDiagArg(trace.cause.clone()),
                expected_found: self.values_str(trace.values),
            }
            .add_to_diagnostic(err),
            infer::Reborrow(span) => RegionOriginNote::Plain { span, msg: fluent::infer::reborrow }
                .add_to_diagnostic(err),
            infer::ReborrowUpvar(span, ref upvar_id) => {
                let var_name = self.tcx.hir().name(upvar_id.var_path.hir_id);
                RegionOriginNote::WithName {
                    span,
                    msg: fluent::infer::reborrow,
                    name: &var_name.to_string(),
                    continues: false,
                }
                .add_to_diagnostic(err);
            }
            infer::RelateObjectBound(span) => {
                RegionOriginNote::Plain { span, msg: fluent::infer::relate_object_bound }
                    .add_to_diagnostic(err);
            }
            infer::DataBorrowed(ty, span) => {
                RegionOriginNote::WithName {
                    span,
                    msg: fluent::infer::data_borrowed,
                    name: &self.ty_to_string(ty),
                    continues: false,
                }
                .add_to_diagnostic(err);
            }
            infer::ReferenceOutlivesReferent(ty, span) => {
                RegionOriginNote::WithName {
                    span,
                    msg: fluent::infer::reference_outlives_referent,
                    name: &self.ty_to_string(ty),
                    continues: false,
                }
                .add_to_diagnostic(err);
            }
            infer::RelateParamBound(span, ty, opt_span) => {
                RegionOriginNote::WithName {
                    span,
                    msg: fluent::infer::relate_param_bound,
                    name: &self.ty_to_string(ty),
                    continues: opt_span.is_some(),
                }
                .add_to_diagnostic(err);
                if let Some(span) = opt_span {
                    RegionOriginNote::Plain { span, msg: fluent::infer::relate_param_bound_2 }
                        .add_to_diagnostic(err);
                }
            }
            infer::RelateRegionParamBound(span) => {
                RegionOriginNote::Plain { span, msg: fluent::infer::relate_region_param_bound }
                    .add_to_diagnostic(err);
            }
            infer::CompareImplItemObligation { span, .. } => {
                RegionOriginNote::Plain { span, msg: fluent::infer::compare_impl_item_obligation }
                    .add_to_diagnostic(err);
            }
            infer::CheckAssociatedTypeBounds { ref parent, .. } => {
                self.note_region_origin(err, &parent);
            }
        }
    }

    pub(super) fn report_concrete_failure(
        &self,
        origin: SubregionOrigin<'tcx>,
        sub: Region<'tcx>,
        sup: Region<'tcx>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        match origin {
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
            infer::ReborrowUpvar(span, ref upvar_id) => {
                let var_name = self.tcx.hir().name(upvar_id.var_path.hir_id);
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0313,
                    "lifetime of borrowed pointer outlives lifetime of captured variable `{}`...",
                    var_name
                );
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "...the borrowed pointer is valid for ",
                    sub,
                    "...",
                    None,
                );
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    &format!("...but `{}` is only valid for ", var_name),
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
            infer::DataBorrowed(ty, span) => {
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0490,
                    "a value of type `{}` is borrowed for too long",
                    self.ty_to_string(ty)
                );
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "the type is valid for ",
                    sub,
                    "",
                    None,
                );
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "but the borrow lasts for ",
                    sup,
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
            infer::CompareImplItemObligation { span, impl_item_def_id, trait_item_def_id } => self
                .report_extra_impl_obligation(
                    span,
                    impl_item_def_id,
                    trait_item_def_id,
                    &format!("`{}: {}`", sup, sub),
                ),
            infer::CheckAssociatedTypeBounds { impl_item_def_id, trait_item_def_id, parent } => {
                let mut err = self.report_concrete_failure(*parent, sub, sup);

                let trait_item_span = self.tcx.def_span(trait_item_def_id);
                let item_name = self.tcx.item_name(impl_item_def_id.to_def_id());
                err.span_label(
                    trait_item_span,
                    format!("definition of `{}` from trait", item_name),
                );

                let trait_predicates = self.tcx.explicit_predicates_of(trait_item_def_id);
                let impl_predicates = self.tcx.explicit_predicates_of(impl_item_def_id);

                let impl_predicates: rustc_data_structures::fx::FxHashSet<_> =
                    impl_predicates.predicates.into_iter().map(|(pred, _)| pred).collect();
                let clauses: Vec<_> = trait_predicates
                    .predicates
                    .into_iter()
                    .filter(|&(pred, _)| !impl_predicates.contains(pred))
                    .map(|(pred, _)| format!("{}", pred))
                    .collect();

                if !clauses.is_empty() {
                    let generics = self.tcx.hir().get_generics(impl_item_def_id).unwrap();
                    let where_clause_span = generics.tail_span_for_predicate_suggestion();

                    let suggestion = format!(
                        "{} {}",
                        generics.add_where_or_trailing_comma(),
                        clauses.join(", "),
                    );
                    err.span_suggestion(
                        where_clause_span,
                        &format!(
                            "try copying {} from the trait",
                            if clauses.len() > 1 { "these clauses" } else { "this clause" }
                        ),
                        suggestion,
                        rustc_errors::Applicability::MaybeIncorrect,
                    );
                }

                err
            }
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
