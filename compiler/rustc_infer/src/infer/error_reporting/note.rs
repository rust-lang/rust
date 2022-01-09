use crate::infer::error_reporting::{note_and_explain_region, ObligationCauseExt};
use crate::infer::{self, InferCtxt, SubregionOrigin};
use rustc_errors::{struct_span_err, DiagnosticBuilder};
use rustc_middle::traits::ObligationCauseCode;
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::{self, Region};

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    pub(super) fn note_region_origin(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        origin: &SubregionOrigin<'tcx>,
    ) {
        let mut label_or_note = |span, msg| {
            let sub_count = err.children.iter().filter(|d| d.span.is_dummy()).count();
            let expanded_sub_count = err.children.iter().filter(|d| !d.span.is_dummy()).count();
            let span_is_primary = err.span.primary_spans().iter().all(|&sp| sp == span);
            if span_is_primary && sub_count == 0 && expanded_sub_count == 0 {
                err.span_label(span, msg);
            } else if span_is_primary && expanded_sub_count == 0 {
                err.note(msg);
            } else {
                err.span_note(span, msg);
            }
        };
        match *origin {
            infer::Subtype(ref trace) => {
                if let Some((expected, found)) = self.values_str(trace.values) {
                    label_or_note(
                        trace.cause.span,
                        &format!("...so that the {}", trace.cause.as_requirement_str()),
                    );

                    err.note_expected_found(&"", expected, &"", found);
                } else {
                    // FIXME: this really should be handled at some earlier stage. Our
                    // handling of region checking when type errors are present is
                    // *terrible*.

                    label_or_note(
                        trace.cause.span,
                        &format!("...so that {}", trace.cause.as_requirement_str()),
                    );
                }
            }
            infer::Reborrow(span) => {
                label_or_note(span, "...so that reference does not outlive borrowed content");
            }
            infer::ReborrowUpvar(span, ref upvar_id) => {
                let var_name = self.tcx.hir().name(upvar_id.var_path.hir_id);
                label_or_note(span, &format!("...so that closure can access `{}`", var_name));
            }
            infer::RelateObjectBound(span) => {
                label_or_note(span, "...so that it can be closed over into an object");
            }
            infer::DataBorrowed(ty, span) => {
                label_or_note(
                    span,
                    &format!(
                        "...so that the type `{}` is not borrowed for too long",
                        self.ty_to_string(ty)
                    ),
                );
            }
            infer::ReferenceOutlivesReferent(ty, span) => {
                label_or_note(
                    span,
                    &format!(
                        "...so that the reference type `{}` does not outlive the data it points at",
                        self.ty_to_string(ty)
                    ),
                );
            }
            infer::RelateParamBound(span, t, opt_span) => {
                label_or_note(
                    span,
                    &format!(
                        "...so that the type `{}` will meet its required lifetime bounds{}",
                        self.ty_to_string(t),
                        if opt_span.is_some() { "..." } else { "" },
                    ),
                );
                if let Some(span) = opt_span {
                    err.span_note(span, "...that is required by this bound");
                }
            }
            infer::RelateRegionParamBound(span) => {
                label_or_note(
                    span,
                    "...so that the declared lifetime parameter bounds are satisfied",
                );
            }
            infer::CompareImplMethodObligation { span, .. } => {
                label_or_note(
                    span,
                    "...so that the definition in impl matches the definition from the trait",
                );
            }
            infer::CompareImplTypeObligation { span, .. } => {
                label_or_note(
                    span,
                    "...so that the definition in impl matches the definition from the trait",
                );
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
    ) -> DiagnosticBuilder<'tcx> {
        match origin {
            infer::Subtype(box trace) => {
                let terr = TypeError::RegionsDoesNotOutlive(sup, sub);
                let mut err = self.report_and_explain_type_error(trace, &terr);
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
            infer::CompareImplMethodObligation { span, impl_item_def_id, trait_item_def_id } => {
                self.report_extra_impl_obligation(
                    span,
                    impl_item_def_id,
                    trait_item_def_id,
                    &format!("`{}: {}`", sup, sub),
                )
            }
            infer::CompareImplTypeObligation { span, impl_item_def_id, trait_item_def_id } => self
                .report_extra_impl_obligation(
                    span,
                    impl_item_def_id,
                    trait_item_def_id,
                    &format!("`{}: {}`", sup, sub),
                ),
            infer::CheckAssociatedTypeBounds { impl_item_def_id, trait_item_def_id, parent } => {
                let mut err = self.report_concrete_failure(*parent, sub, sup);

                let trait_item_span = self.tcx.def_span(trait_item_def_id);
                let item_name = self.tcx.item_name(impl_item_def_id);
                err.span_label(
                    trait_item_span,
                    format!("definition of `{}` from trait", item_name),
                );

                let trait_predicates = self.tcx.explicit_predicates_of(trait_item_def_id);
                let impl_predicates = self.tcx.explicit_predicates_of(impl_item_def_id);

                let impl_predicates: rustc_data_structures::stable_set::FxHashSet<_> =
                    impl_predicates.predicates.into_iter().map(|(pred, _)| pred).collect();
                let clauses: Vec<_> = trait_predicates
                    .predicates
                    .into_iter()
                    .filter(|&(pred, _)| !impl_predicates.contains(pred))
                    .map(|(pred, _)| format!("{}", pred))
                    .collect();

                if !clauses.is_empty() {
                    let where_clause_span = self
                        .tcx
                        .hir()
                        .get_generics(impl_item_def_id.expect_local())
                        .unwrap()
                        .where_clause
                        .tail_span_for_suggestion();

                    let suggestion = format!(
                        "{} {}",
                        if !impl_predicates.is_empty() { "," } else { " where" },
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
    ) -> DiagnosticBuilder<'tcx> {
        // I can't think how to do better than this right now. -nikomatsakis
        debug!(?placeholder_origin, ?sub, ?sup, "report_placeholder_failure");
        match placeholder_origin {
            infer::Subtype(box ref trace)
                if matches!(
                    &trace.cause.code().peel_derives(),
                    ObligationCauseCode::BindingObligation(..)
                ) =>
            {
                // Hack to get around the borrow checker because trace.cause has an `Rc`.
                if let ObligationCauseCode::BindingObligation(_, span) =
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
                return self.report_and_explain_type_error(trace, &terr);
            }
            _ => return self.report_concrete_failure(placeholder_origin, sub, sup),
        }
    }
}
