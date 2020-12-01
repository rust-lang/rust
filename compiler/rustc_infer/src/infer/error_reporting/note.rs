use crate::infer::error_reporting::{note_and_explain_region, ObligationCauseExt};
use crate::infer::{self, InferCtxt, SubregionOrigin};
use rustc_errors::{struct_span_err, DiagnosticBuilder};
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
            infer::CallReturn(span) => {
                label_or_note(span, "...so that return value is valid for the call");
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
            infer::RelateParamBound(span, t) => {
                label_or_note(
                    span,
                    &format!(
                        "...so that the type `{}` will meet its required lifetime bounds",
                        self.ty_to_string(t)
                    ),
                );
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
                note_and_explain_region(self.tcx, &mut err, "", sup, "...");
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "...does not necessarily outlive ",
                    sub,
                    "",
                );
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
                );
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "...but the borrowed content is only valid for ",
                    sup,
                    "",
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
                );
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    &format!("...but `{}` is only valid for ", var_name),
                    sup,
                    "",
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
                note_and_explain_region(self.tcx, &mut err, "object type is valid for ", sub, "");
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "source pointer is only valid for ",
                    sup,
                    "",
                );
                err
            }
            infer::RelateParamBound(span, ty) => {
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0477,
                    "the type `{}` does not fulfill the required lifetime",
                    self.ty_to_string(ty)
                );
                match *sub {
                    ty::ReStatic => {
                        note_and_explain_region(self.tcx, &mut err, "type must satisfy ", sub, "")
                    }
                    _ => note_and_explain_region(self.tcx, &mut err, "type must outlive ", sub, ""),
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
                );
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "but lifetime parameter must outlive ",
                    sub,
                    "",
                );
                err
            }
            infer::CallReturn(span) => {
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0482,
                    "lifetime of return value does not outlive the function call"
                );
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "the return value is only valid for ",
                    sup,
                    "",
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
                note_and_explain_region(self.tcx, &mut err, "the type is valid for ", sub, "");
                note_and_explain_region(self.tcx, &mut err, "but the borrow lasts for ", sup, "");
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
                note_and_explain_region(self.tcx, &mut err, "the pointer is valid for ", sub, "");
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    "but the referenced data is only valid for ",
                    sup,
                    "",
                );
                err
            }
            infer::CompareImplMethodObligation {
                span,
                item_name,
                impl_item_def_id,
                trait_item_def_id,
            } => self.report_extra_impl_obligation(
                span,
                item_name,
                impl_item_def_id,
                trait_item_def_id,
                &format!("`{}: {}`", sup, sub),
            ),
        }
    }

    pub(super) fn report_placeholder_failure(
        &self,
        placeholder_origin: SubregionOrigin<'tcx>,
        sub: Region<'tcx>,
        sup: Region<'tcx>,
    ) -> DiagnosticBuilder<'tcx> {
        // I can't think how to do better than this right now. -nikomatsakis
        match placeholder_origin {
            infer::Subtype(box trace) => {
                let terr = TypeError::RegionsPlaceholderMismatch;
                self.report_and_explain_type_error(trace, &terr)
            }

            _ => self.report_concrete_failure(placeholder_origin, sub, sup),
        }
    }
}
