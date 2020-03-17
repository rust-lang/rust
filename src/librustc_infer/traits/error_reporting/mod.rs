use super::ObjectSafetyViolation;

use crate::infer::InferCtxt;
use rustc::ty::TyCtxt;
use rustc_ast::ast;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_span::Span;
use std::fmt;

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    pub fn report_extra_impl_obligation(
        &self,
        error_span: Span,
        item_name: ast::Name,
        _impl_item_def_id: DefId,
        trait_item_def_id: DefId,
        requirement: &dyn fmt::Display,
    ) -> DiagnosticBuilder<'tcx> {
        let msg = "impl has stricter requirements than trait";
        let sp = self.tcx.sess.source_map().def_span(error_span);

        let mut err = struct_span_err!(self.tcx.sess, sp, E0276, "{}", msg);

        if let Some(trait_item_span) = self.tcx.hir().span_if_local(trait_item_def_id) {
            let span = self.tcx.sess.source_map().def_span(trait_item_span);
            err.span_label(span, format!("definition of `{}` from trait", item_name));
        }

        err.span_label(sp, format!("impl has extra requirement {}", requirement));

        err
    }
}

pub fn report_object_safety_error(
    tcx: TyCtxt<'tcx>,
    span: Span,
    trait_def_id: DefId,
    violations: Vec<ObjectSafetyViolation>,
) -> DiagnosticBuilder<'tcx> {
    let trait_str = tcx.def_path_str(trait_def_id);
    let trait_span = tcx.hir().get_if_local(trait_def_id).and_then(|node| match node {
        hir::Node::Item(item) => Some(item.ident.span),
        _ => None,
    });
    let span = tcx.sess.source_map().def_span(span);
    let mut err = struct_span_err!(
        tcx.sess,
        span,
        E0038,
        "the trait `{}` cannot be made into an object",
        trait_str
    );
    err.span_label(span, format!("the trait `{}` cannot be made into an object", trait_str));

    let mut reported_violations = FxHashSet::default();
    let mut had_span_label = false;
    for violation in violations {
        if let ObjectSafetyViolation::SizedSelf(sp) = &violation {
            if !sp.is_empty() {
                // Do not report `SizedSelf` without spans pointing at `SizedSelf` obligations
                // with a `Span`.
                reported_violations.insert(ObjectSafetyViolation::SizedSelf(vec![].into()));
            }
        }
        if reported_violations.insert(violation.clone()) {
            let spans = violation.spans();
            let msg = if trait_span.is_none() || spans.is_empty() {
                format!("the trait cannot be made into an object because {}", violation.error_msg())
            } else {
                had_span_label = true;
                format!("...because {}", violation.error_msg())
            };
            if spans.is_empty() {
                err.note(&msg);
            } else {
                for span in spans {
                    err.span_label(span, &msg);
                }
            }
            match (trait_span, violation.solution()) {
                (Some(_), Some((note, None))) => {
                    err.help(&note);
                }
                (Some(_), Some((note, Some((sugg, span))))) => {
                    err.span_suggestion(span, &note, sugg, Applicability::MachineApplicable);
                }
                // Only provide the help if its a local trait, otherwise it's not actionable.
                _ => {}
            }
        }
    }
    if let (Some(trait_span), true) = (trait_span, had_span_label) {
        err.span_label(trait_span, "this trait cannot be made into an object...");
    }

    if tcx.sess.trait_methods_not_found.borrow().contains(&span) {
        // Avoid emitting error caused by non-existing method (#58734)
        err.cancel();
    }

    err
}
