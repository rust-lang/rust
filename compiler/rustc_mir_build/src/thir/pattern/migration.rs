//! Automatic migration of Rust 2021 patterns to a form valid in both Editions 2021 and 2024.

use rustc_ast::BindingMode;
use rustc_errors::MultiSpan;
use rustc_hir::{ByRef, HirId, Mutability};
use rustc_lint as lint;
use rustc_middle::span_bug;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::{Ident, Span};

use crate::errors::{Rust2024IncompatiblePat, Rust2024IncompatiblePatSugg};
use crate::fluent_generated as fluent;

/// For patterns flagged for migration during HIR typeck, this handles constructing and emitting
/// a diagnostic suggestion.
pub(super) struct PatMigration<'a> {
    suggestion: Vec<(Span, String)>,
    ref_pattern_count: usize,
    binding_mode_count: usize,
    /// Labeled spans for subpatterns invalid in Rust 2024.
    labels: &'a [(Span, String)],
}

impl<'a> PatMigration<'a> {
    pub(super) fn new(labels: &'a Vec<(Span, String)>) -> Self {
        PatMigration {
            suggestion: Vec::new(),
            ref_pattern_count: 0,
            binding_mode_count: 0,
            labels: labels.as_slice(),
        }
    }

    /// On Rust 2024, this emits a hard error. On earlier Editions, this emits the
    /// future-incompatibility lint `rust_2024_incompatible_pat`.
    pub(super) fn emit<'tcx>(self, tcx: TyCtxt<'tcx>, pat_id: HirId) {
        let mut spans = MultiSpan::from_spans(self.labels.iter().map(|(span, _)| *span).collect());
        for (span, label) in self.labels {
            spans.push_span_label(*span, label.clone());
        }
        let sugg = Rust2024IncompatiblePatSugg {
            suggestion: self.suggestion,
            ref_pattern_count: self.ref_pattern_count,
            binding_mode_count: self.binding_mode_count,
        };
        // If a relevant span is from at least edition 2024, this is a hard error.
        let is_hard_error = spans.primary_spans().iter().any(|span| span.at_least_rust_2024());
        if is_hard_error {
            let mut err =
                tcx.dcx().struct_span_err(spans, fluent::mir_build_rust_2024_incompatible_pat);
            if let Some(info) = lint::builtin::RUST_2024_INCOMPATIBLE_PAT.future_incompatible {
                // provide the same reference link as the lint
                err.note(format!("for more information, see {}", info.reference));
            }
            err.subdiagnostic(sugg);
            err.emit();
        } else {
            tcx.emit_node_span_lint(
                lint::builtin::RUST_2024_INCOMPATIBLE_PAT,
                pat_id,
                spans,
                Rust2024IncompatiblePat { sugg },
            );
        }
    }

    pub(super) fn visit_implicit_derefs<'tcx>(&mut self, pat_span: Span, adjustments: &[Ty<'tcx>]) {
        let suggestion_str: String = adjustments
            .iter()
            .map(|ref_ty| {
                let &ty::Ref(_, _, mutbl) = ref_ty.kind() else {
                    span_bug!(pat_span, "pattern implicitly dereferences a non-ref type");
                };

                mutbl.ref_prefix_str()
            })
            .collect();
        self.suggestion.push((pat_span.shrink_to_lo(), suggestion_str));
        self.ref_pattern_count += adjustments.len();
    }

    pub(super) fn visit_binding(
        &mut self,
        pat_span: Span,
        mode: BindingMode,
        explicit_ba: BindingMode,
        ident: Ident,
    ) {
        if explicit_ba.0 == ByRef::No
            && let ByRef::Yes(mutbl) = mode.0
        {
            let sugg_str = match mutbl {
                Mutability::Not => "ref ",
                Mutability::Mut => "ref mut ",
            };
            self.suggestion
                .push((pat_span.with_lo(ident.span.lo()).shrink_to_lo(), sugg_str.to_owned()));
            self.binding_mode_count += 1;
        }
    }
}
