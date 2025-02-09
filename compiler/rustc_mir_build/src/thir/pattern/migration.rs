//! Automatic migration of Rust 2021 patterns to a form valid in both Editions 2021 and 2024.

use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::MultiSpan;
use rustc_hir::HirId;
use rustc_lint as lint;
use rustc_middle::ty::{self, Rust2024IncompatiblePatInfo, TyCtxt};
use rustc_span::Span;

use crate::errors::{Rust2024IncompatiblePat, Rust2024IncompatiblePatSugg};
use crate::fluent_generated as fluent;

/// For patterns flagged for migration during HIR typeck, this handles constructing and emitting
/// a diagnostic suggestion.
pub(super) struct PatMigration<'a> {
    pub(super) suggestion: Vec<(Span, String)>,
    pub(super) ref_pattern_count: usize,
    pub(super) binding_mode_count: usize,
    /// Internal state: the ref-mutability of the default binding mode at the subpattern being
    /// lowered, with the span where it was introduced. `None` for a by-value default mode.
    pub(super) default_mode_span: Option<(Span, ty::Mutability)>,
    /// Labels for where incompatibility-causing by-ref default binding modes were introduced.
    // FIXME(ref_pat_eat_one_layer_2024_structural): To track the default binding mode, we duplicate
    // logic from HIR typeck (in order to avoid needing to store all changes to the dbm in
    // TypeckResults). Since the default binding mode acts differently under this feature gate, the
    // labels will be wrong.
    pub(super) default_mode_labels: FxIndexMap<Span, Mutability>,
    /// Information collected from typeck, including spans for subpatterns invalid in Rust 2024.
    pub(super) info: &'a Rust2024IncompatiblePatInfo,
}

impl<'a> PatMigration<'a> {
    pub(super) fn new(info: &'a Rust2024IncompatiblePatInfo) -> Self {
        PatMigration {
            suggestion: Vec::new(),
            ref_pattern_count: 0,
            binding_mode_count: 0,
            default_mode_span: None,
            default_mode_labels: Default::default(),
            info,
        }
    }

    /// On Rust 2024, this emits a hard error. On earlier Editions, this emits the
    /// future-incompatibility lint `rust_2024_incompatible_pat`.
    pub(super) fn emit<'tcx>(self, tcx: TyCtxt<'tcx>, pat_id: HirId) {
        let mut spans =
            MultiSpan::from_spans(self.info.primary_labels.iter().map(|(span, _)| *span).collect());
        for (span, label) in self.info.primary_labels.iter() {
            spans.push_span_label(*span, label.clone());
        }
        let sugg = Rust2024IncompatiblePatSugg {
            suggest_eliding_modes: self.info.suggest_eliding_modes,
            suggestion: self.suggestion,
            ref_pattern_count: self.ref_pattern_count,
            binding_mode_count: self.binding_mode_count,
            default_mode_labels: self.default_mode_labels,
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
            err.arg("bad_modifiers", self.info.bad_modifiers);
            err.arg("bad_ref_pats", self.info.bad_ref_pats);
            err.arg("is_hard_error", true);
            err.subdiagnostic(sugg);
            err.emit();
        } else {
            tcx.emit_node_span_lint(
                lint::builtin::RUST_2024_INCOMPATIBLE_PAT,
                pat_id,
                spans,
                Rust2024IncompatiblePat {
                    sugg,
                    bad_modifiers: self.info.bad_modifiers,
                    bad_ref_pats: self.info.bad_ref_pats,
                    is_hard_error,
                },
            );
        }
    }
}
