//! Automatic migration of Rust 2021 patterns to a form valid in both Editions 2021 and 2024.

use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::MultiSpan;
use rustc_hir::{BindingMode, ByRef, HirId, Mutability};
use rustc_lint as lint;
use rustc_middle::ty::{self, Rust2024IncompatiblePatInfo, TyCtxt};
use rustc_span::{Ident, Span};

use crate::errors::{Rust2024IncompatiblePat, Rust2024IncompatiblePatSugg};
use crate::fluent_generated as fluent;

/// For patterns flagged for migration during HIR typeck, this handles constructing and emitting
/// a diagnostic suggestion.
pub(super) struct PatMigration<'a> {
    suggestion: Vec<(Span, String)>,
    ref_pattern_count: usize,
    binding_mode_count: usize,
    /// Internal state: the ref-mutability of the default binding mode at the subpattern being
    /// lowered, with the span where it was introduced. `None` for a by-value default mode.
    default_mode_span: Option<(Span, ty::Mutability)>,
    /// Labels for where incompatibility-causing by-ref default binding modes were introduced.
    // FIXME(ref_pat_eat_one_layer_2024_structural): To track the default binding mode, we duplicate
    // logic from HIR typeck (in order to avoid needing to store all changes to the dbm in
    // TypeckResults). Since the default binding mode acts differently under this feature gate, the
    // labels will be wrong.
    default_mode_labels: FxIndexMap<Span, Mutability>,
    /// Information collected from typeck, including spans for subpatterns invalid in Rust 2024.
    info: &'a Rust2024IncompatiblePatInfo,
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

    /// Tracks when we're lowering a pattern that implicitly dereferences the scrutinee.
    /// This should only be called when the pattern type adjustments list `adjustments` contains an
    /// implicit deref of a reference type. Returns the prior default binding mode; this should be
    /// followed by a call to [`PatMigration::leave_ref`] to restore it when we leave the pattern.
    pub(super) fn visit_implicit_derefs<'tcx>(
        &mut self,
        pat_span: Span,
        adjustments: &[ty::adjustment::PatAdjustment<'tcx>],
    ) -> Option<(Span, Mutability)> {
        // Implicitly dereferencing references changes the default binding mode, but implicit derefs
        // of smart pointers do not. Thus, we only consider implicit derefs of reference types.
        let implicit_deref_mutbls = adjustments.iter().filter_map(|adjust| {
            if let &ty::Ref(_, _, mutbl) = adjust.source.kind() { Some(mutbl) } else { None }
        });

        if !self.info.suggest_eliding_modes {
            // If we can't fix the pattern by eliding modifiers, we'll need to make the pattern
            // fully explicit. i.e. we'll need to suggest reference patterns for this.
            let suggestion_str: String =
                implicit_deref_mutbls.clone().map(|mutbl| mutbl.ref_prefix_str()).collect();
            self.suggestion.push((pat_span.shrink_to_lo(), suggestion_str));
            self.ref_pattern_count += adjustments.len();
        }

        // Remember if this changed the default binding mode, in case we want to label it.
        let min_mutbl = implicit_deref_mutbls.min().unwrap();
        if self.default_mode_span.is_none_or(|(_, old_mutbl)| min_mutbl < old_mutbl) {
            // This changes the default binding mode to `ref` or `ref mut`. Return the old mode so
            // it can be reinstated when we leave the pattern.
            self.default_mode_span.replace((pat_span, min_mutbl))
        } else {
            // This does not change the default binding mode; it was already `ref` or `ref mut`.
            self.default_mode_span
        }
    }

    /// Tracks the default binding mode when we're lowering a `&` or `&mut` pattern.
    /// Returns the prior default binding mode; this should be followed by a call to
    /// [`PatMigration::leave_ref`] to restore it when we leave the pattern.
    pub(super) fn visit_explicit_deref(&mut self) -> Option<(Span, Mutability)> {
        if let Some((default_mode_span, default_ref_mutbl)) = self.default_mode_span {
            // If this eats a by-ref default binding mode, label the binding mode.
            self.default_mode_labels.insert(default_mode_span, default_ref_mutbl);
        }
        // Set the default binding mode to by-value and return the old default binding mode so it
        // can be reinstated when we leave the pattern.
        self.default_mode_span.take()
    }

    /// Restores the default binding mode after lowering a pattern that could change it.
    /// This should follow a call to either [`PatMigration::visit_explicit_deref`] or
    /// [`PatMigration::visit_implicit_derefs`].
    pub(super) fn leave_ref(&mut self, old_mode_span: Option<(Span, Mutability)>) {
        self.default_mode_span = old_mode_span
    }

    /// Determines if a binding is relevant to the diagnostic and adjusts the notes/suggestion if
    /// so. Bindings are relevant if they have a modifier under a by-ref default mode (invalid in
    /// Rust 2024) or if we need to suggest a binding modifier for them.
    pub(super) fn visit_binding(
        &mut self,
        pat_span: Span,
        mode: BindingMode,
        explicit_ba: BindingMode,
        ident: Ident,
    ) {
        if explicit_ba != BindingMode::NONE
            && let Some((default_mode_span, default_ref_mutbl)) = self.default_mode_span
        {
            // If this overrides a by-ref default binding mode, label the binding mode.
            self.default_mode_labels.insert(default_mode_span, default_ref_mutbl);
            // If our suggestion is to elide redundnt modes, this will be one of them.
            if self.info.suggest_eliding_modes {
                self.suggestion.push((pat_span.with_hi(ident.span.lo()), String::new()));
                self.binding_mode_count += 1;
            }
        }
        if !self.info.suggest_eliding_modes
            && explicit_ba.0 == ByRef::No
            && let ByRef::Yes(mutbl) = mode.0
        {
            // If we can't fix the pattern by eliding modifiers, we'll need to make the pattern
            // fully explicit. i.e. we'll need to suggest reference patterns for this.
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
