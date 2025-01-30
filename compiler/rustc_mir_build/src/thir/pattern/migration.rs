//! Automatic migration of Rust 2021 patterns to a form valid in both Editions 2021 and 2024.

use rustc_ast::BindingMode;
use rustc_errors::MultiSpan;
use rustc_hir::{self as hir, ByRef, HirId, Mutability};
use rustc_lint as lint;
use rustc_middle::span_bug;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeckResults};
use rustc_span::{Ident, Span};

use crate::errors::*;
use crate::fluent_generated as fluent;

/// For patterns flagged for migration during HIR typeck, this handles constructing and emitting
/// a diagnostic suggestion.
pub(super) struct PatMigration<'a> {
    /// `&` and `&mut` patterns we may need to suggest removing.
    explicit_derefs: Vec<Span>,
    /// Variable binding modes we may need to suggest making implicit.
    explicit_modes: Vec<Span>,
    /// Implicit dereferences we may need to suggest adding `&` or `&mut` patterns for, together
    /// with the HIR id of the pattern where they occur, for formatting.
    implicit_derefs: Vec<(Span, HirId)>,
    /// Implicit by-reference binding modes we may need to suggest making explicit.
    implicit_modes: Vec<(Span, Mutability)>,
    /// How many references deep is the current pattern? For determining default binding mode.
    current_ref_depth: usize,
    /// How many references deep is the binding mode able to be `ref mut`?
    current_max_ref_mut_depth: usize,
    /// Whether we can suggest making derefs and binding modes implicit rather than explicit.
    can_suggest_removing: bool,
    /// Labeled spans for subpatterns invalid in Rust 2024.
    labels: &'a [(Span, String)],
}

impl<'a> PatMigration<'a> {
    pub(super) fn new(labels: &'a Vec<(Span, String)>) -> Self {
        PatMigration {
            explicit_derefs: Vec::new(),
            explicit_modes: Vec::new(),
            implicit_derefs: Vec::new(),
            implicit_modes: Vec::new(),
            current_ref_depth: 0,
            current_max_ref_mut_depth: 0,
            can_suggest_removing: true,
            labels: labels.as_slice(),
        }
    }

    /// On Rust 2024, this emits a hard error. On earlier Editions, this emits the
    /// future-incompatibility lint `rust_2024_incompatible_pat`.
    pub(super) fn emit<'tcx>(
        self,
        tcx: TyCtxt<'tcx>,
        typeck_results: &'a TypeckResults<'tcx>,
        pat_id: HirId,
    ) {
        let mut spans = MultiSpan::from_spans(self.labels.iter().map(|(span, _)| *span).collect());
        for (span, label) in self.labels {
            spans.push_span_label(*span, label.clone());
        }
        let sugg = self.build_suggestion(typeck_results);
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

    fn build_suggestion(
        &self,
        typeck_results: &'a TypeckResults<'_>,
    ) -> Rust2024IncompatiblePatSugg {
        if self.can_suggest_removing {
            // We can suggest a simple pattern by removing all explicit derefs and binding modes.
            let suggestion = self
                .explicit_modes
                .iter()
                .chain(&self.explicit_derefs)
                .map(|&removed_sp| (removed_sp, String::new()))
                .collect();
            Rust2024IncompatiblePatSugg {
                suggestion,
                kind: Rust2024IncompatiblePatSuggKind::Subtractive,
                ref_pattern_count: self.explicit_derefs.len(),
                binding_mode_count: self.explicit_modes.len(),
            }
        } else {
            // We can't suggest a simple pattern, so fully elaborate the pattern's match ergonomics.
            let modes = self.implicit_modes.iter().map(|&(sp, mutbl)| {
                let sugg_str = match mutbl {
                    Mutability::Not => "ref ",
                    Mutability::Mut => "ref mut ",
                };
                (sp, sugg_str.to_owned())
            });
            let mut ref_pattern_count = 0;
            let derefs = self.implicit_derefs.iter().map(|&(sp, hir_id)| {
                let adjustments = typeck_results.pat_adjustments().get(hir_id).unwrap();
                let ref_pat_str = adjustments
                    .iter()
                    .map(|ref_ty| {
                        let &ty::Ref(_, _, mutbl) = ref_ty.kind() else {
                            span_bug!(sp, "pattern implicitly dereferences a non-ref type");
                        };

                        mutbl.ref_prefix_str()
                    })
                    .collect();
                ref_pattern_count += adjustments.len();
                (sp.shrink_to_lo(), ref_pat_str)
            });
            Rust2024IncompatiblePatSugg {
                suggestion: modes.chain(derefs).collect(),
                kind: Rust2024IncompatiblePatSuggKind::Additive,
                ref_pattern_count,
                binding_mode_count: self.implicit_modes.len(),
            }
        }
    }

    /// The default binding mode at the current pattern, if all reference patterns were removed.
    fn default_mode(&self) -> ByRef {
        if self.current_ref_depth == 0 {
            ByRef::No
        } else {
            // If all `&` and `&mut` patterns are removed, the default binding mode's reference
            // mutability is mutable if and only if there are only `&mut` reference types.
            // See `FnCtxt::peel_off_references` in `rustc_hir_typeck::pat` for more information.
            let mutbl = if self.current_max_ref_mut_depth == self.current_ref_depth {
                Mutability::Mut
            } else {
                Mutability::Not
            };
            ByRef::Yes(mutbl)
        }
    }

    /// Tracks when we're lowering a `&` or `&mut` pattern and adjusts the suggestion if necessary.
    /// This should be followed by a call to [`PatMigration::leave_ref`] when we leave the pattern.
    pub(super) fn visit_explicit_deref<'tcx>(
        &mut self,
        pat_span: Span,
        mutbl: Mutability,
        subpat: &'tcx hir::Pat<'tcx>,
    ) {
        self.explicit_derefs.push(pat_span.with_hi(subpat.span.lo()));

        // If the immediate subpattern is a binding, removing this reference pattern would change
        // its type, so we opt not to remove any, for simplicity.
        // FIXME(ref_pat_eat_one_layer_2024): This assumes ref pats can't eat the binding mode
        // alone. Depending on the pattern typing rules in use, we can be more precise here.
        if matches!(subpat.kind, hir::PatKind::Binding(_, _, _, _)) {
            self.can_suggest_removing = false;
        }

        // Keep track of the reference depth for determining the default binding mode.
        if self.current_max_ref_mut_depth == self.current_ref_depth && mutbl.is_mut() {
            self.current_max_ref_mut_depth += 1;
        }
        self.current_ref_depth += 1;
    }

    /// Tracks when we're lowering a pattern that implicitly dereferences the scrutinee.
    /// This should be followed by a call to [`PatMigration::leave_ref`] when we leave the pattern.
    pub(super) fn visit_implicit_derefs<'tcx>(
        &mut self,
        pat: &'tcx hir::Pat<'tcx>,
        adjustments: &[Ty<'tcx>],
    ) {
        self.implicit_derefs.push((pat.span, pat.hir_id));

        // Keep track of the reference depth for determining the default binding mode.
        if self.current_max_ref_mut_depth == self.current_ref_depth
            && adjustments.iter().all(|ref_ty| {
                let &ty::Ref(_, _, mutbl) = ref_ty.kind() else {
                    span_bug!(pat.span, "pattern implicitly dereferences a non-ref type");
                };
                mutbl.is_mut()
            })
        {
            self.current_max_ref_mut_depth += 1;
        }
        self.current_ref_depth += 1;
    }

    /// Tracks when we leave a reference (either implicitly or explicitly derefed) while lowering.
    /// This should follow a call to [`PatMigration::visit_explicit_deref`] or
    /// [`PatMigration::visit_implicit_derefs`].
    pub(super) fn leave_ref(&mut self) {
        if self.current_max_ref_mut_depth == self.current_ref_depth {
            self.current_max_ref_mut_depth -= 1;
        }
        self.current_ref_depth -= 1;
    }

    /// Keeps track of bindings and adjusts the suggestion if necessary.
    pub(super) fn visit_binding(
        &mut self,
        pat_span: Span,
        mode: BindingMode,
        explicit_ba: BindingMode,
        ident: Ident,
    ) {
        // Note any binding modes we may need to make explicit.
        if explicit_ba.0 == ByRef::No
            && let ByRef::Yes(mutbl) = mode.0
        {
            self.implicit_modes.push((ident.span.shrink_to_lo(), mutbl));
        }

        if self.can_suggest_removing {
            if mode == BindingMode(self.default_mode(), Mutability::Not) {
                // Note any binding modes we may need to make implicit.
                if matches!(explicit_ba.0, ByRef::Yes(_)) {
                    self.explicit_modes.push(pat_span.with_hi(ident.span.lo()))
                }
            } else {
                // If removing reference patterns would change the mode of this binding, we opt not
                // to remove any, for simplicity.
                self.can_suggest_removing = false;
            }
        }
    }
}
