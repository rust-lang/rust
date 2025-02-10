//! Automatic migration of Rust 2021 patterns to a form valid in both Editions 2021 and 2024.

use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::MultiSpan;
use rustc_hir::{BindingMode, ByRef, HirId, Mutability};
use rustc_index::IndexVec;
use rustc_lint as lint;
use rustc_middle::span_bug;
use rustc_middle::ty::{self, Rust2024IncompatiblePatInfo, Ty, TyCtxt};
use rustc_span::{Ident, Span};

use crate::errors::{Rust2024IncompatiblePat, Rust2024IncompatiblePatSugg};
use crate::fluent_generated as fluent;

/// For patterns flagged for migration during HIR typeck, this handles constructing and emitting
/// a diagnostic suggestion.
pub(super) struct PatMigration<'a> {
    suggestion: Vec<(Span, String)>,
    ref_pattern_count: usize,
    binding_mode_count: usize,
    /// All the dereferences encountered in lowering the pattern, along with how their corresponding
    /// patterns affect the default binding mode.
    derefs: IndexVec<PatDerefIdx, PatDeref>,
    /// Internal state: the innermost deref above the pattern currently being lowered.
    innermost_deref: Option<PatDerefIdx>,
    /// Labels for where incompatibility-causing by-ref default binding modes were introduced.
    // FIXME(ref_pat_eat_one_layer_2024_structural): To track the default binding mode, we duplicate
    // logic from HIR typeck (in order to avoid needing to store all changes to the dbm in
    // TypeckResults). Since the default binding mode acts differently under this feature gate, the
    // labels will be wrong.
    default_mode_labels: FxIndexMap<Span, Mutability>,
    /// Information collected from typeck, including spans for subpatterns invalid in Rust 2024.
    info: &'a Rust2024IncompatiblePatInfo,
}

rustc_index::newtype_index! {
    struct PatDerefIdx {}
}

struct PatDeref {
    /// The default binding mode for variables under this deref.
    real_default_mode: ByRef,
    /// The span that introduced the current default binding mode, or `None` for the top-level pat.
    default_mode_origin: Option<Span>,
    /// The next deref above this. Since we can't suggest using `&` or `&mut` on a by-ref default
    /// binding mode, a suggested deref's ancestors must also all be suggested.
    // FIXME(ref_pat_eat_one_layer_2024): By suggesting `&` and `&mut` patterns that can eat the
    // default binding mode, we'll be able to make more local suggestions. That may make this forest
    // structure unnecessary.
    parent: Option<PatDerefIdx>,
}

impl<'a> PatMigration<'a> {
    pub(super) fn new(info: &'a Rust2024IncompatiblePatInfo) -> Self {
        PatMigration {
            suggestion: Vec::new(),
            ref_pattern_count: 0,
            binding_mode_count: 0,
            derefs: IndexVec::new(),
            innermost_deref: None,
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

    /// When lowering a reference pattern or a binding with a modifier, this checks if the default
    /// binding mode is by-ref, and if so, adds a labeled note to the diagnostic with the origin of
    /// the current default binding mode.
    fn add_default_mode_label_if_needed(&mut self) {
        if let ByRef::Yes(ref_mutbl) = self.real_default_mode() {
            // The by-ref default binding mode must have come from an implicit deref. If there was a
            // problem in tracking that for the diagnostic, try to avoid ICE on release builds.
            debug_assert!(
                self.innermost_deref
                    .is_some_and(|ix| self.derefs[ix].default_mode_origin.is_some())
            );
            if let Some(ix) = self.innermost_deref
                && let Some(span) = self.derefs[ix].default_mode_origin
            {
                self.default_mode_labels.insert(span, ref_mutbl);
            }
        }
    }

    /// The default binding mode at the current pattern.
    fn real_default_mode(&self) -> ByRef {
        if let Some(current_ix) = self.innermost_deref {
            self.derefs[current_ix].real_default_mode
        } else {
            ByRef::No
        }
    }

    /// Tracks when we're lowering a pattern that implicitly dereferences the scrutinee.
    /// This should only be called when the pattern type adjustments list `adjustments` is
    /// non-empty.
    /// This should be followed by a call to [`PatMigration::leave_ref`] when we leave the pattern.
    pub(super) fn visit_implicit_derefs<'tcx>(&mut self, pat_span: Span, adjustments: &[Ty<'tcx>]) {
        let implicit_deref_mutbls = adjustments.iter().map(|ref_ty| {
            let &ty::Ref(_, _, mutbl) = ref_ty.kind() else {
                span_bug!(pat_span, "pattern implicitly dereferences a non-ref type");
            };
            mutbl
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
        let new_real_ref_mutbl = match self.real_default_mode() {
            ByRef::Yes(Mutability::Not) => Mutability::Not,
            _ => implicit_deref_mutbls.min().unwrap(),
        };
        self.push_deref(pat_span, ByRef::Yes(new_real_ref_mutbl));
    }

    /// Tracks the default binding mode when we're lowering a `&` or `&mut` pattern.
    /// This should be followed by a call to [`PatMigration::leave_ref`] when we leave the pattern.
    // FIXME(ref_pat_eat_one_layer_2024): This assumes reference patterns correspond to real
    // dereferences. If reference patterns can match the default binding mode alone, we may need to
    // check `TypeckResults::skipped_ref_pats` to tell if this pattern corresponds to an implicit
    // dereference we've already visited.
    pub(super) fn visit_explicit_deref(&mut self, pat_span: Span) {
        // If this eats a by-ref default binding mode, label the binding mode.
        self.add_default_mode_label_if_needed();
        // Set the default binding mode to by-value.
        self.push_deref(pat_span, ByRef::No);
    }

    /// Adds a deref to our deref-forest, so that we can track the default binding mode.
    // TODO: this is also for propagating binding mode changes when we suggest adding patterns
    fn push_deref(&mut self, span: Span, real_default_mode: ByRef) {
        let parent = self.innermost_deref;
        // If this keeps the default binding mode the same, it shares a mode origin with its
        // parent. If it changes the default binding mode, its mode origin is itself.
        let default_mode_origin = if real_default_mode == self.real_default_mode() {
            parent.and_then(|p| self.derefs[p].default_mode_origin)
        } else {
            Some(span)
        };
        let my_ix = self.derefs.push(PatDeref { real_default_mode, default_mode_origin, parent });
        self.innermost_deref = Some(my_ix);
    }

    /// Restores the default binding mode after lowering a pattern that could change it.
    /// This should follow a call to either [`PatMigration::visit_explicit_deref`] or
    /// [`PatMigration::visit_implicit_derefs`].
    pub(super) fn leave_ref(&mut self) {
        debug_assert!(self.innermost_deref.is_some(), "entering/leaving refs should be paired");
        if let Some(child_ix) = self.innermost_deref {
            self.innermost_deref = self.derefs[child_ix].parent;
        }
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
        if explicit_ba != BindingMode::NONE {
            // If this overrides a by-ref default binding mode, label the binding mode.
            self.add_default_mode_label_if_needed();
            if self.info.suggest_eliding_modes && matches!(mode.0, ByRef::Yes(_)) {
                // If our suggestion is to elide redundant modes, this will be one of them.
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
