//! Automatic migration of Rust 2021 patterns to a form valid in both Editions 2021 and 2024.

use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::MultiSpan;
use rustc_hir::{self as hir, BindingMode, ByRef, HirId, Mutability};
use rustc_index::IndexVec;
use rustc_lint as lint;
use rustc_middle::span_bug;
use rustc_middle::ty::{self, Rust2024IncompatiblePatInfo, Ty, TyCtxt};
use rustc_span::{Ident, Span};

use crate::errors::{Rust2024IncompatiblePat, Rust2024IncompatiblePatSugg};
use crate::fluent_generated as fluent;

/// For patterns flagged for migration during HIR typeck, this handles constructing and emitting
/// a diagnostic suggestion.
pub(super) struct PatMigration<'a, 'tcx> {
    /// All the variable bindings encountered in lowering the pattern, along with whether to
    /// suggest adding/removing them.
    bindings: IndexVec<PatBindingIdx, PatBinding>,
    /// All the dereferences encountered in lowering the pattern, along with how their corresponding
    /// patterns affect the default binding mode, and whether to suggest adding/removing them.
    derefs: IndexVec<PatDerefIdx, PatDeref<'a, 'tcx>>,
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
    struct PatBindingIdx {}
}

rustc_index::newtype_index! {
    struct PatDerefIdx {}
}

struct PatBinding {
    /// The span of the binding modifier (empty if no explicit modifier was provided).
    span: Span,
    /// The actual binding mode of this binding.
    mode: BindingMode,
    /// Whether to include a binding modifier (e.g. `ref` or `mut`) in the suggested pattern.
    suggest: bool,
}

struct PatDeref<'a, 'tcx> {
    /// The span of the pattern where this deref occurs (implicitly or explicitly).
    span: Span,
    /// Whether this span is for a potentially-removable explicitly-provided deref, or an implicit
    /// dereference which we can potentially suggest making explicit.
    kind: PatDerefKind<'a, 'tcx>,
    /// Whether to include this as a `&` or `&mut` in the suggested pattern.
    suggest: bool,
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

enum PatDerefKind<'a, 'tcx> {
    /// For dereferences from lowering `&` and `&mut` patterns
    Explicit,
    /// For dereferences inserted by match ergonomics
    Implicit { ref_tys: &'a [Ty<'tcx>] },
}

/// Assuming the input is a slice of reference types implicitly dereferenced by match ergonomics
/// (stored in [`ty::TypeckResults::pat_adjustments`]), iterate over their reference mutabilities.
/// A span is provided for debugging purposes.
fn iter_ref_mutbls<'a, 'tcx>(
    span: Span,
    ref_tys: &'a [Ty<'tcx>],
) -> impl Iterator<Item = Mutability> + use<'a, 'tcx> {
    ref_tys.iter().map(move |ref_ty| {
        let &ty::Ref(_, _, mutbl) = ref_ty.kind() else {
            span_bug!(span, "pattern implicitly dereferences a non-ref type");
        };
        mutbl
    })
}

impl<'a, 'tcx> PatMigration<'a, 'tcx> {
    pub(super) fn new(info: &'a Rust2024IncompatiblePatInfo) -> Self {
        PatMigration {
            bindings: IndexVec::new(),
            derefs: IndexVec::new(),
            innermost_deref: None,
            default_mode_labels: Default::default(),
            info,
        }
    }

    /// On Rust 2024, this emits a hard error. On earlier Editions, this emits the
    /// future-incompatibility lint `rust_2024_incompatible_pat`.
    pub(super) fn emit(self, tcx: TyCtxt<'tcx>, pat_id: HirId) {
        let mut spans =
            MultiSpan::from_spans(self.info.primary_labels.iter().map(|(span, _)| *span).collect());
        for (span, label) in self.info.primary_labels.iter() {
            spans.push_span_label(*span, label.clone());
        }
        let sugg = self.build_suggestion();
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

    fn build_suggestion<'m>(&'m self) -> Rust2024IncompatiblePatSugg<'m> {
        let mut removed_modifiers = 0;
        let mut added_modifiers = 0;
        let modes = self.bindings.iter().filter_map(|binding| {
            if binding.mode == BindingMode::NONE {
                // This binding mode is written as the empty string; no need to suggest.
                None
            } else {
                if !binding.suggest && !binding.span.is_empty() {
                    // This binding is in the source but not the suggestion; suggest removing it.
                    removed_modifiers += 1;
                    Some((binding.span, String::new()))
                } else if binding.suggest && binding.span.is_empty() {
                    // This binding is in the suggestion but not the source; suggest adding it.
                    added_modifiers += 1;
                    Some((binding.span, binding.mode.prefix_str().to_owned()))
                } else {
                    // This binding is as it should be.
                    None
                }
            }
        });

        let mut added_ref_pats = 0;
        let derefs = self.derefs.iter().filter_map(|deref| match deref.kind {
            PatDerefKind::Explicit if !deref.suggest => {
                // This is a ref pattern in the source but not the suggestion; suggest removing it.
                // TODO: we don't yet suggest removing reference patterns
                todo!();
            }
            PatDerefKind::Implicit { ref_tys } if deref.suggest => {
                // This is a ref pattern in the suggestion but not the source; suggest adding it.
                let ref_pat_str =
                    iter_ref_mutbls(deref.span, ref_tys).map(Mutability::ref_prefix_str).collect();
                added_ref_pats += ref_tys.len();
                Some((deref.span.shrink_to_lo(), ref_pat_str))
            }
            _ => None,
        });

        let suggestion = modes.chain(derefs).collect();
        let binding_mode_count = if added_modifiers == 0 && added_ref_pats == 0 {
            removed_modifiers
        } else {
            added_modifiers
        };
        Rust2024IncompatiblePatSugg {
            suggest_eliding_modes: self.info.suggest_eliding_modes,
            suggestion,
            binding_mode_count,
            ref_pattern_count: added_ref_pats,
            default_mode_labels: &self.default_mode_labels,
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
    /// This should only be called when the pattern type adjustments list `ref_tys` is non-empty.
    /// This should be followed by a call to [`PatMigration::leave_ref`] when we leave the pattern.
    pub(super) fn visit_implicit_derefs(&mut self, pat: &hir::Pat<'_>, ref_tys: &'a [Ty<'tcx>]) {
        let mutbl = iter_ref_mutbls(pat.span, ref_tys)
            .min()
            .expect("`ref_tys` should have at least one element");
        self.push_deref(pat.span, mutbl, PatDerefKind::Implicit { ref_tys });
    }

    /// Tracks the default binding mode when we're lowering a `&` or `&mut` pattern.
    /// This should be followed by a call to [`PatMigration::leave_ref`] when we leave the pattern.
    // FIXME(ref_pat_eat_one_layer_2024): This assumes reference patterns correspond to real
    // dereferences. If reference patterns can match the default binding mode alone, we may need to
    // check `TypeckResults::skipped_ref_pats` to tell if this pattern corresponds to an implicit
    // dereference we've already visited.
    pub(super) fn visit_explicit_deref(&mut self, pat_span: Span, mutbl: Mutability) {
        // If this eats a by-ref default binding mode, label the binding mode.
        self.add_default_mode_label_if_needed();
        // Set the default binding mode to by-value.
        self.push_deref(pat_span, mutbl, PatDerefKind::Explicit);
    }

    /// Adds a deref to our deref-forest, so that we can track the default binding mode.
    // TODO: this is also for propagating binding mode changes when we suggest adding patterns
    fn push_deref(&mut self, span: Span, mutbl: Mutability, kind: PatDerefKind<'a, 'tcx>) {
        let parent = self.innermost_deref;
        // Get the new default binding mode in the pattern the user wrote.
        let real_default_mode = match kind {
            PatDerefKind::Implicit { .. } => match self.real_default_mode() {
                ByRef::Yes(old_mutbl) => ByRef::Yes(Ord::min(mutbl, old_mutbl)),
                ByRef::No => ByRef::Yes(mutbl),
            },
            PatDerefKind::Explicit => ByRef::No,
        };
        // If this keeps the default binding mode the same, it shares a mode origin with its
        // parent. If it changes the default binding mode, its mode origin is itself.
        let default_mode_origin = if real_default_mode == self.real_default_mode() {
            parent.and_then(|p| self.derefs[p].default_mode_origin)
        } else {
            Some(span)
        };
        let my_ix = self.derefs.push(PatDeref {
            span,
            suggest: !self.info.suggest_eliding_modes
                || matches!(kind, PatDerefKind::Explicit { .. }),
            kind,
            real_default_mode,
            default_mode_origin,
            parent,
        });
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
                self.bindings.push(PatBinding {
                    span: pat_span.with_hi(ident.span.lo()),
                    mode,
                    suggest: false,
                });
            }
        }
        if !self.info.suggest_eliding_modes
            && explicit_ba.0 == ByRef::No
            && matches!(mode.0, ByRef::Yes(_))
        {
            // If we can't fix the pattern by eliding modifiers, we'll need to make the pattern
            // fully explicit. i.e. we'll need to suggest reference patterns for this.
            self.bindings.push(PatBinding { span: pat_span.shrink_to_lo(), mode, suggest: true });
        }
    }
}
