//! Automatic migration of Rust 2021 patterns to a form valid in both Editions 2021 and 2024.

use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::MultiSpan;
use rustc_hir::{self as hir, BindingMode, ByRef, HirId, Mutability};
use rustc_index::IndexVec;
use rustc_lint as lint;
use rustc_middle::span_bug;
use rustc_middle::ty::{self, Rust2024IncompatiblePatInfo, Ty, TyCtxt};
use rustc_span::source_map::SourceMap;
use rustc_span::{Ident, Span};

use crate::errors::{
    Rust2024IncompatiblePat, Rust2024IncompatiblePatSugg, Rust2024IncompatiblePatSuggKind,
};
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
    /// The next binding in the innermost enclosing deref's list of bindings.
    next_sibling: Option<PatBindingIdx>,
}

struct PatDeref<'a, 'tcx> {
    /// The span of the pattern where this deref occurs (implicitly or explicitly).
    span: Span,
    /// The mutability of the ref pattern (or for implicit derefs, of the reference type).
    // FIXME(ref_pattern_eat_one_layer_2024): Under RFC 3627's Rule 5, a `&` pattern can match a
    // `&mut` type or `ref mut` binding mode. Thus, an omitted `&` could result in a `ref mut`
    // default binding mode. We may want to track both the pattern and ref type's mutabilities.
    mutbl: Mutability,
    /// Whether this span is for a potentially-removable explicitly-provided deref, or an implicit
    /// dereference which we can potentially suggest making explicit.
    kind: PatDerefKind<'a, 'tcx>,
    /// Whether to include this as a `&` or `&mut` in the suggested pattern.
    suggest: bool,
    /// The default binding mode for variables under this deref in the user's pattern.
    real_default_mode: ByRef,
    /// The default binding mode for variable under this deref in our suggestion.
    sugg_default_mode: ByRef,
    /// The span that introduced the current default binding mode, or `None` for the top-level pat.
    default_mode_origin: Option<Span>,
    /// Whether this is an instance of `&ref x` which we may be able to simplify to `x`.
    /// Stores the HIR id of the binding pattern `ref x`, to identify it later.
    simplify_deref_ref: Option<HirId>,
    /// The next deref above this. Since we can't suggest using `&` or `&mut` on a by-ref default
    /// binding mode, a suggested deref's ancestors must also all be suggested.
    // FIXME(ref_pat_eat_one_layer_2024): By suggesting `&` and `&mut` patterns that can eat the
    // default binding mode, we'll be able to make more local suggestions. That may make this forest
    // structure unnecessary.
    parent: Option<PatDerefIdx>,
    /// The head of the linked list of child derefs directly under this. When we suggest a `&`
    /// pattern, any implicit `&mut` children will go from producing a `ref` default binding mode
    /// to `ref mut`, so we check recursively in that case to see if any bindings would change.
    // FIXME(ref_pat_eat_one_layer_2024_structural): Aside from this maybe being unnecessary if we
    // can make more local suggestions (see the above fixme), RFC 3627's Rule 3 should also obsolete
    // this (see the comments on `propagate_default_mode_change`).
    first_child: Option<PatDerefIdx>,
    /// The next child in their parents' linked list of children.
    next_sibling: Option<PatDerefIdx>,
    /// The head of the linked list of bindings directly under this deref. If we suggest this
    /// deref, we'll also need to suggest binding modifiers for any by-ref bindings.
    first_binding: Option<PatBindingIdx>,
}

enum PatDerefKind<'a, 'tcx> {
    /// For dereferences from lowering `&` and `&mut` patterns
    Explicit { inner_span: Span },
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
        let sugg = self.build_suggestion(tcx.sess.source_map());
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

    fn build_suggestion<'m>(&'m self, source_map: &SourceMap) -> Rust2024IncompatiblePatSugg<'m> {
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

        let mut removed_ref_pats = 0;
        let mut added_ref_pats = 0;
        let derefs = self.derefs.iter().filter_map(|deref| match deref.kind {
            PatDerefKind::Explicit { inner_span } if !deref.suggest => {
                // This is a ref pattern in the source but not the suggestion; suggest removing it.
                removed_ref_pats += 1;
                // Avoid eating the '(' in `&(...)`
                let span = source_map.span_until_char(deref.span.with_hi(inner_span.lo()), '(');
                // But *do* eat the ' ' in `&mut [...]`
                Some((source_map.span_extend_while_whitespace(span), String::new()))
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
        let (kind, binding_mode_count, ref_pattern_count) =
            if added_modifiers == 0 && added_ref_pats == 0 {
                let kind = Rust2024IncompatiblePatSuggKind::Subtractive;
                (Some(kind), removed_modifiers, removed_ref_pats)
            } else if removed_modifiers == 0 && removed_ref_pats == 0 {
                (Some(Rust2024IncompatiblePatSuggKind::Additive), added_modifiers, added_ref_pats)
            } else {
                (None, 0, 0)
            };
        Rust2024IncompatiblePatSugg {
            suggestion,
            kind,
            binding_mode_count,
            ref_pattern_count,
            default_mode_labels: &self.default_mode_labels,
        }
    }

    /// The default binding mode at the current point in the pattern the user wrote.
    fn real_default_mode(&self) -> ByRef {
        if let Some(current_ix) = self.innermost_deref {
            self.derefs[current_ix].real_default_mode
        } else {
            ByRef::No
        }
    }

    /// The default binding mode at the current point in the pattern we're suggesting.
    fn sugg_default_mode(&self) -> ByRef {
        if let Some(deref_ix) = self.innermost_deref {
            self.derefs[deref_ix].sugg_default_mode
        } else {
            ByRef::No
        }
    }

    /// Tracks when we're lowering a pattern that implicitly dereferences the scrutinee.
    /// This should only be called when the pattern type adjustments list `ref_tys` is non-empty.
    /// This should be followed by a call to [`PatMigration::leave_ref`] when we leave the pattern.
    pub(super) fn visit_implicit_derefs(&mut self, pat: &hir::Pat<'_>, ref_tys: &'a [Ty<'tcx>]) {
        // The effective mutability of this (as far as the default binding mode goes) is `ref` if
        // any of `ref_tys` are shared, and `ref mut` if they're all mutable.
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
    pub(super) fn visit_explicit_deref(
        &mut self,
        pat_span: Span,
        mutbl: Mutability,
        subpat: &hir::Pat<'_>,
    ) {
        // If this eats a by-ref default binding mode, label the binding mode.
        self.add_default_mode_label_if_needed();
        // This sets the default binding mode to by-value in the user's pattern, but we'll try to
        // suggest removing it.
        let my_ix =
            self.push_deref(pat_span, mutbl, PatDerefKind::Explicit { inner_span: subpat.span });

        // If this is inside a macro expansion, we won't be able to remove it.
        if pat_span.from_expansion() {
            self.add_derefs_to_suggestion(self.innermost_deref);
            return;
        }

        // If the subpattern is a binding, removing this reference pattern would change its type.
        // FIXME(ref_pat_eat_one_layer_2024): This assumes ref pats can't eat the binding mode
        // alone. Depending on the pattern typing rules in use, we can be more precise here.
        if let hir::PatKind::Binding(explicit_ba, _, _, _) = subpat.kind {
            if explicit_ba == BindingMode(ByRef::Yes(mutbl), Mutability::Not) {
                // If the binding has a `ref` modifier, we can elide both this `&` and the `ref`;
                // i.e. we can simplify `&ref x` to `x`, as long as all parent derefs are explicit.
                // NB: We don't rewrite `&ref x @ ...` to `x @ &...`, so we may end up needing to
                // reinstate this `&` later if the binding's subpattern requires it.
                // FIXME(ref_pat_eat_one_layer_2024): With RFC 3627's Rule 5, `&` patterns can match
                // `&mut` types; we'll have to check the mutability of the type rather than the
                // pattern to see whether we can elide it.
                self.derefs[my_ix].simplify_deref_ref = Some(subpat.hir_id);
                self.add_derefs_to_suggestion(self.derefs[my_ix].parent);
            } else {
                // Otherwise, we need to suggest including this `&` as well.
                self.add_derefs_to_suggestion(self.innermost_deref);
            }
        }
    }

    /// Adds a deref to our deref-forest, so that we can track the default binding mode and
    /// propagate binding mode changes when we suggest adding patterns.
    /// See [`PatMigration::propagate_default_mode_change`].
    fn push_deref(
        &mut self,
        span: Span,
        mutbl: Mutability,
        kind: PatDerefKind<'a, 'tcx>,
    ) -> PatDerefIdx {
        let parent = self.innermost_deref;
        // Get the new default binding mode in the pattern the user wrote.
        let real_default_mode = match kind {
            PatDerefKind::Implicit { .. } => match self.real_default_mode() {
                ByRef::Yes(old_mutbl) => ByRef::Yes(Ord::min(mutbl, old_mutbl)),
                ByRef::No => ByRef::Yes(mutbl),
            },
            PatDerefKind::Explicit { .. } => ByRef::No,
        };
        // If this keeps the default binding mode the same, it shares a mode origin with its
        // parent. If it changes the default binding mode, its mode origin is itself.
        let default_mode_origin = if real_default_mode == self.real_default_mode() {
            parent.and_then(|p| self.derefs[p].default_mode_origin)
        } else {
            Some(span)
        };
        // Get the default binding mode in the suggestion, assuming we don't include a reference
        // pattern for this deref. We may add one later if necessary.
        let sugg_default_mode = ByRef::Yes(match self.sugg_default_mode() {
            ByRef::Yes(parent_mutbl) => Ord::min(mutbl, parent_mutbl),
            ByRef::No => mutbl,
        });
        let my_ix = self.derefs.push(PatDeref {
            span,
            mutbl,
            kind,
            suggest: false,
            sugg_default_mode,
            real_default_mode,
            default_mode_origin,
            simplify_deref_ref: None,
            parent,
            next_sibling: parent.and_then(|p| self.derefs[p].first_child),
            first_child: None,
            first_binding: None,
        });
        if let Some(p) = parent {
            self.derefs[p].first_child = Some(my_ix);
        }
        self.innermost_deref = Some(my_ix);
        my_ix
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
        pat: &hir::Pat<'_>,
        mode: BindingMode,
        explicit_ba: BindingMode,
        ident: Ident,
    ) {
        if explicit_ba != BindingMode::NONE {
            // If this overrides a by-ref default binding mode, label the binding mode.
            self.add_default_mode_label_if_needed();
        }

        // As a special case, we may simplify `&ref x` to `x`; check our parent to see if we can.
        // The default binding mode will always be by-move in this case.
        let simplify_deref_ref = self.innermost_deref.is_some_and(|p| {
            self.derefs[p].simplify_deref_ref.is_some_and(|binding_id| pat.hir_id == binding_id)
        });

        // Otherwise, if `mode` doesn't match the default, we'll need to specify its binding
        // modifiers explicitly, which in turn necessitates a by-move default binding mode.
        // Additionally, if this is inside a macro expansion, we won't be able to change it. If a
        // binding modifier is missing inside the expansion, there's not much we can do, but we can
        // avoid suggestions to elide binding modifiers that are explicit within expansions.
        let suggest = !simplify_deref_ref
            && mode != BindingMode(self.sugg_default_mode(), Mutability::Not)
            || pat.span.from_expansion() && explicit_ba != BindingMode::NONE;

        // Track the binding
        let span = if explicit_ba == BindingMode::NONE {
            pat.span.shrink_to_lo()
        } else {
            pat.span.with_hi(ident.span.lo())
        };
        // If we're not already suggesting an explicit binding modifier for this binding, we may
        // need to later, if adding reference patterns above it changes the default binding mode.
        // In that case, track it as a child of the innermost dereference above it.
        let parent_deref = if suggest { None } else { self.innermost_deref };
        let next_sibling = parent_deref.and_then(|p| self.derefs[p].first_binding);
        let bind_ix = self.bindings.push(PatBinding { span, mode, suggest, next_sibling });
        if let Some(p) = parent_deref {
            self.derefs[p].first_binding = Some(bind_ix);
        }

        // If there was a mismatch, add `&`s to make sure we're in a by-move default binding mode.
        if suggest {
            self.add_derefs_to_suggestion(self.innermost_deref);
        }
    }

    /// Include a deref and all its ancestors in the suggestion. If this would change the mode of
    /// a binding, we include a binding modifier for it in the suggestion, which may in turn
    /// require including more explicit dereferences, etc.
    fn add_derefs_to_suggestion(&mut self, mut opt_ix: Option<PatDerefIdx>) {
        while let Some(ix) = opt_ix {
            let deref = &mut self.derefs[ix];
            if deref.suggest {
                // If this is already marked as suggested, its ancestors will be too.
                break;
            }
            deref.suggest = true;
            deref.sugg_default_mode = ByRef::No;
            deref.simplify_deref_ref = None;
            opt_ix = deref.parent;
            let propagate_downstream_ref_mut = deref.mutbl.is_not();
            self.propagate_default_mode_change(ix, propagate_downstream_ref_mut);
        }
    }

    /// If including a `&` or `&mut` pattern in our suggestion would change the binding mode of any
    /// variables, add any necessary binding modifiers and reference patterns to keep them the same.
    fn propagate_default_mode_change(&mut self, start_ix: PatDerefIdx, propagate_ref_mut: bool) {
        // After suggesting a deref, any immediate-child bindings will by default be by-value, so
        // we'll need to suggest modifiers if they should be by-ref. Likewise, if suggesting a `&`
        // changes the ref-mutability of a downstream binding under an implicit `&mut`, we'll need
        // to add a binding modifier and `&mut` patterns.
        let mut opt_bind_ix = self.derefs[start_ix].first_binding;
        while let Some(bind_ix) = opt_bind_ix {
            let binding = &mut self.bindings[bind_ix];
            opt_bind_ix = binding.next_sibling;
            // FIXME(ref_pat_eat_one_layer_2024_structural): With RFC 3627's Rule 3, an implicit
            // `&mut` under a `&` pattern won't set the default binding mode to `ref mut`, so we
            // won't need to do any mutability checks or ref-mutability propagation. We'd only call
            // this on `&`/`&mut` patterns we suggest, not their descendants, so we can assume the
            // default binding mode is by-move and that the deref is already suggested.
            if binding.mode.0 != self.derefs[start_ix].sugg_default_mode {
                binding.suggest = true;
                self.add_derefs_to_suggestion(Some(start_ix));
            }
        }

        // If we change an implicit dereference of a shared reference to a `&` pattern, any implicit
        // derefs of `&mut` references in children (until we hit another implicit `&`) will now
        // produce a `ref mut` default binding mode instead of `ref`. We'll need to recur in case
        // any downstream bindings' modes are changed.
        // FIXME(ref_pat_eat_one_layer_2024_structural): See the above fixme. This can all go.
        if propagate_ref_mut {
            let mut opt_child_ix = self.derefs[start_ix].first_child;
            while let Some(child_ix) = opt_child_ix {
                let child = &mut self.derefs[child_ix];
                opt_child_ix = child.next_sibling;
                if child.mutbl.is_mut() {
                    child.sugg_default_mode = ByRef::Yes(Mutability::Mut);
                    self.propagate_default_mode_change(child_ix, true);
                }
            }
        }
    }
}
