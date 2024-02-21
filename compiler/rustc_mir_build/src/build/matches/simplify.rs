//! Simplifying Candidates
//!
//! *Simplifying* a match pair `place @ pattern` means breaking it down
//! into bindings or other, simpler match pairs. For example:
//!
//! - `place @ (P1, P2)` can be simplified to `[place.0 @ P1, place.1 @ P2]`
//! - `place @ x` can be simplified to `[]` by binding `x` to `place`
//!
//! The `simplify_match_pairs` routine just repeatedly applies these
//! sort of simplifications until there is nothing left to
//! simplify. Match pairs cannot be simplified if they require some
//! sort of test: for example, testing which variant an enum is, or
//! testing a value against a constant.

use crate::build::expr::as_place::PlaceBuilder;
use crate::build::matches::{Ascription, Binding, Candidate, MatchPair};
use crate::build::Builder;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_middle::thir::{self, *};
use rustc_middle::ty;

use std::mem;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Simplify a list of match pairs so they all require a test. Stores relevant bindings and
    /// ascriptions in the provided `Vec`s.
    #[instrument(skip(self), level = "debug")]
    pub(super) fn simplify_match_pairs<'pat>(
        &mut self,
        match_pairs: &mut Vec<MatchPair<'pat, 'tcx>>,
        candidate_bindings: &mut Vec<Binding<'tcx>>,
        candidate_ascriptions: &mut Vec<Ascription<'tcx>>,
    ) {
        // In order to please the borrow checker, in a pattern like `x @ pat` we must lower the
        // bindings in `pat` before `x`. E.g. (#69971):
        //
        // struct NonCopyStruct {
        //     copy_field: u32,
        // }
        //
        // fn foo1(x: NonCopyStruct) {
        //     let y @ NonCopyStruct { copy_field: z } = x;
        //     // the above should turn into
        //     let z = x.copy_field;
        //     let y = x;
        // }
        //
        // We can't just reverse the binding order, because we must preserve pattern-order
        // otherwise, e.g. in `let (Some(a), Some(b)) = (x, y)`. Our rule then is: deepest-first,
        // and bindings at the same depth stay in source order.
        //
        // To do this, every time around the loop we prepend the newly found bindings to the
        // bindings we already had.
        //
        // example:
        // candidate.bindings = [1, 2, 3]
        // bindings in iter 1: [4, 5]
        // bindings in iter 2: [6, 7]
        //
        // final bindings: [6, 7, 4, 5, 1, 2, 3]
        let mut accumulated_bindings = mem::take(candidate_bindings);
        let mut simplified_match_pairs = Vec::new();
        // Repeatedly simplify match pairs until we're left with only unsimplifiable ones.
        loop {
            for match_pair in mem::take(match_pairs) {
                if let Err(match_pair) = self.simplify_match_pair(
                    match_pair,
                    candidate_bindings,
                    candidate_ascriptions,
                    match_pairs,
                ) {
                    simplified_match_pairs.push(match_pair);
                }
            }

            // This does: accumulated_bindings = candidate.bindings.take() ++ accumulated_bindings
            candidate_bindings.extend_from_slice(&accumulated_bindings);
            mem::swap(candidate_bindings, &mut accumulated_bindings);
            candidate_bindings.clear();

            if match_pairs.is_empty() {
                break;
            }
        }

        // Store computed bindings back in `candidate_bindings`.
        mem::swap(candidate_bindings, &mut accumulated_bindings);
        // Store simplified match pairs back in `match_pairs`.
        mem::swap(match_pairs, &mut simplified_match_pairs);

        // Move or-patterns to the end, because they can result in us
        // creating additional candidates, so we want to test them as
        // late as possible.
        match_pairs.sort_by_key(|pair| matches!(pair.pattern.kind, PatKind::Or { .. }));
        debug!(simplified = ?match_pairs, "simplify_match_pairs");
    }

    /// Create a new candidate for each pattern in `pats`, and recursively simplify tje
    /// single-or-pattern case.
    pub(super) fn create_or_subcandidates<'pat>(
        &mut self,
        place: &PlaceBuilder<'tcx>,
        pats: &'pat [Box<Pat<'tcx>>],
        has_guard: bool,
    ) -> Vec<Candidate<'pat, 'tcx>> {
        pats.iter()
            .map(|box pat| {
                let mut candidate = Candidate::new(place.clone(), pat, has_guard, self);
                if let [MatchPair { pattern: Pat { kind: PatKind::Or { pats }, .. }, place, .. }] =
                    &*candidate.match_pairs
                {
                    candidate.subcandidates =
                        self.create_or_subcandidates(place, pats, candidate.has_guard);
                    candidate.match_pairs.pop();
                }
                candidate
            })
            .collect()
    }

    /// Tries to simplify `match_pair`, returning `Ok(())` if successful. If successful, new match
    /// pairs and bindings will have been pushed into the respective `Vec`s. If no simplification is
    /// possible, `Err` is returned.
    fn simplify_match_pair<'pat>(
        &mut self,
        mut match_pair: MatchPair<'pat, 'tcx>,
        bindings: &mut Vec<Binding<'tcx>>,
        ascriptions: &mut Vec<Ascription<'tcx>>,
        match_pairs: &mut Vec<MatchPair<'pat, 'tcx>>,
    ) -> Result<(), MatchPair<'pat, 'tcx>> {
        match match_pair.pattern.kind {
            PatKind::Leaf { .. }
            | PatKind::Deref { .. }
            | PatKind::Array { .. }
            | PatKind::Never
            | PatKind::Wild
            | PatKind::Error(_) => {}

            PatKind::AscribeUserType {
                ascription: thir::Ascription { ref annotation, variance },
                ..
            } => {
                // Apply the type ascription to the value at `match_pair.place`
                if let Some(source) = match_pair.place.try_to_place(self) {
                    ascriptions.push(Ascription {
                        annotation: annotation.clone(),
                        source,
                        variance,
                    });
                }
            }

            PatKind::Binding {
                name: _,
                mutability: _,
                mode,
                var,
                ty: _,
                subpattern: _,
                is_primary: _,
            } => {
                if let Some(source) = match_pair.place.try_to_place(self) {
                    bindings.push(Binding {
                        span: match_pair.pattern.span,
                        source,
                        var_id: var,
                        binding_mode: mode,
                    });
                }
            }

            PatKind::InlineConstant { subpattern: ref pattern, def } => {
                // Apply a type ascription for the inline constant to the value at `match_pair.place`
                if let Some(source) = match_pair.place.try_to_place(self) {
                    let span = match_pair.pattern.span;
                    let parent_id = self.tcx.typeck_root_def_id(self.def_id.to_def_id());
                    let args = ty::InlineConstArgs::new(
                        self.tcx,
                        ty::InlineConstArgsParts {
                            parent_args: ty::GenericArgs::identity_for_item(self.tcx, parent_id),
                            ty: self.infcx.next_ty_var(TypeVariableOrigin {
                                kind: TypeVariableOriginKind::MiscVariable,
                                span,
                            }),
                        },
                    )
                    .args;
                    let user_ty =
                        self.infcx.canonicalize_user_type_annotation(ty::UserType::TypeOf(
                            def.to_def_id(),
                            ty::UserArgs { args, user_self_ty: None },
                        ));
                    let annotation = ty::CanonicalUserTypeAnnotation {
                        inferred_ty: pattern.ty,
                        span,
                        user_ty: Box::new(user_ty),
                    };
                    ascriptions.push(Ascription {
                        annotation,
                        source,
                        variance: ty::Contravariant,
                    });
                }
            }

            PatKind::Constant { .. } => {
                // FIXME normalize patterns when possible
                return Err(match_pair);
            }

            PatKind::Range(ref range) => {
                if range.is_full_range(self.tcx) != Some(true) {
                    return Err(match_pair);
                }
            }

            PatKind::Slice { ref prefix, ref slice, ref suffix } => {
                if !(prefix.is_empty() && slice.is_some() && suffix.is_empty()) {
                    self.simplify_match_pairs(&mut match_pair.subpairs, bindings, ascriptions);
                    return Err(match_pair);
                }
            }

            PatKind::Variant { adt_def, args, variant_index, subpatterns: _ } => {
                let irrefutable = adt_def.variants().iter_enumerated().all(|(i, v)| {
                    i == variant_index || {
                        (self.tcx.features().exhaustive_patterns
                            || self.tcx.features().min_exhaustive_patterns)
                            && !v
                                .inhabited_predicate(self.tcx, adt_def)
                                .instantiate(self.tcx, args)
                                .apply_ignore_module(self.tcx, self.param_env)
                    }
                }) && (adt_def.did().is_local()
                    || !adt_def.is_variant_list_non_exhaustive());
                if !irrefutable {
                    self.simplify_match_pairs(&mut match_pair.subpairs, bindings, ascriptions);
                    return Err(match_pair);
                }
            }

            PatKind::Or { .. } => return Err(match_pair),
        }

        // Simplifiable pattern; we replace it with its subpairs.
        match_pairs.append(&mut match_pair.subpairs);
        Ok(())
    }
}
