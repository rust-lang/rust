//! Simplifying Candidates
//!
//! *Simplifying* a match pair `place @ pattern` means breaking it down
//! into bindings or other, simpler match pairs. For example:
//!
//! - `place @ (P1, P2)` can be simplified to `[place.0 @ P1, place.1 @ P2]`
//! - `place @ x` can be simplified to `[]` by binding `x` to `place`
//!
//! The `simplify_candidate` routine just repeatedly applies these
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
    /// Simplify a candidate so that all match pairs require a test.
    ///
    /// This method will also split a candidate, in which the only
    /// match-pair is an or-pattern, into multiple candidates.
    /// This is so that
    ///
    /// match x {
    ///     0 | 1 => { ... },
    ///     2 | 3 => { ... },
    /// }
    ///
    /// only generates a single switch. If this happens this method returns
    /// `true`.
    #[instrument(skip(self, candidate), level = "debug")]
    pub(super) fn simplify_candidate<'pat>(
        &mut self,
        candidate: &mut Candidate<'pat, 'tcx>,
    ) -> bool {
        debug!("{candidate:#?}");
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
        let mut accumulated_bindings = mem::take(&mut candidate.bindings);
        // Repeatedly simplify match pairs until fixed point is reached
        loop {
            let mut changed = false;
            for match_pair in mem::take(&mut candidate.match_pairs) {
                match self.simplify_match_pair(match_pair, candidate) {
                    Ok(()) => {
                        changed = true;
                    }
                    Err(match_pair) => {
                        candidate.match_pairs.push(match_pair);
                    }
                }
            }

            // This does: accumulated_bindings = candidate.bindings.take() ++ accumulated_bindings
            candidate.bindings.extend_from_slice(&accumulated_bindings);
            mem::swap(&mut candidate.bindings, &mut accumulated_bindings);
            candidate.bindings.clear();

            if !changed {
                // If we were not able to simplify anymore, done.
                break;
            }
        }

        // Store computed bindings back in `candidate`.
        mem::swap(&mut candidate.bindings, &mut accumulated_bindings);

        let did_expand_or =
            if let [MatchPair { pattern: Pat { kind: PatKind::Or { pats }, .. }, place }] =
                &*candidate.match_pairs
            {
                candidate.subcandidates = self.create_or_subcandidates(candidate, place, pats);
                candidate.match_pairs.clear();
                true
            } else {
                false
            };

        // Move or-patterns to the end, because they can result in us
        // creating additional candidates, so we want to test them as
        // late as possible.
        candidate.match_pairs.sort_by_key(|pair| matches!(pair.pattern.kind, PatKind::Or { .. }));
        debug!(simplified = ?candidate, "simplify_candidate");

        did_expand_or
    }

    /// Given `candidate` that has a single or-pattern for its match-pairs,
    /// creates a fresh candidate for each of its input subpatterns passed via
    /// `pats`.
    fn create_or_subcandidates<'pat>(
        &mut self,
        candidate: &Candidate<'pat, 'tcx>,
        place: &PlaceBuilder<'tcx>,
        pats: &'pat [Box<Pat<'tcx>>],
    ) -> Vec<Candidate<'pat, 'tcx>> {
        pats.iter()
            .map(|box pat| {
                let mut candidate = Candidate::new(place.clone(), pat, candidate.has_guard, self);
                self.simplify_candidate(&mut candidate);
                candidate
            })
            .collect()
    }

    /// Tries to simplify `match_pair`, returning `Ok(())` if
    /// successful. If successful, new match pairs and bindings will
    /// have been pushed into the candidate. If no simplification is
    /// possible, `Err` is returned and no changes are made to
    /// candidate.
    fn simplify_match_pair<'pat>(
        &mut self,
        match_pair: MatchPair<'pat, 'tcx>,
        candidate: &mut Candidate<'pat, 'tcx>,
    ) -> Result<(), MatchPair<'pat, 'tcx>> {
        match match_pair.pattern.kind {
            PatKind::AscribeUserType {
                ref subpattern,
                ascription: thir::Ascription { ref annotation, variance },
            } => {
                // Apply the type ascription to the value at `match_pair.place`
                if let Some(source) = match_pair.place.try_to_place(self) {
                    candidate.ascriptions.push(Ascription {
                        annotation: annotation.clone(),
                        source,
                        variance,
                    });
                }

                candidate.match_pairs.push(MatchPair::new(match_pair.place, subpattern, self));

                Ok(())
            }

            PatKind::Wild | PatKind::Error(_) => {
                // nothing left to do
                Ok(())
            }

            PatKind::Binding {
                name: _,
                mutability: _,
                mode,
                var,
                ty: _,
                ref subpattern,
                is_primary: _,
            } => {
                if let Some(source) = match_pair.place.try_to_place(self) {
                    candidate.bindings.push(Binding {
                        span: match_pair.pattern.span,
                        source,
                        var_id: var,
                        binding_mode: mode,
                    });
                }

                if let Some(subpattern) = subpattern.as_ref() {
                    // this is the `x @ P` case; have to keep matching against `P` now
                    candidate.match_pairs.push(MatchPair::new(match_pair.place, subpattern, self));
                }

                Ok(())
            }

            PatKind::Never => {
                // A never pattern acts like a load from the place.
                // FIXME(never_patterns): load from the place
                Ok(())
            }

            PatKind::Constant { .. } => {
                // FIXME normalize patterns when possible
                Err(match_pair)
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
                    candidate.ascriptions.push(Ascription {
                        annotation,
                        source,
                        variance: ty::Contravariant,
                    });
                }
                candidate.match_pairs.push(MatchPair::new(match_pair.place, pattern, self));

                Ok(())
            }

            PatKind::Range(ref range) => {
                if let Some(true) = range.is_full_range(self.tcx) {
                    // Irrefutable pattern match.
                    return Ok(());
                }
                Err(match_pair)
            }

            PatKind::Slice { ref prefix, ref slice, ref suffix } => {
                if prefix.is_empty() && slice.is_some() && suffix.is_empty() {
                    // irrefutable
                    self.prefix_slice_suffix(
                        &mut candidate.match_pairs,
                        &match_pair.place,
                        prefix,
                        slice,
                        suffix,
                    );
                    Ok(())
                } else {
                    Err(match_pair)
                }
            }

            PatKind::Variant { adt_def, args, variant_index, ref subpatterns } => {
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
                if irrefutable {
                    let place_builder = match_pair.place.downcast(adt_def, variant_index);
                    candidate
                        .match_pairs
                        .extend(self.field_match_pairs(place_builder, subpatterns));
                    Ok(())
                } else {
                    Err(match_pair)
                }
            }

            PatKind::Array { ref prefix, ref slice, ref suffix } => {
                self.prefix_slice_suffix(
                    &mut candidate.match_pairs,
                    &match_pair.place,
                    prefix,
                    slice,
                    suffix,
                );
                Ok(())
            }

            PatKind::Leaf { ref subpatterns } => {
                // tuple struct, match subpats (if any)
                candidate.match_pairs.extend(self.field_match_pairs(match_pair.place, subpatterns));
                Ok(())
            }

            PatKind::Deref { ref subpattern } => {
                let place_builder = match_pair.place.deref();
                candidate.match_pairs.push(MatchPair::new(place_builder, subpattern, self));
                Ok(())
            }

            PatKind::Or { .. } => Err(match_pair),
        }
    }
}
