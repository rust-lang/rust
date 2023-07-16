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
use rustc_hir::RangeEnd;
use rustc_middle::thir::{self, *};
use rustc_middle::ty;
use rustc_middle::ty::layout::IntegerExt;
use rustc_target::abi::{Integer, Size};

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
        // repeatedly simplify match pairs until fixed point is reached
        debug!("{candidate:#?}");

        // existing_bindings and new_bindings exists to keep the semantics in order.
        // Reversing the binding order for bindings after `@` changes the binding order in places
        // it shouldn't be changed, for example `let (Some(a), Some(b)) = (x, y)`
        //
        // To avoid this, the binding occurs in the following manner:
        // * the bindings for one iteration of the following loop occurs in order (i.e. left to
        // right)
        // * the bindings from the previous iteration of the loop is prepended to the bindings from
        // the current iteration (in the implementation this is done by mem::swap and extend)
        // * after all iterations, these new bindings are then appended to the bindings that were
        // preexisting (i.e. `candidate.binding` when the function was called).
        //
        // example:
        // candidate.bindings = [1, 2, 3]
        // binding in iter 1: [4, 5]
        // binding in iter 2: [6, 7]
        //
        // final binding: [1, 2, 3, 6, 7, 4, 5]
        let mut existing_bindings = mem::take(&mut candidate.bindings);
        let mut new_bindings = Vec::new();
        loop {
            let match_pairs = mem::take(&mut candidate.match_pairs);

            if let [MatchPair { pattern: Pat { kind: PatKind::Or { pats }, .. }, place }] =
                &*match_pairs
            {
                existing_bindings.extend_from_slice(&new_bindings);
                mem::swap(&mut candidate.bindings, &mut existing_bindings);
                candidate.subcandidates = self.create_or_subcandidates(candidate, &place, pats);
                return true;
            }

            let mut changed = false;
            for match_pair in match_pairs {
                match self.simplify_match_pair(match_pair, candidate) {
                    Ok(()) => {
                        changed = true;
                    }
                    Err(match_pair) => {
                        candidate.match_pairs.push(match_pair);
                    }
                }
            }
            // Avoid issue #69971: the binding order should be right to left if there are more
            // bindings after `@` to please the borrow checker
            // Ex
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
            candidate.bindings.extend_from_slice(&new_bindings);
            mem::swap(&mut candidate.bindings, &mut new_bindings);
            candidate.bindings.clear();

            if !changed {
                existing_bindings.extend_from_slice(&new_bindings);
                mem::swap(&mut candidate.bindings, &mut existing_bindings);
                // Move or-patterns to the end, because they can result in us
                // creating additional candidates, so we want to test them as
                // late as possible.
                candidate
                    .match_pairs
                    .sort_by_key(|pair| matches!(pair.pattern.kind, PatKind::Or { .. }));
                debug!(simplified = ?candidate, "simplify_candidate");
                return false; // if we were not able to simplify any, done.
            }
        }
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
        let tcx = self.tcx;
        match match_pair.pattern.kind {
            PatKind::AscribeUserType {
                ref subpattern,
                ascription: thir::Ascription { ref annotation, variance },
            } => {
                // Apply the type ascription to the value at `match_pair.place`, which is the
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

            PatKind::Wild => {
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

            PatKind::Constant { .. } => {
                // FIXME normalize patterns when possible
                Err(match_pair)
            }

            PatKind::Range(box PatRange { lo, hi, end }) => {
                let (range, bias) = match *lo.ty().kind() {
                    ty::Char => {
                        (Some(('\u{0000}' as u128, '\u{10FFFF}' as u128, Size::from_bits(32))), 0)
                    }
                    ty::Int(ity) => {
                        let size = Integer::from_int_ty(&tcx, ity).size();
                        let max = size.truncate(u128::MAX);
                        let bias = 1u128 << (size.bits() - 1);
                        (Some((0, max, size)), bias)
                    }
                    ty::Uint(uty) => {
                        let size = Integer::from_uint_ty(&tcx, uty).size();
                        let max = size.truncate(u128::MAX);
                        (Some((0, max, size)), 0)
                    }
                    _ => (None, 0),
                };
                if let Some((min, max, sz)) = range {
                    // We want to compare ranges numerically, but the order of the bitwise
                    // representation of signed integers does not match their numeric order. Thus,
                    // to correct the ordering, we need to shift the range of signed integers to
                    // correct the comparison. This is achieved by XORing with a bias (see
                    // pattern/_match.rs for another pertinent example of this pattern).
                    //
                    // Also, for performance, it's important to only do the second `try_to_bits` if
                    // necessary.
                    let lo = lo.try_to_bits(sz).unwrap() ^ bias;
                    if lo <= min {
                        let hi = hi.try_to_bits(sz).unwrap() ^ bias;
                        if hi > max || hi == max && end == RangeEnd::Included {
                            // Irrefutable pattern match.
                            return Ok(());
                        }
                    }
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
                        self.tcx.features().exhaustive_patterns
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
