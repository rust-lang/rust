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

use build::Builder;
use build::matches::{Ascription, Binding, MatchPair, Candidate};
use hair::*;
use rustc::ty;
use rustc::ty::layout::{Integer, IntegerExt, Size};
use syntax::attr::{SignedInt, UnsignedInt};
use rustc::hir::RangeEnd;

use std::mem;

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    pub fn simplify_candidate<'pat>(&mut self,
                                    candidate: &mut Candidate<'pat, 'tcx>) {
        // repeatedly simplify match pairs until fixed point is reached
        loop {
            let match_pairs = mem::replace(&mut candidate.match_pairs, vec![]);
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
            if !changed {
                return; // if we were not able to simplify any, done.
            }
        }
    }

    /// Tries to simplify `match_pair`, returning true if
    /// successful. If successful, new match pairs and bindings will
    /// have been pushed into the candidate. If no simplification is
    /// possible, Err is returned and no changes are made to
    /// candidate.
    fn simplify_match_pair<'pat>(&mut self,
                                 match_pair: MatchPair<'pat, 'tcx>,
                                 candidate: &mut Candidate<'pat, 'tcx>)
                                 -> Result<(), MatchPair<'pat, 'tcx>> {
        let tcx = self.hir.tcx();
        match *match_pair.pattern.kind {
            PatternKind::AscribeUserType {
                ref subpattern,
                variance,
                ref user_ty,
                user_ty_span
            } => {
                // Apply the type ascription to the value at `match_pair.place`, which is the
                // value being matched, taking the variance field into account.
                candidate.ascriptions.push(Ascription {
                    span: user_ty_span,
                    user_ty: user_ty.clone(),
                    source: match_pair.place.clone(),
                    variance,
                });

                candidate.match_pairs.push(MatchPair::new(match_pair.place, subpattern));

                Ok(())
            }

            PatternKind::Wild => {
                // nothing left to do
                Ok(())
            }

            PatternKind::Binding { name, mutability, mode, var, ty, ref subpattern } => {
                candidate.bindings.push(Binding {
                    name,
                    mutability,
                    span: match_pair.pattern.span,
                    source: match_pair.place.clone(),
                    var_id: var,
                    var_ty: ty,
                    binding_mode: mode,
                });

                if let Some(subpattern) = subpattern.as_ref() {
                    // this is the `x @ P` case; have to keep matching against `P` now
                    candidate.match_pairs.push(MatchPair::new(match_pair.place, subpattern));
                }

                Ok(())
            }

            PatternKind::Constant { .. } => {
                // FIXME normalize patterns when possible
                Err(match_pair)
            }

            PatternKind::Range(PatternRange { lo, hi, ty, end }) => {
                let range = match ty.sty {
                    ty::Char => {
                        Some(('\u{0000}' as u128, '\u{10FFFF}' as u128, Size::from_bits(32)))
                    }
                    ty::Int(ity) => {
                        // FIXME(49937): refactor these bit manipulations into interpret.
                        let size = Integer::from_attr(&tcx, SignedInt(ity)).size();
                        let min = 1u128 << (size.bits() - 1);
                        let max = (1u128 << (size.bits() - 1)) - 1;
                        Some((min, max, size))
                    }
                    ty::Uint(uty) => {
                        // FIXME(49937): refactor these bit manipulations into interpret.
                        let size = Integer::from_attr(&tcx, UnsignedInt(uty)).size();
                        let max = !0u128 >> (128 - size.bits());
                        Some((0, max, size))
                    }
                    _ => None,
                };
                if let Some((min, max, sz)) = range {
                    if let (Some(lo), Some(hi)) = (lo.val.try_to_bits(sz), hi.val.try_to_bits(sz)) {
                        if lo <= min && (hi > max || hi == max && end == RangeEnd::Included) {
                            // Irrefutable pattern match.
                            return Ok(());
                        }
                    }
                }
                Err(match_pair)
            }

            PatternKind::Slice { ref prefix, ref slice, ref suffix } => {
                if prefix.is_empty() && slice.is_some() && suffix.is_empty() {
                    // irrefutable
                    self.prefix_slice_suffix(&mut candidate.match_pairs,
                                             &match_pair.place,
                                             prefix,
                                             slice.as_ref(),
                                             suffix);
                    Ok(())
                } else {
                    Err(match_pair)
                }
            }

            PatternKind::Variant { adt_def, substs, variant_index, ref subpatterns } => {
                let irrefutable = adt_def.variants.iter_enumerated().all(|(i, v)| {
                    i == variant_index || {
                        self.hir.tcx().features().never_type &&
                        self.hir.tcx().features().exhaustive_patterns &&
                        self.hir.tcx().is_variant_uninhabited_from_all_modules(v, substs)
                    }
                });
                if irrefutable {
                    let place = match_pair.place.downcast(adt_def, variant_index);
                    candidate.match_pairs.extend(self.field_match_pairs(place, subpatterns));
                    Ok(())
                } else {
                    Err(match_pair)
                }
            },

            PatternKind::Array { ref prefix, ref slice, ref suffix } => {
                self.prefix_slice_suffix(&mut candidate.match_pairs,
                                         &match_pair.place,
                                         prefix,
                                         slice.as_ref(),
                                         suffix);
                Ok(())
            }

            PatternKind::Leaf { ref subpatterns } => {
                // tuple struct, match subpats (if any)
                candidate.match_pairs
                         .extend(self.field_match_pairs(match_pair.place, subpatterns));
                Ok(())
            }

            PatternKind::Deref { ref subpattern } => {
                let place = match_pair.place.deref();
                candidate.match_pairs.push(MatchPair::new(place, subpattern));
                Ok(())
            }
        }
    }
}
