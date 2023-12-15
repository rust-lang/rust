use smallvec::SmallVec;

use rustc_data_structures::captures::Captures;
use rustc_middle::ty::{self, Ty};
use rustc_session::lint;
use rustc_session::lint::builtin::NON_EXHAUSTIVE_OMITTED_PATTERNS;
use rustc_span::Span;

use crate::constructor::{IntRange, MaybeInfiniteInt};
use crate::errors::{
    NonExhaustiveOmittedPattern, NonExhaustiveOmittedPatternLintOnArm, Overlap,
    OverlappingRangeEndpoints, Uncovered,
};
use crate::rustc::{
    Constructor, DeconstructedPat, MatchArm, MatchCtxt, PlaceCtxt, RustcMatchCheckCtxt,
    SplitConstructorSet, WitnessPat,
};
use crate::TypeCx;

/// A column of patterns in the matrix, where a column is the intuitive notion of "subpatterns that
/// inspect the same subvalue/place".
/// This is used to traverse patterns column-by-column for lints. Despite similarities with the
/// algorithm in [`crate::usefulness`], this does a different traversal. Notably this is linear in
/// the depth of patterns, whereas `compute_exhaustiveness_and_usefulness` is worst-case exponential
/// (exhaustiveness is NP-complete). The core difference is that we treat sub-columns separately.
///
/// This must not contain an or-pattern. `specialize` takes care to expand them.
///
/// This is not used in the main algorithm; only in lints.
#[derive(Debug)]
pub(crate) struct PatternColumn<'a, 'p, 'tcx> {
    patterns: Vec<&'a DeconstructedPat<'p, 'tcx>>,
}

impl<'a, 'p, 'tcx> PatternColumn<'a, 'p, 'tcx> {
    pub(crate) fn new(arms: &[MatchArm<'p, 'tcx>]) -> Self {
        let mut patterns = Vec::with_capacity(arms.len());
        for arm in arms {
            if arm.pat.is_or_pat() {
                patterns.extend(arm.pat.flatten_or_pat())
            } else {
                patterns.push(arm.pat)
            }
        }
        Self { patterns }
    }

    fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
    fn head_ty(&self) -> Option<Ty<'tcx>> {
        if self.patterns.len() == 0 {
            return None;
        }
        // If the type is opaque and it is revealed anywhere in the column, we take the revealed
        // version. Otherwise we could encounter constructors for the revealed type and crash.
        let first_ty = self.patterns[0].ty();
        if RustcMatchCheckCtxt::is_opaque_ty(first_ty) {
            for pat in &self.patterns {
                let ty = pat.ty();
                if !RustcMatchCheckCtxt::is_opaque_ty(ty) {
                    return Some(ty);
                }
            }
        }
        Some(first_ty)
    }

    /// Do constructor splitting on the constructors of the column.
    fn analyze_ctors(&self, pcx: &PlaceCtxt<'_, 'p, 'tcx>) -> SplitConstructorSet<'p, 'tcx> {
        let column_ctors = self.patterns.iter().map(|p| p.ctor());
        pcx.ctors_for_ty().split(pcx, column_ctors)
    }

    fn iter<'b>(&'b self) -> impl Iterator<Item = &'a DeconstructedPat<'p, 'tcx>> + Captures<'b> {
        self.patterns.iter().copied()
    }

    /// Does specialization: given a constructor, this takes the patterns from the column that match
    /// the constructor, and outputs their fields.
    /// This returns one column per field of the constructor. They usually all have the same length
    /// (the number of patterns in `self` that matched `ctor`), except that we expand or-patterns
    /// which may change the lengths.
    fn specialize(
        &self,
        pcx: &PlaceCtxt<'a, 'p, 'tcx>,
        ctor: &Constructor<'p, 'tcx>,
    ) -> Vec<PatternColumn<'a, 'p, 'tcx>> {
        let arity = ctor.arity(pcx);
        if arity == 0 {
            return Vec::new();
        }

        // We specialize the column by `ctor`. This gives us `arity`-many columns of patterns. These
        // columns may have different lengths in the presence of or-patterns (this is why we can't
        // reuse `Matrix`).
        let mut specialized_columns: Vec<_> =
            (0..arity).map(|_| Self { patterns: Vec::new() }).collect();
        let relevant_patterns =
            self.patterns.iter().filter(|pat| ctor.is_covered_by(pcx, pat.ctor()));
        for pat in relevant_patterns {
            let specialized = pat.specialize(pcx, ctor);
            for (subpat, column) in specialized.iter().zip(&mut specialized_columns) {
                if subpat.is_or_pat() {
                    column.patterns.extend(subpat.flatten_or_pat())
                } else {
                    column.patterns.push(subpat)
                }
            }
        }

        assert!(
            !specialized_columns[0].is_empty(),
            "ctor {ctor:?} was listed as present but isn't;
            there is an inconsistency between `Constructor::is_covered_by` and `ConstructorSet::split`"
        );
        specialized_columns
    }
}

/// Traverse the patterns to collect any variants of a non_exhaustive enum that fail to be mentioned
/// in a given column.
#[instrument(level = "debug", skip(cx), ret)]
fn collect_nonexhaustive_missing_variants<'a, 'p, 'tcx>(
    cx: MatchCtxt<'a, 'p, 'tcx>,
    column: &PatternColumn<'a, 'p, 'tcx>,
) -> Vec<WitnessPat<'p, 'tcx>> {
    let Some(ty) = column.head_ty() else {
        return Vec::new();
    };
    let pcx = &PlaceCtxt::new_dummy(cx, ty);

    let set = column.analyze_ctors(pcx);
    if set.present.is_empty() {
        // We can't consistently handle the case where no constructors are present (since this would
        // require digging deep through any type in case there's a non_exhaustive enum somewhere),
        // so for consistency we refuse to handle the top-level case, where we could handle it.
        return vec![];
    }

    let mut witnesses = Vec::new();
    if cx.tycx.is_foreign_non_exhaustive_enum(ty) {
        witnesses.extend(
            set.missing
                .into_iter()
                // This will list missing visible variants.
                .filter(|c| !matches!(c, Constructor::Hidden | Constructor::NonExhaustive))
                .map(|missing_ctor| WitnessPat::wild_from_ctor(pcx, missing_ctor)),
        )
    }

    // Recurse into the fields.
    for ctor in set.present {
        let specialized_columns = column.specialize(pcx, &ctor);
        let wild_pat = WitnessPat::wild_from_ctor(pcx, ctor);
        for (i, col_i) in specialized_columns.iter().enumerate() {
            // Compute witnesses for each column.
            let wits_for_col_i = collect_nonexhaustive_missing_variants(cx, col_i);
            // For each witness, we build a new pattern in the shape of `ctor(_, _, wit, _, _)`,
            // adding enough wildcards to match `arity`.
            for wit in wits_for_col_i {
                let mut pat = wild_pat.clone();
                pat.fields[i] = wit;
                witnesses.push(pat);
            }
        }
    }
    witnesses
}

pub(crate) fn lint_nonexhaustive_missing_variants<'a, 'p, 'tcx>(
    cx: MatchCtxt<'a, 'p, 'tcx>,
    arms: &[MatchArm<'p, 'tcx>],
    pat_column: &PatternColumn<'a, 'p, 'tcx>,
    scrut_ty: Ty<'tcx>,
) {
    let rcx: &RustcMatchCheckCtxt<'_, '_> = cx.tycx;
    if !matches!(
        rcx.tcx.lint_level_at_node(NON_EXHAUSTIVE_OMITTED_PATTERNS, rcx.match_lint_level).0,
        rustc_session::lint::Level::Allow
    ) {
        let witnesses = collect_nonexhaustive_missing_variants(cx, pat_column);
        if !witnesses.is_empty() {
            // Report that a match of a `non_exhaustive` enum marked with `non_exhaustive_omitted_patterns`
            // is not exhaustive enough.
            //
            // NB: The partner lint for structs lives in `compiler/rustc_hir_analysis/src/check/pat.rs`.
            rcx.tcx.emit_spanned_lint(
                NON_EXHAUSTIVE_OMITTED_PATTERNS,
                rcx.match_lint_level,
                rcx.scrut_span,
                NonExhaustiveOmittedPattern {
                    scrut_ty,
                    uncovered: Uncovered::new(rcx.scrut_span, rcx, witnesses),
                },
            );
        }
    } else {
        // We used to allow putting the `#[allow(non_exhaustive_omitted_patterns)]` on a match
        // arm. This no longer makes sense so we warn users, to avoid silently breaking their
        // usage of the lint.
        for arm in arms {
            let (lint_level, lint_level_source) =
                rcx.tcx.lint_level_at_node(NON_EXHAUSTIVE_OMITTED_PATTERNS, arm.arm_data);
            if !matches!(lint_level, rustc_session::lint::Level::Allow) {
                let decorator = NonExhaustiveOmittedPatternLintOnArm {
                    lint_span: lint_level_source.span(),
                    suggest_lint_on_match: rcx.whole_match_span.map(|span| span.shrink_to_lo()),
                    lint_level: lint_level.as_str(),
                    lint_name: "non_exhaustive_omitted_patterns",
                };

                use rustc_errors::DecorateLint;
                let mut err = rcx.tcx.sess.struct_span_warn(*arm.pat.data(), "");
                err.set_primary_message(decorator.msg());
                decorator.decorate_lint(&mut err);
                err.emit();
            }
        }
    }
}

/// Traverse the patterns to warn the user about ranges that overlap on their endpoints.
#[instrument(level = "debug", skip(cx))]
pub(crate) fn lint_overlapping_range_endpoints<'a, 'p, 'tcx>(
    cx: MatchCtxt<'a, 'p, 'tcx>,
    column: &PatternColumn<'a, 'p, 'tcx>,
) {
    let Some(ty) = column.head_ty() else {
        return;
    };
    let pcx = &PlaceCtxt::new_dummy(cx, ty);
    let rcx: &RustcMatchCheckCtxt<'_, '_> = cx.tycx;

    let set = column.analyze_ctors(pcx);

    if matches!(ty.kind(), ty::Char | ty::Int(_) | ty::Uint(_)) {
        let emit_lint = |overlap: &IntRange, this_span: Span, overlapped_spans: &[Span]| {
            let overlap_as_pat = rcx.hoist_pat_range(overlap, ty);
            let overlaps: Vec<_> = overlapped_spans
                .iter()
                .copied()
                .map(|span| Overlap { range: overlap_as_pat.clone(), span })
                .collect();
            rcx.tcx.emit_spanned_lint(
                lint::builtin::OVERLAPPING_RANGE_ENDPOINTS,
                rcx.match_lint_level,
                this_span,
                OverlappingRangeEndpoints { overlap: overlaps, range: this_span },
            );
        };

        // If two ranges overlapped, the split set will contain their intersection as a singleton.
        let split_int_ranges = set.present.iter().filter_map(|c| c.as_int_range());
        for overlap_range in split_int_ranges.clone() {
            if overlap_range.is_singleton() {
                let overlap: MaybeInfiniteInt = overlap_range.lo;
                // Ranges that look like `lo..=overlap`.
                let mut prefixes: SmallVec<[_; 1]> = Default::default();
                // Ranges that look like `overlap..=hi`.
                let mut suffixes: SmallVec<[_; 1]> = Default::default();
                // Iterate on patterns that contained `overlap`.
                for pat in column.iter() {
                    let this_span = *pat.data();
                    let Constructor::IntRange(this_range) = pat.ctor() else { continue };
                    if this_range.is_singleton() {
                        // Don't lint when one of the ranges is a singleton.
                        continue;
                    }
                    if this_range.lo == overlap {
                        // `this_range` looks like `overlap..=this_range.hi`; it overlaps with any
                        // ranges that look like `lo..=overlap`.
                        if !prefixes.is_empty() {
                            emit_lint(overlap_range, this_span, &prefixes);
                        }
                        suffixes.push(this_span)
                    } else if this_range.hi == overlap.plus_one() {
                        // `this_range` looks like `this_range.lo..=overlap`; it overlaps with any
                        // ranges that look like `overlap..=hi`.
                        if !suffixes.is_empty() {
                            emit_lint(overlap_range, this_span, &suffixes);
                        }
                        prefixes.push(this_span)
                    }
                }
            }
        }
    } else {
        // Recurse into the fields.
        for ctor in set.present {
            for col in column.specialize(pcx, &ctor) {
                lint_overlapping_range_endpoints(cx, &col);
            }
        }
    }
}
