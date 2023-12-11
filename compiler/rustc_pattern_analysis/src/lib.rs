//! Analysis of patterns, notably match exhaustiveness checking.

pub mod constructor;
pub mod cx;
pub mod errors;
pub(crate) mod lints;
pub mod pat;
pub mod usefulness;

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

use lints::PatternColumn;
use rustc_hir::HirId;
use rustc_middle::ty::Ty;
use usefulness::{compute_match_usefulness, UsefulnessReport};

use crate::cx::MatchCheckCtxt;
use crate::lints::{lint_nonexhaustive_missing_variants, lint_overlapping_range_endpoints};
use crate::pat::DeconstructedPat;

/// The arm of a match expression.
#[derive(Clone, Copy, Debug)]
pub struct MatchArm<'p, 'tcx> {
    /// The pattern must have been lowered through `check_match::MatchVisitor::lower_pattern`.
    pub pat: &'p DeconstructedPat<'p, 'tcx>,
    pub hir_id: HirId,
    pub has_guard: bool,
}

/// The entrypoint for this crate. Computes whether a match is exhaustive and which of its arms are
/// useful, and runs some lints.
pub fn analyze_match<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    arms: &[MatchArm<'p, 'tcx>],
    scrut_ty: Ty<'tcx>,
) -> UsefulnessReport<'p, 'tcx> {
    let pat_column = PatternColumn::new(arms);

    let report = compute_match_usefulness(cx, arms, scrut_ty);

    // Lint on ranges that overlap on their endpoints, which is likely a mistake.
    lint_overlapping_range_endpoints(cx, &pat_column);

    // Run the non_exhaustive_omitted_patterns lint. Only run on refutable patterns to avoid hitting
    // `if let`s. Only run if the match is exhaustive otherwise the error is redundant.
    if cx.refutable && report.non_exhaustiveness_witnesses.is_empty() {
        lint_nonexhaustive_missing_variants(cx, arms, &pat_column, scrut_ty)
    }

    report
}
