//! Analysis of patterns, notably match exhaustiveness checking.

pub mod constructor;
pub mod errors;
pub(crate) mod lints;
pub mod pat;
pub mod rustc;
pub mod usefulness;

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

use std::fmt;

use constructor::{Constructor, ConstructorSet};
use lints::PatternColumn;
use rustc_hir::HirId;
use rustc_index::Idx;
use rustc_middle::ty::Ty;
use usefulness::{compute_match_usefulness, ValidityConstraint};

use crate::lints::{lint_nonexhaustive_missing_variants, lint_overlapping_range_endpoints};
use crate::pat::DeconstructedPat;
use crate::rustc::RustcCtxt;

pub trait MatchCx: Sized + Clone + fmt::Debug {
    type Ty: Copy + Clone + fmt::Debug; // FIXME: remove Copy
    type Span: Clone + Default;
    type VariantIdx: Clone + Idx;
    type StrLit: Clone + PartialEq + fmt::Debug;

    fn is_opaque_ty(ty: Self::Ty) -> bool;
    fn is_exhaustive_patterns_feature_on(&self) -> bool;

    /// The number of fields for this constructor.
    fn ctor_arity(&self, ctor: &Constructor<Self>, ty: Self::Ty) -> usize;

    /// The types of the fields for this constructor. The result must have a length of
    /// `ctor_arity()`.
    fn ctor_sub_tys(&self, ctor: &Constructor<Self>, ty: Self::Ty) -> &[Self::Ty];

    /// The set of all the constructors for `ty`.
    ///
    /// This must follow the invariants of `ConstructorSet`
    fn ctors_for_ty(&self, ty: Self::Ty) -> ConstructorSet<Self>;

    /// Best-effort `Debug` implementation.
    fn debug_pat(f: &mut fmt::Formatter<'_>, pat: &DeconstructedPat<'_, Self>) -> fmt::Result;

    /// Raise a bug.
    fn bug(&self, fmt: fmt::Arguments<'_>) -> !;
}

/// The arm of a match expression.
#[derive(Clone, Debug)]
pub struct MatchArm<'p, Cx: MatchCx> {
    /// The pattern must have been lowered through `check_match::MatchVisitor::lower_pattern`.
    pub pat: &'p DeconstructedPat<'p, Cx>,
    pub hir_id: HirId,
    pub has_guard: bool,
}

impl<'p, Cx: MatchCx> Copy for MatchArm<'p, Cx> {}

/// The entrypoint for this crate. Computes whether a match is exhaustive and which of its arms are
/// useful, and runs some lints.
pub fn analyze_match<'p, 'tcx>(
    cx: &RustcCtxt<'p, 'tcx>,
    arms: &[rustc::MatchArm<'p, 'tcx>],
    scrut_ty: Ty<'tcx>,
) -> rustc::UsefulnessReport<'p, 'tcx> {
    // Arena to store the extra wildcards we construct during analysis.
    let wildcard_arena = cx.pattern_arena;
    let pat_column = PatternColumn::new(arms);

    let scrut_validity = ValidityConstraint::from_bool(cx.known_valid_scrutinee);
    let report = compute_match_usefulness(cx, arms, scrut_ty, scrut_validity, wildcard_arena);

    // Lint on ranges that overlap on their endpoints, which is likely a mistake.
    lint_overlapping_range_endpoints(cx, &pat_column, wildcard_arena);

    // Run the non_exhaustive_omitted_patterns lint. Only run on refutable patterns to avoid hitting
    // `if let`s. Only run if the match is exhaustive otherwise the error is redundant.
    if cx.refutable && report.non_exhaustiveness_witnesses.is_empty() {
        lint_nonexhaustive_missing_variants(cx, arms, &pat_column, scrut_ty, wildcard_arena)
    }

    report
}
