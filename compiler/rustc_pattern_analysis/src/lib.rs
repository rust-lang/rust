//! Analysis of patterns, notably match exhaustiveness checking.

pub mod constructor;
#[cfg(feature = "rustc")]
pub mod errors;
#[cfg(feature = "rustc")]
pub(crate) mod lints;
pub mod pat;
pub mod pat_column;
#[cfg(feature = "rustc")]
pub mod rustc;
pub mod usefulness;

#[macro_use]
extern crate tracing;
#[cfg(feature = "rustc")]
#[macro_use]
extern crate rustc_middle;

#[cfg(feature = "rustc")]
rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

use std::fmt;

#[cfg(feature = "rustc")]
pub mod index {
    // Faster version when the indices of variants are `0..variants.len()`.
    pub use rustc_index::bit_set::BitSet as IdxSet;
    pub use rustc_index::Idx;
    pub use rustc_index::IndexVec as IdxContainer;
}
#[cfg(not(feature = "rustc"))]
pub mod index {
    // Slower version when the indices of variants are something else.
    pub trait Idx: Copy + PartialEq + Eq + std::hash::Hash {}
    impl<T: Copy + PartialEq + Eq + std::hash::Hash> Idx for T {}

    #[derive(Debug)]
    pub struct IdxContainer<K, V>(pub rustc_hash::FxHashMap<K, V>);
    impl<K: Idx, V> IdxContainer<K, V> {
        pub fn len(&self) -> usize {
            self.0.len()
        }
        pub fn iter_enumerated(&self) -> impl Iterator<Item = (K, &V)> {
            self.0.iter().map(|(k, v)| (*k, v))
        }
    }

    #[derive(Debug)]
    pub struct IdxSet<T>(pub rustc_hash::FxHashSet<T>);
    impl<T: Idx> IdxSet<T> {
        pub fn new_empty(_len: usize) -> Self {
            Self(Default::default())
        }
        pub fn contains(&self, elem: T) -> bool {
            self.0.contains(&elem)
        }
        pub fn insert(&mut self, elem: T) {
            self.0.insert(elem);
        }
    }
}

#[cfg(feature = "rustc")]
use rustc_middle::ty::Ty;
#[cfg(feature = "rustc")]
use rustc_span::ErrorGuaranteed;

use crate::constructor::{Constructor, ConstructorSet, IntRange};
#[cfg(feature = "rustc")]
use crate::lints::lint_nonexhaustive_missing_variants;
use crate::pat::DeconstructedPat;
use crate::pat_column::PatternColumn;
#[cfg(feature = "rustc")]
use crate::rustc::RustcMatchCheckCtxt;
#[cfg(feature = "rustc")]
use crate::usefulness::{compute_match_usefulness, ValidityConstraint};

pub trait Captures<'a> {}
impl<'a, T: ?Sized> Captures<'a> for T {}

/// Context that provides type information about constructors.
///
/// Most of the crate is parameterized on a type that implements this trait.
pub trait TypeCx: Sized + fmt::Debug {
    /// The type of a pattern.
    type Ty: Clone + fmt::Debug;
    /// Errors that can abort analysis.
    type Error: fmt::Debug;
    /// The index of an enum variant.
    type VariantIdx: Clone + index::Idx + fmt::Debug;
    /// A string literal
    type StrLit: Clone + PartialEq + fmt::Debug;
    /// Extra data to store in a match arm.
    type ArmData: Copy + Clone + fmt::Debug;
    /// Extra data to store in a pattern.
    type PatData: Clone;

    fn is_exhaustive_patterns_feature_on(&self) -> bool;
    fn is_min_exhaustive_patterns_feature_on(&self) -> bool;

    /// The number of fields for this constructor.
    fn ctor_arity(&self, ctor: &Constructor<Self>, ty: &Self::Ty) -> usize;

    /// The types of the fields for this constructor. The result must have a length of
    /// `ctor_arity()`.
    fn ctor_sub_tys<'a>(
        &'a self,
        ctor: &'a Constructor<Self>,
        ty: &'a Self::Ty,
    ) -> impl Iterator<Item = Self::Ty> + ExactSizeIterator + Captures<'a>;

    /// The set of all the constructors for `ty`.
    ///
    /// This must follow the invariants of `ConstructorSet`
    fn ctors_for_ty(&self, ty: &Self::Ty) -> Result<ConstructorSet<Self>, Self::Error>;

    /// Write the name of the variant represented by `pat`. Used for the best-effort `Debug` impl of
    /// `DeconstructedPat`. Only invoqued when `pat.ctor()` is `Struct | Variant(_) | UnionField`.
    fn write_variant_name(
        f: &mut fmt::Formatter<'_>,
        pat: &crate::pat::DeconstructedPat<Self>,
    ) -> fmt::Result;

    /// Raise a bug.
    fn bug(&self, fmt: fmt::Arguments<'_>) -> Self::Error;

    /// Lint that the range `pat` overlapped with all the ranges in `overlaps_with`, where the range
    /// they overlapped over is `overlaps_on`. We only detect singleton overlaps.
    /// The default implementation does nothing.
    fn lint_overlapping_range_endpoints(
        &self,
        _pat: &DeconstructedPat<Self>,
        _overlaps_on: IntRange,
        _overlaps_with: &[&DeconstructedPat<Self>],
    ) {
    }
}

/// The arm of a match expression.
#[derive(Debug)]
pub struct MatchArm<'p, Cx: TypeCx> {
    pub pat: &'p DeconstructedPat<Cx>,
    pub has_guard: bool,
    pub arm_data: Cx::ArmData,
}

impl<'p, Cx: TypeCx> Clone for MatchArm<'p, Cx> {
    fn clone(&self) -> Self {
        Self { pat: self.pat, has_guard: self.has_guard, arm_data: self.arm_data }
    }
}

impl<'p, Cx: TypeCx> Copy for MatchArm<'p, Cx> {}

/// The entrypoint for this crate. Computes whether a match is exhaustive and which of its arms are
/// useful, and runs some lints.
#[cfg(feature = "rustc")]
pub fn analyze_match<'p, 'tcx>(
    tycx: &RustcMatchCheckCtxt<'p, 'tcx>,
    arms: &[rustc::MatchArm<'p, 'tcx>],
    scrut_ty: Ty<'tcx>,
) -> Result<rustc::UsefulnessReport<'p, 'tcx>, ErrorGuaranteed> {
    let scrut_ty = tycx.reveal_opaque_ty(scrut_ty);
    let scrut_validity = ValidityConstraint::from_bool(tycx.known_valid_scrutinee);
    let report = compute_match_usefulness(tycx, arms, scrut_ty, scrut_validity)?;

    // Run the non_exhaustive_omitted_patterns lint. Only run on refutable patterns to avoid hitting
    // `if let`s. Only run if the match is exhaustive otherwise the error is redundant.
    if tycx.refutable && report.non_exhaustiveness_witnesses.is_empty() {
        let pat_column = PatternColumn::new(arms);
        lint_nonexhaustive_missing_variants(tycx, arms, &pat_column, scrut_ty)?;
    }

    Ok(report)
}
