//! Analysis of patterns, notably match exhaustiveness checking. The main entrypoint for this crate
//! is [`usefulness::compute_match_usefulness`]. For rustc-specific types and entrypoints, see the
//! [`rustc`] module.

// tidy-alphabetical-start
#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]
#![allow(unused_crate_dependencies)]
// tidy-alphabetical-end

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

#[cfg(feature = "rustc")]
rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

use std::fmt;

pub use rustc_index::{Idx, IndexVec}; // re-exported to avoid rustc_index version issues

use crate::constructor::{Constructor, ConstructorSet, IntRange};
use crate::pat::DeconstructedPat;

pub trait Captures<'a> {}
impl<'a, T: ?Sized> Captures<'a> for T {}

/// `bool` newtype that indicates whether this is a privately uninhabited field that we should skip
/// during analysis.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PrivateUninhabitedField(pub bool);

/// Context that provides type information about constructors.
///
/// Most of the crate is parameterized on a type that implements this trait.
pub trait PatCx: Sized + fmt::Debug {
    /// The type of a pattern.
    type Ty: Clone + fmt::Debug;
    /// Errors that can abort analysis.
    type Error: fmt::Debug;
    /// The index of an enum variant.
    type VariantIdx: Clone + Idx + fmt::Debug;
    /// A string literal
    type StrLit: Clone + PartialEq + fmt::Debug;
    /// Extra data to store in a match arm.
    type ArmData: Copy + Clone + fmt::Debug;
    /// Extra data to store in a pattern.
    type PatData: Clone;

    fn is_exhaustive_patterns_feature_on(&self) -> bool;

    /// The number of fields for this constructor.
    fn ctor_arity(&self, ctor: &Constructor<Self>, ty: &Self::Ty) -> usize;

    /// The types of the fields for this constructor. The result must contain `ctor_arity()` fields.
    fn ctor_sub_tys(
        &self,
        ctor: &Constructor<Self>,
        ty: &Self::Ty,
    ) -> impl Iterator<Item = (Self::Ty, PrivateUninhabitedField)> + ExactSizeIterator;

    /// The set of all the constructors for `ty`.
    ///
    /// This must follow the invariants of `ConstructorSet`
    fn ctors_for_ty(&self, ty: &Self::Ty) -> Result<ConstructorSet<Self>, Self::Error>;

    /// Write the name of the variant represented by `pat`. Used for the best-effort `Debug` impl of
    /// `DeconstructedPat`. Only invoqued when `pat.ctor()` is `Struct | Variant(_) | UnionField`.
    fn write_variant_name(
        f: &mut fmt::Formatter<'_>,
        ctor: &crate::constructor::Constructor<Self>,
        ty: &Self::Ty,
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

    /// The maximum pattern complexity limit was reached.
    fn complexity_exceeded(&self) -> Result<(), Self::Error>;

    /// Lint that there is a gap `gap` between `pat` and all of `gapped_with` such that the gap is
    /// not matched by another range. If `gapped_with` is empty, then `gap` is `T::MAX`. We only
    /// detect singleton gaps.
    /// The default implementation does nothing.
    fn lint_non_contiguous_range_endpoints(
        &self,
        _pat: &DeconstructedPat<Self>,
        _gap: IntRange,
        _gapped_with: &[&DeconstructedPat<Self>],
    ) {
    }
}

/// The arm of a match expression.
#[derive(Debug)]
pub struct MatchArm<'p, Cx: PatCx> {
    pub pat: &'p DeconstructedPat<Cx>,
    pub has_guard: bool,
    pub arm_data: Cx::ArmData,
}

impl<'p, Cx: PatCx> Clone for MatchArm<'p, Cx> {
    fn clone(&self) -> Self {
        Self { pat: self.pat, has_guard: self.has_guard, arm_data: self.arm_data }
    }
}

impl<'p, Cx: PatCx> Copy for MatchArm<'p, Cx> {}
