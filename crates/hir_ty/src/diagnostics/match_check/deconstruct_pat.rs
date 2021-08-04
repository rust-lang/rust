//! [`super::usefulness`] explains most of what is happening in this file. As explained there,
//! values and patterns are made from constructors applied to fields. This file defines a
//! `Constructor` enum, a `Fields` struct, and various operations to manipulate them and convert
//! them from/to patterns.
//!
//! There's one idea that is not detailed in [`super::usefulness`] because the details are not
//! needed there: _constructor splitting_.
//!
//! # Constructor splitting
//!
//! The idea is as follows: given a constructor `c` and a matrix, we want to specialize in turn
//! with all the value constructors that are covered by `c`, and compute usefulness for each.
//! Instead of listing all those constructors (which is intractable), we group those value
//! constructors together as much as possible. Example:
//!
//! ```
//! match (0, false) {
//!     (0 ..=100, true) => {} // `p_1`
//!     (50..=150, false) => {} // `p_2`
//!     (0 ..=200, _) => {} // `q`
//! }
//! ```
//!
//! The naive approach would try all numbers in the range `0..=200`. But we can be a lot more
//! clever: `0` and `1` for example will match the exact same rows, and return equivalent
//! witnesses. In fact all of `0..50` would. We can thus restrict our exploration to 4
//! constructors: `0..50`, `50..=100`, `101..=150` and `151..=200`. That is enough and infinitely
//! more tractable.
//!
//! We capture this idea in a function `split(p_1 ... p_n, c)` which returns a list of constructors
//! `c'` covered by `c`. Given such a `c'`, we require that all value ctors `c''` covered by `c'`
//! return an equivalent set of witnesses after specializing and computing usefulness.
//! In the example above, witnesses for specializing by `c''` covered by `0..50` will only differ
//! in their first element.
//!
//! We usually also ask that the `c'` together cover all of the original `c`. However we allow
//! skipping some constructors as long as it doesn't change whether the resulting list of witnesses
//! is empty of not. We use this in the wildcard `_` case.
//!
//! Splitting is implemented in the [`Constructor::split`] function. We don't do splitting for
//! or-patterns; instead we just try the alternatives one-by-one. For details on splitting
//! wildcards, see [`SplitWildcard`]; for integer ranges, see [`SplitIntRange`].

use std::{
    cmp::{max, min},
    iter::once,
    ops::RangeInclusive,
};

use hir_def::{EnumVariantId, HasModule, LocalFieldId, VariantId};
use smallvec::{smallvec, SmallVec};

use crate::{AdtId, Interner, Scalar, Ty, TyExt, TyKind};

use super::{
    usefulness::{MatchCheckCtx, PatCtxt},
    FieldPat, Pat, PatId, PatKind,
};

use self::Constructor::*;

/// [Constructor] uses this in umimplemented variants.
/// It allows porting match expressions from upstream algorithm without losing semantics.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Void {}

/// An inclusive interval, used for precise integer exhaustiveness checking.
/// `IntRange`s always store a contiguous range. This means that values are
/// encoded such that `0` encodes the minimum value for the integer,
/// regardless of the signedness.
/// For example, the pattern `-128..=127i8` is encoded as `0..=255`.
/// This makes comparisons and arithmetic on interval endpoints much more
/// straightforward. See `signed_bias` for details.
///
/// `IntRange` is never used to encode an empty range or a "range" that wraps
/// around the (offset) space: i.e., `range.lo <= range.hi`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct IntRange {
    range: RangeInclusive<u128>,
}

impl IntRange {
    #[inline]
    fn is_integral(ty: &Ty) -> bool {
        match ty.kind(&Interner) {
            TyKind::Scalar(Scalar::Char | Scalar::Int(_) | Scalar::Uint(_) | Scalar::Bool) => true,
            _ => false,
        }
    }

    fn is_singleton(&self) -> bool {
        self.range.start() == self.range.end()
    }

    fn boundaries(&self) -> (u128, u128) {
        (*self.range.start(), *self.range.end())
    }

    #[inline]
    fn from_bool(value: bool) -> IntRange {
        let val = value as u128;
        IntRange { range: val..=val }
    }

    #[inline]
    fn from_range(lo: u128, hi: u128, scalar_ty: Scalar) -> IntRange {
        if let Scalar::Bool = scalar_ty {
            IntRange { range: lo..=hi }
        } else {
            unimplemented!()
        }
    }

    fn is_subrange(&self, other: &Self) -> bool {
        other.range.start() <= self.range.start() && self.range.end() <= other.range.end()
    }

    fn intersection(&self, other: &Self) -> Option<Self> {
        let (lo, hi) = self.boundaries();
        let (other_lo, other_hi) = other.boundaries();
        if lo <= other_hi && other_lo <= hi {
            Some(IntRange { range: max(lo, other_lo)..=min(hi, other_hi) })
        } else {
            None
        }
    }

    /// See `Constructor::is_covered_by`
    fn is_covered_by(&self, other: &Self) -> bool {
        if self.intersection(other).is_some() {
            // Constructor splitting should ensure that all intersections we encounter are actually
            // inclusions.
            assert!(self.is_subrange(other));
            true
        } else {
            false
        }
    }
}

/// Represents a border between 2 integers. Because the intervals spanning borders must be able to
/// cover every integer, we need to be able to represent 2^128 + 1 such borders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum IntBorder {
    JustBefore(u128),
    AfterMax,
}

/// A range of integers that is partitioned into disjoint subranges. This does constructor
/// splitting for integer ranges as explained at the top of the file.
///
/// This is fed multiple ranges, and returns an output that covers the input, but is split so that
/// the only intersections between an output range and a seen range are inclusions. No output range
/// straddles the boundary of one of the inputs.
///
/// The following input:
/// ```
///   |-------------------------| // `self`
/// |------|  |----------|   |----|
///    |-------| |-------|
/// ```
/// would be iterated over as follows:
/// ```
///   ||---|--||-|---|---|---|--|
/// ```
#[derive(Debug, Clone)]
struct SplitIntRange {
    /// The range we are splitting
    range: IntRange,
    /// The borders of ranges we have seen. They are all contained within `range`. This is kept
    /// sorted.
    borders: Vec<IntBorder>,
}

impl SplitIntRange {
    fn new(range: IntRange) -> Self {
        SplitIntRange { range, borders: Vec::new() }
    }

    /// Internal use
    fn to_borders(r: IntRange) -> [IntBorder; 2] {
        use IntBorder::*;
        let (lo, hi) = r.boundaries();
        let lo = JustBefore(lo);
        let hi = match hi.checked_add(1) {
            Some(m) => JustBefore(m),
            None => AfterMax,
        };
        [lo, hi]
    }

    /// Add ranges relative to which we split.
    fn split(&mut self, ranges: impl Iterator<Item = IntRange>) {
        let this_range = &self.range;
        let included_ranges = ranges.filter_map(|r| this_range.intersection(&r));
        let included_borders = included_ranges.flat_map(|r| {
            let borders = Self::to_borders(r);
            once(borders[0]).chain(once(borders[1]))
        });
        self.borders.extend(included_borders);
        self.borders.sort_unstable();
    }

    /// Iterate over the contained ranges.
    fn iter(&self) -> impl Iterator<Item = IntRange> + '_ {
        use IntBorder::*;

        let self_range = Self::to_borders(self.range.clone());
        // Start with the start of the range.
        let mut prev_border = self_range[0];
        self.borders
            .iter()
            .copied()
            // End with the end of the range.
            .chain(once(self_range[1]))
            // List pairs of adjacent borders.
            .map(move |border| {
                let ret = (prev_border, border);
                prev_border = border;
                ret
            })
            // Skip duplicates.
            .filter(|(prev_border, border)| prev_border != border)
            // Finally, convert to ranges.
            .map(|(prev_border, border)| {
                let range = match (prev_border, border) {
                    (JustBefore(n), JustBefore(m)) if n < m => n..=(m - 1),
                    (JustBefore(n), AfterMax) => n..=u128::MAX,
                    _ => unreachable!(), // Ruled out by the sorting and filtering we did
                };
                IntRange { range }
            })
    }
}

/// A constructor for array and slice patterns.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Slice {
    _unimplemented: Void,
}

impl Slice {
    /// See `Constructor::is_covered_by`
    fn is_covered_by(self, _other: Self) -> bool {
        unimplemented!() // never called as Slice contains Void
    }
}

/// A value can be decomposed into a constructor applied to some fields. This struct represents
/// the constructor. See also `Fields`.
///
/// `pat_constructor` retrieves the constructor corresponding to a pattern.
/// `specialize_constructor` returns the list of fields corresponding to a pattern, given a
/// constructor. `Constructor::apply` reconstructs the pattern from a pair of `Constructor` and
/// `Fields`.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Constructor {
    /// The constructor for patterns that have a single constructor, like tuples, struct patterns
    /// and fixed-length arrays.
    Single,
    /// Enum variants.
    Variant(EnumVariantId),
    /// Ranges of integer literal values (`2`, `2..=5` or `2..5`).
    IntRange(IntRange),
    /// Ranges of floating-point literal values (`2.0..=5.2`).
    FloatRange(Void),
    /// String literals. Strings are not quite the same as `&[u8]` so we treat them separately.
    Str(Void),
    /// Array and slice patterns.
    Slice(Slice),
    /// Constants that must not be matched structurally. They are treated as black
    /// boxes for the purposes of exhaustiveness: we must not inspect them, and they
    /// don't count towards making a match exhaustive.
    Opaque,
    /// Fake extra constructor for enums that aren't allowed to be matched exhaustively. Also used
    /// for those types for which we cannot list constructors explicitly, like `f64` and `str`.
    NonExhaustive,
    /// Stands for constructors that are not seen in the matrix, as explained in the documentation
    /// for [`SplitWildcard`].
    Missing,
    /// Wildcard pattern.
    Wildcard,
}

impl Constructor {
    pub(super) fn is_wildcard(&self) -> bool {
        matches!(self, Wildcard)
    }

    fn as_int_range(&self) -> Option<&IntRange> {
        match self {
            IntRange(range) => Some(range),
            _ => None,
        }
    }

    fn as_slice(&self) -> Option<Slice> {
        match self {
            Slice(slice) => Some(*slice),
            _ => None,
        }
    }

    fn variant_id_for_adt(&self, adt: hir_def::AdtId) -> VariantId {
        match *self {
            Variant(id) => id.into(),
            Single => {
                assert!(!matches!(adt, hir_def::AdtId::EnumId(_)));
                match adt {
                    hir_def::AdtId::EnumId(_) => unreachable!(),
                    hir_def::AdtId::StructId(id) => id.into(),
                    hir_def::AdtId::UnionId(id) => id.into(),
                }
            }
            _ => panic!("bad constructor {:?} for adt {:?}", self, adt),
        }
    }

    /// Determines the constructor that the given pattern can be specialized to.
    pub(super) fn from_pat(cx: &MatchCheckCtx<'_>, pat: PatId) -> Self {
        match cx.pattern_arena.borrow()[pat].kind.as_ref() {
            PatKind::Binding { .. } | PatKind::Wild => Wildcard,
            PatKind::Leaf { .. } | PatKind::Deref { .. } => Single,
            &PatKind::Variant { enum_variant, .. } => Variant(enum_variant),
            &PatKind::LiteralBool { value } => IntRange(IntRange::from_bool(value)),
            PatKind::Or { .. } => cx.bug("Or-pattern should have been expanded earlier on."),
        }
    }

    /// Some constructors (namely `Wildcard`, `IntRange` and `Slice`) actually stand for a set of actual
    /// constructors (like variants, integers or fixed-sized slices). When specializing for these
    /// constructors, we want to be specialising for the actual underlying constructors.
    /// Naively, we would simply return the list of constructors they correspond to. We instead are
    /// more clever: if there are constructors that we know will behave the same wrt the current
    /// matrix, we keep them grouped. For example, all slices of a sufficiently large length
    /// will either be all useful or all non-useful with a given matrix.
    ///
    /// See the branches for details on how the splitting is done.
    ///
    /// This function may discard some irrelevant constructors if this preserves behavior and
    /// diagnostics. Eg. for the `_` case, we ignore the constructors already present in the
    /// matrix, unless all of them are.
    pub(super) fn split<'a>(
        &self,
        pcx: PatCtxt<'_>,
        ctors: impl Iterator<Item = &'a Constructor> + Clone,
    ) -> SmallVec<[Self; 1]> {
        match self {
            Wildcard => {
                let mut split_wildcard = SplitWildcard::new(pcx);
                split_wildcard.split(pcx, ctors);
                split_wildcard.into_ctors(pcx)
            }
            // Fast-track if the range is trivial. In particular, we don't do the overlapping
            // ranges check.
            IntRange(ctor_range) if !ctor_range.is_singleton() => {
                let mut split_range = SplitIntRange::new(ctor_range.clone());
                let int_ranges = ctors.filter_map(|ctor| ctor.as_int_range());
                split_range.split(int_ranges.cloned());
                split_range.iter().map(IntRange).collect()
            }
            Slice(_) => unimplemented!(),
            // Any other constructor can be used unchanged.
            _ => smallvec![self.clone()],
        }
    }

    /// Returns whether `self` is covered by `other`, i.e. whether `self` is a subset of `other`.
    /// For the simple cases, this is simply checking for equality. For the "grouped" constructors,
    /// this checks for inclusion.
    // We inline because this has a single call site in `Matrix::specialize_constructor`.
    #[inline]
    pub(super) fn is_covered_by(&self, pcx: PatCtxt<'_>, other: &Self) -> bool {
        // This must be kept in sync with `is_covered_by_any`.
        match (self, other) {
            // Wildcards cover anything
            (_, Wildcard) => true,
            // The missing ctors are not covered by anything in the matrix except wildcards.
            (Missing | Wildcard, _) => false,

            (Single, Single) => true,
            (Variant(self_id), Variant(other_id)) => self_id == other_id,

            (IntRange(self_range), IntRange(other_range)) => self_range.is_covered_by(other_range),
            (FloatRange(..), FloatRange(..)) => {
                unimplemented!()
            }
            (Str(..), Str(..)) => {
                unimplemented!()
            }
            (Slice(self_slice), Slice(other_slice)) => self_slice.is_covered_by(*other_slice),

            // We are trying to inspect an opaque constant. Thus we skip the row.
            (Opaque, _) | (_, Opaque) => false,
            // Only a wildcard pattern can match the special extra constructor.
            (NonExhaustive, _) => false,

            _ => pcx.cx.bug(&format!(
                "trying to compare incompatible constructors {:?} and {:?}",
                self, other
            )),
        }
    }

    /// Faster version of `is_covered_by` when applied to many constructors. `used_ctors` is
    /// assumed to be built from `matrix.head_ctors()` with wildcards filtered out, and `self` is
    /// assumed to have been split from a wildcard.
    fn is_covered_by_any(&self, pcx: PatCtxt<'_>, used_ctors: &[Constructor]) -> bool {
        if used_ctors.is_empty() {
            return false;
        }

        // This must be kept in sync with `is_covered_by`.
        match self {
            // If `self` is `Single`, `used_ctors` cannot contain anything else than `Single`s.
            Single => !used_ctors.is_empty(),
            Variant(_) => used_ctors.iter().any(|c| c == self),
            IntRange(range) => used_ctors
                .iter()
                .filter_map(|c| c.as_int_range())
                .any(|other| range.is_covered_by(other)),
            Slice(slice) => used_ctors
                .iter()
                .filter_map(|c| c.as_slice())
                .any(|other| slice.is_covered_by(other)),
            // This constructor is never covered by anything else
            NonExhaustive => false,
            Str(..) | FloatRange(..) | Opaque | Missing | Wildcard => {
                pcx.cx.bug(&format!("found unexpected ctor in all_ctors: {:?}", self))
            }
        }
    }
}

/// A wildcard constructor that we split relative to the constructors in the matrix, as explained
/// at the top of the file.
///
/// A constructor that is not present in the matrix rows will only be covered by the rows that have
/// wildcards. Thus we can group all of those constructors together; we call them "missing
/// constructors". Splitting a wildcard would therefore list all present constructors individually
/// (or grouped if they are integers or slices), and then all missing constructors together as a
/// group.
///
/// However we can go further: since any constructor will match the wildcard rows, and having more
/// rows can only reduce the amount of usefulness witnesses, we can skip the present constructors
/// and only try the missing ones.
/// This will not preserve the whole list of witnesses, but will preserve whether the list is empty
/// or not. In fact this is quite natural from the point of view of diagnostics too. This is done
/// in `to_ctors`: in some cases we only return `Missing`.
#[derive(Debug)]
pub(super) struct SplitWildcard {
    /// Constructors seen in the matrix.
    matrix_ctors: Vec<Constructor>,
    /// All the constructors for this type
    all_ctors: SmallVec<[Constructor; 1]>,
}

impl SplitWildcard {
    pub(super) fn new(pcx: PatCtxt<'_>) -> Self {
        let cx = pcx.cx;
        let make_range = |start, end, scalar| IntRange(IntRange::from_range(start, end, scalar));

        // Unhandled types are treated as non-exhaustive. Being explicit here instead of falling
        // to catchall arm to ease further implementation.
        let unhandled = || smallvec![NonExhaustive];

        // This determines the set of all possible constructors for the type `pcx.ty`. For numbers,
        // arrays and slices we use ranges and variable-length slices when appropriate.
        //
        // If the `exhaustive_patterns` feature is enabled, we make sure to omit constructors that
        // are statically impossible. E.g., for `Option<!>`, we do not include `Some(_)` in the
        // returned list of constructors.
        // Invariant: this is empty if and only if the type is uninhabited (as determined by
        // `cx.is_uninhabited()`).
        let all_ctors = match pcx.ty.kind(&Interner) {
            TyKind::Scalar(Scalar::Bool) => smallvec![make_range(0, 1, Scalar::Bool)],
            // TyKind::Array(..) if ... => unhandled(),
            TyKind::Array(..) | TyKind::Slice(..) => unhandled(),
            &TyKind::Adt(AdtId(hir_def::AdtId::EnumId(enum_id)), ref _substs) => {
                let enum_data = cx.db.enum_data(enum_id);

                // If the enum is declared as `#[non_exhaustive]`, we treat it as if it had an
                // additional "unknown" constructor.
                // There is no point in enumerating all possible variants, because the user can't
                // actually match against them all themselves. So we always return only the fictitious
                // constructor.
                // E.g., in an example like:
                //
                // ```
                //     let err: io::ErrorKind = ...;
                //     match err {
                //         io::ErrorKind::NotFound => {},
                //     }
                // ```
                //
                // we don't want to show every possible IO error, but instead have only `_` as the
                // witness.
                let is_declared_nonexhaustive = cx.is_foreign_non_exhaustive_enum(enum_id);

                // If `exhaustive_patterns` is disabled and our scrutinee is an empty enum, we treat it
                // as though it had an "unknown" constructor to avoid exposing its emptiness. The
                // exception is if the pattern is at the top level, because we want empty matches to be
                // considered exhaustive.
                let is_secretly_empty = enum_data.variants.is_empty()
                    && !cx.feature_exhaustive_patterns()
                    && !pcx.is_top_level;

                if is_secretly_empty || is_declared_nonexhaustive {
                    smallvec![NonExhaustive]
                } else if cx.feature_exhaustive_patterns() {
                    unimplemented!() // see MatchCheckCtx.feature_exhaustive_patterns()
                } else {
                    enum_data
                        .variants
                        .iter()
                        .map(|(local_id, ..)| Variant(EnumVariantId { parent: enum_id, local_id }))
                        .collect()
                }
            }
            TyKind::Scalar(Scalar::Char) => unhandled(),
            TyKind::Scalar(Scalar::Int(..) | Scalar::Uint(..)) => unhandled(),
            TyKind::Never if !cx.feature_exhaustive_patterns() && !pcx.is_top_level => {
                smallvec![NonExhaustive]
            }
            TyKind::Never => SmallVec::new(),
            _ if cx.is_uninhabited(pcx.ty) => SmallVec::new(),
            TyKind::Adt(..) | TyKind::Tuple(..) | TyKind::Ref(..) => smallvec![Single],
            // This type is one for which we cannot list constructors, like `str` or `f64`.
            _ => smallvec![NonExhaustive],
        };
        SplitWildcard { matrix_ctors: Vec::new(), all_ctors }
    }

    /// Pass a set of constructors relative to which to split this one. Don't call twice, it won't
    /// do what you want.
    pub(super) fn split<'a>(
        &mut self,
        pcx: PatCtxt<'_>,
        ctors: impl Iterator<Item = &'a Constructor> + Clone,
    ) {
        // Since `all_ctors` never contains wildcards, this won't recurse further.
        self.all_ctors =
            self.all_ctors.iter().flat_map(|ctor| ctor.split(pcx, ctors.clone())).collect();
        self.matrix_ctors = ctors.filter(|c| !c.is_wildcard()).cloned().collect();
    }

    /// Whether there are any value constructors for this type that are not present in the matrix.
    fn any_missing(&self, pcx: PatCtxt<'_>) -> bool {
        self.iter_missing(pcx).next().is_some()
    }

    /// Iterate over the constructors for this type that are not present in the matrix.
    pub(super) fn iter_missing<'a>(
        &'a self,
        pcx: PatCtxt<'a>,
    ) -> impl Iterator<Item = &'a Constructor> {
        self.all_ctors.iter().filter(move |ctor| !ctor.is_covered_by_any(pcx, &self.matrix_ctors))
    }

    /// Return the set of constructors resulting from splitting the wildcard. As explained at the
    /// top of the file, if any constructors are missing we can ignore the present ones.
    fn into_ctors(self, pcx: PatCtxt<'_>) -> SmallVec<[Constructor; 1]> {
        if self.any_missing(pcx) {
            // Some constructors are missing, thus we can specialize with the special `Missing`
            // constructor, which stands for those constructors that are not seen in the matrix,
            // and matches the same rows as any of them (namely the wildcard rows). See the top of
            // the file for details.
            // However, when all constructors are missing we can also specialize with the full
            // `Wildcard` constructor. The difference will depend on what we want in diagnostics.

            // If some constructors are missing, we typically want to report those constructors,
            // e.g.:
            // ```
            //     enum Direction { N, S, E, W }
            //     let Direction::N = ...;
            // ```
            // we can report 3 witnesses: `S`, `E`, and `W`.
            //
            // However, if the user didn't actually specify a constructor
            // in this arm, e.g., in
            // ```
            //     let x: (Direction, Direction, bool) = ...;
            //     let (_, _, false) = x;
            // ```
            // we don't want to show all 16 possible witnesses `(<direction-1>, <direction-2>,
            // true)` - we are satisfied with `(_, _, true)`. So if all constructors are missing we
            // prefer to report just a wildcard `_`.
            //
            // The exception is: if we are at the top-level, for example in an empty match, we
            // sometimes prefer reporting the list of constructors instead of just `_`.
            let report_when_all_missing = pcx.is_top_level && !IntRange::is_integral(pcx.ty);
            let ctor = if !self.matrix_ctors.is_empty() || report_when_all_missing {
                Missing
            } else {
                Wildcard
            };
            return smallvec![ctor];
        }

        // All the constructors are present in the matrix, so we just go through them all.
        self.all_ctors
    }
}

/// A value can be decomposed into a constructor applied to some fields. This struct represents
/// those fields, generalized to allow patterns in each field. See also `Constructor`.
/// This is constructed from a constructor using [`Fields::wildcards()`].
///
/// If a private or `non_exhaustive` field is uninhabited, the code mustn't observe that it is
/// uninhabited. For that, we filter these fields out of the matrix. This is handled automatically
/// in `Fields`. This filtering is uncommon in practice, because uninhabited fields are rarely used,
/// so we avoid it when possible to preserve performance.
#[derive(Debug, Clone)]
pub(super) enum Fields {
    /// Lists of patterns that don't contain any filtered fields.
    /// `Slice` and `Vec` behave the same; the difference is only to avoid allocating and
    /// triple-dereferences when possible. Frankly this is premature optimization, I (Nadrieril)
    /// have not measured if it really made a difference.
    Vec(SmallVec<[PatId; 2]>),
}

impl Fields {
    /// Internal use. Use `Fields::wildcards()` instead.
    /// Must not be used if the pattern is a field of a struct/tuple/variant.
    fn from_single_pattern(pat: PatId) -> Self {
        Fields::Vec(smallvec![pat])
    }

    /// Convenience; internal use.
    fn wildcards_from_tys(cx: &MatchCheckCtx<'_>, tys: impl IntoIterator<Item = Ty>) -> Self {
        let wilds = tys.into_iter().map(Pat::wildcard_from_ty);
        let pats = wilds.map(|pat| cx.alloc_pat(pat)).collect();
        Fields::Vec(pats)
    }

    /// Creates a new list of wildcard fields for a given constructor.
    pub(crate) fn wildcards(pcx: PatCtxt<'_>, constructor: &Constructor) -> Self {
        let ty = pcx.ty;
        let cx = pcx.cx;
        let wildcard_from_ty = |ty: &Ty| cx.alloc_pat(Pat::wildcard_from_ty(ty.clone()));

        let ret = match constructor {
            Single | Variant(_) => match ty.kind(&Interner) {
                TyKind::Tuple(_, substs) => {
                    let tys = substs.iter(&Interner).map(|ty| ty.assert_ty_ref(&Interner));
                    Fields::wildcards_from_tys(cx, tys.cloned())
                }
                TyKind::Ref(.., rty) => Fields::from_single_pattern(wildcard_from_ty(rty)),
                &TyKind::Adt(AdtId(adt), ref substs) => {
                    if adt_is_box(adt, cx) {
                        // Use T as the sub pattern type of Box<T>.
                        let subst_ty = substs.at(&Interner, 0).assert_ty_ref(&Interner);
                        Fields::from_single_pattern(wildcard_from_ty(subst_ty))
                    } else {
                        let variant_id = constructor.variant_id_for_adt(adt);
                        let adt_is_local =
                            variant_id.module(cx.db.upcast()).krate() == cx.module.krate();
                        // Whether we must not match the fields of this variant exhaustively.
                        let is_non_exhaustive =
                            is_field_list_non_exhaustive(variant_id, cx) && !adt_is_local;

                        cov_mark::hit!(match_check_wildcard_expanded_to_substitutions);
                        let field_ty_data = cx.db.field_types(variant_id);
                        let field_tys = || {
                            field_ty_data
                                .iter()
                                .map(|(_, binders)| binders.clone().substitute(&Interner, substs))
                        };

                        // In the following cases, we don't need to filter out any fields. This is
                        // the vast majority of real cases, since uninhabited fields are uncommon.
                        let has_no_hidden_fields = (matches!(adt, hir_def::AdtId::EnumId(_))
                            && !is_non_exhaustive)
                            || !field_tys().any(|ty| cx.is_uninhabited(&ty));

                        if has_no_hidden_fields {
                            Fields::wildcards_from_tys(cx, field_tys())
                        } else {
                            //FIXME(iDawer): see MatchCheckCtx::is_uninhabited, has_no_hidden_fields is always true
                            unimplemented!("exhaustive_patterns feature")
                        }
                    }
                }
                ty_kind => {
                    cx.bug(&format!("Unexpected type for `Single` constructor: {:?}", ty_kind))
                }
            },
            Slice(..) => {
                unimplemented!()
            }
            Str(..) | FloatRange(..) | IntRange(..) | NonExhaustive | Opaque | Missing
            | Wildcard => Fields::Vec(Default::default()),
        };
        ret
    }

    /// Apply a constructor to a list of patterns, yielding a new pattern. `self`
    /// must have as many elements as this constructor's arity.
    ///
    /// This is roughly the inverse of `specialize_constructor`.
    ///
    /// Examples:
    /// `ctor`: `Constructor::Single`
    /// `ty`: `Foo(u32, u32, u32)`
    /// `self`: `[10, 20, _]`
    /// returns `Foo(10, 20, _)`
    ///
    /// `ctor`: `Constructor::Variant(Option::Some)`
    /// `ty`: `Option<bool>`
    /// `self`: `[false]`
    /// returns `Some(false)`
    pub(super) fn apply(self, pcx: PatCtxt<'_>, ctor: &Constructor) -> Pat {
        let subpatterns_and_indices = self.patterns_and_indices();
        let mut subpatterns =
            subpatterns_and_indices.iter().map(|&(_, p)| pcx.cx.pattern_arena.borrow()[p].clone());
        // FIXME(iDawer) witnesses are not yet used
        const UNHANDLED: PatKind = PatKind::Wild;

        let pat = match ctor {
            Single | Variant(_) => match pcx.ty.kind(&Interner) {
                TyKind::Adt(..) | TyKind::Tuple(..) => {
                    // We want the real indices here.
                    let subpatterns = subpatterns_and_indices
                        .iter()
                        .map(|&(field, pat)| FieldPat {
                            field,
                            pattern: pcx.cx.pattern_arena.borrow()[pat].clone(),
                        })
                        .collect();

                    if let Some((adt, substs)) = pcx.ty.as_adt() {
                        if let hir_def::AdtId::EnumId(_) = adt {
                            let enum_variant = match ctor {
                                &Variant(id) => id,
                                _ => unreachable!(),
                            };
                            PatKind::Variant { substs: substs.clone(), enum_variant, subpatterns }
                        } else {
                            PatKind::Leaf { subpatterns }
                        }
                    } else {
                        PatKind::Leaf { subpatterns }
                    }
                }
                // Note: given the expansion of `&str` patterns done in `expand_pattern`, we should
                // be careful to reconstruct the correct constant pattern here. However a string
                // literal pattern will never be reported as a non-exhaustiveness witness, so we
                // can ignore this issue.
                TyKind::Ref(..) => PatKind::Deref { subpattern: subpatterns.next().unwrap() },
                TyKind::Slice(..) | TyKind::Array(..) => {
                    pcx.cx.bug(&format!("bad slice pattern {:?} {:?}", ctor, pcx.ty))
                }
                _ => PatKind::Wild,
            },
            Constructor::Slice(_) => UNHANDLED,
            Str(_) => UNHANDLED,
            FloatRange(..) => UNHANDLED,
            Constructor::IntRange(_) => UNHANDLED,
            NonExhaustive => PatKind::Wild,
            Wildcard => return Pat::wildcard_from_ty(pcx.ty.clone()),
            Opaque => pcx.cx.bug("we should not try to apply an opaque constructor"),
            Missing => pcx.cx.bug(
                "trying to apply the `Missing` constructor;\
                this should have been done in `apply_constructors`",
            ),
        };

        Pat { ty: pcx.ty.clone(), kind: Box::new(pat) }
    }

    /// Returns the number of patterns. This is the same as the arity of the constructor used to
    /// construct `self`.
    pub(super) fn len(&self) -> usize {
        match self {
            Fields::Vec(pats) => pats.len(),
        }
    }

    /// Returns the list of patterns along with the corresponding field indices.
    fn patterns_and_indices(&self) -> SmallVec<[(LocalFieldId, PatId); 2]> {
        match self {
            Fields::Vec(pats) => pats
                .iter()
                .copied()
                .enumerate()
                .map(|(i, p)| (LocalFieldId::from_raw((i as u32).into()), p))
                .collect(),
        }
    }

    pub(super) fn into_patterns(self) -> SmallVec<[PatId; 2]> {
        match self {
            Fields::Vec(pats) => pats,
        }
    }

    /// Overrides some of the fields with the provided patterns. Exactly like
    /// `replace_fields_indexed`, except that it takes `FieldPat`s as input.
    fn replace_with_fieldpats(
        &self,
        new_pats: impl IntoIterator<Item = (LocalFieldId, PatId)>,
    ) -> Self {
        self.replace_fields_indexed(
            new_pats.into_iter().map(|(field, pat)| (u32::from(field.into_raw()) as usize, pat)),
        )
    }

    /// Overrides some of the fields with the provided patterns. This is used when a pattern
    /// defines some fields but not all, for example `Foo { field1: Some(_), .. }`: here we start
    /// with a `Fields` that is just one wildcard per field of the `Foo` struct, and override the
    /// entry corresponding to `field1` with the pattern `Some(_)`. This is also used for slice
    /// patterns for the same reason.
    fn replace_fields_indexed(&self, new_pats: impl IntoIterator<Item = (usize, PatId)>) -> Self {
        let mut fields = self.clone();

        match &mut fields {
            Fields::Vec(pats) => {
                for (i, pat) in new_pats {
                    if let Some(p) = pats.get_mut(i) {
                        *p = pat;
                    }
                }
            }
        }
        fields
    }

    /// Replaces contained fields with the given list of patterns. There must be `len()` patterns
    /// in `pats`.
    pub(super) fn replace_fields(
        &self,
        cx: &MatchCheckCtx<'_>,
        pats: impl IntoIterator<Item = Pat>,
    ) -> Self {
        let pats = pats.into_iter().map(|pat| cx.alloc_pat(pat)).collect();

        match self {
            Fields::Vec(_) => Fields::Vec(pats),
        }
    }

    /// Replaces contained fields with the arguments of the given pattern. Only use on a pattern
    /// that is compatible with the constructor used to build `self`.
    /// This is meant to be used on the result of `Fields::wildcards()`. The idea is that
    /// `wildcards` constructs a list of fields where all entries are wildcards, and the pattern
    /// provided to this function fills some of the fields with non-wildcards.
    /// In the following example `Fields::wildcards` would return `[_, _, _, _]`. If we call
    /// `replace_with_pattern_arguments` on it with the pattern, the result will be `[Some(0), _,
    /// _, _]`.
    /// ```rust
    /// let x: [Option<u8>; 4] = foo();
    /// match x {
    ///     [Some(0), ..] => {}
    /// }
    /// ```
    /// This is guaranteed to preserve the number of patterns in `self`.
    pub(super) fn replace_with_pattern_arguments(
        &self,
        pat: PatId,
        cx: &MatchCheckCtx<'_>,
    ) -> Self {
        // FIXME(iDawer): Factor out pattern deep cloning. See discussion:
        // https://github.com/rust-analyzer/rust-analyzer/pull/8717#discussion_r633086640
        let mut arena = cx.pattern_arena.borrow_mut();
        match arena[pat].kind.as_ref() {
            PatKind::Deref { subpattern } => {
                assert_eq!(self.len(), 1);
                let subpattern = subpattern.clone();
                Fields::from_single_pattern(arena.alloc(subpattern))
            }
            PatKind::Leaf { subpatterns } | PatKind::Variant { subpatterns, .. } => {
                let subpatterns = subpatterns.clone();
                let subpatterns = subpatterns
                    .iter()
                    .map(|field_pat| (field_pat.field, arena.alloc(field_pat.pattern.clone())));
                self.replace_with_fieldpats(subpatterns)
            }

            PatKind::Wild
            | PatKind::Binding { .. }
            | PatKind::LiteralBool { .. }
            | PatKind::Or { .. } => self.clone(),
        }
    }
}

fn is_field_list_non_exhaustive(variant_id: VariantId, cx: &MatchCheckCtx<'_>) -> bool {
    let attr_def_id = match variant_id {
        VariantId::EnumVariantId(id) => id.into(),
        VariantId::StructId(id) => id.into(),
        VariantId::UnionId(id) => id.into(),
    };
    cx.db.attrs(attr_def_id).by_key("non_exhaustive").exists()
}

fn adt_is_box(adt: hir_def::AdtId, cx: &MatchCheckCtx<'_>) -> bool {
    use hir_def::lang_item::LangItemTarget;
    match cx.db.lang_item(cx.module.krate(), "owned_box".into()) {
        Some(LangItemTarget::StructId(box_id)) => adt == box_id.into(),
        _ => false,
    }
}
