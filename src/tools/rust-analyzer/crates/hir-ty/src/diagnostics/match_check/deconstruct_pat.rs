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
    cell::Cell,
    cmp::{max, min},
    iter::once,
    ops::RangeInclusive,
};

use hir_def::{EnumVariantId, HasModule, LocalFieldId, VariantId};
use smallvec::{smallvec, SmallVec};
use stdx::never;

use crate::{
    infer::normalize, inhabitedness::is_enum_variant_uninhabited_from, AdtId, Interner, Scalar, Ty,
    TyExt, TyKind,
};

use super::{
    is_box,
    usefulness::{helper::Captures, MatchCheckCtx, PatCtxt},
    FieldPat, Pat, PatKind,
};

use self::Constructor::*;

/// Recursively expand this pattern into its subpatterns. Only useful for or-patterns.
fn expand_or_pat(pat: &Pat) -> Vec<&Pat> {
    fn expand<'p>(pat: &'p Pat, vec: &mut Vec<&'p Pat>) {
        if let PatKind::Or { pats } = pat.kind.as_ref() {
            for pat in pats {
                expand(pat, vec);
            }
        } else {
            vec.push(pat)
        }
    }

    let mut pats = Vec::new();
    expand(pat, &mut pats);
    pats
}

/// [Constructor] uses this in unimplemented variants.
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
        matches!(
            ty.kind(Interner),
            TyKind::Scalar(Scalar::Char | Scalar::Int(_) | Scalar::Uint(_) | Scalar::Bool)
        )
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
        match scalar_ty {
            Scalar::Bool => IntRange { range: lo..=hi },
            _ => unimplemented!(),
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

    fn to_pat(&self, _cx: &MatchCheckCtx<'_, '_>, ty: Ty) -> Pat {
        match ty.kind(Interner) {
            TyKind::Scalar(Scalar::Bool) => {
                let kind = match self.boundaries() {
                    (0, 0) => PatKind::LiteralBool { value: false },
                    (1, 1) => PatKind::LiteralBool { value: true },
                    (0, 1) => PatKind::Wild,
                    (lo, hi) => {
                        never!("bad range for bool pattern: {}..={}", lo, hi);
                        PatKind::Wild
                    }
                };
                Pat { ty, kind: kind.into() }
            }
            _ => unimplemented!(),
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
    fn arity(self) -> usize {
        match self._unimplemented {}
    }

    /// See `Constructor::is_covered_by`
    fn is_covered_by(self, _other: Self) -> bool {
        match self._unimplemented {}
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
    /// for [`SplitWildcard`]. The carried `bool` is used for the `non_exhaustive_omitted_patterns`
    /// lint.
    Missing { nonexhaustive_enum_missing_real_variants: bool },
    /// Wildcard pattern.
    Wildcard,
    /// Or-pattern.
    Or,
}

impl Constructor {
    pub(super) fn is_wildcard(&self) -> bool {
        matches!(self, Wildcard)
    }

    pub(super) fn is_non_exhaustive(&self) -> bool {
        matches!(self, NonExhaustive)
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

    pub(super) fn is_unstable_variant(&self, _pcx: PatCtxt<'_, '_>) -> bool {
        false //FIXME: implement this
    }

    pub(super) fn is_doc_hidden_variant(&self, _pcx: PatCtxt<'_, '_>) -> bool {
        false //FIXME: implement this
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
            _ => panic!("bad constructor {self:?} for adt {adt:?}"),
        }
    }

    /// The number of fields for this constructor. This must be kept in sync with
    /// `Fields::wildcards`.
    pub(super) fn arity(&self, pcx: PatCtxt<'_, '_>) -> usize {
        match self {
            Single | Variant(_) => match *pcx.ty.kind(Interner) {
                TyKind::Tuple(arity, ..) => arity,
                TyKind::Ref(..) => 1,
                TyKind::Adt(adt, ..) => {
                    if is_box(pcx.cx.db, adt.0) {
                        // The only legal patterns of type `Box` (outside `std`) are `_` and box
                        // patterns. If we're here we can assume this is a box pattern.
                        1
                    } else {
                        let variant = self.variant_id_for_adt(adt.0);
                        Fields::list_variant_nonhidden_fields(pcx.cx, pcx.ty, variant).count()
                    }
                }
                _ => {
                    never!("Unexpected type for `Single` constructor: {:?}", pcx.ty);
                    0
                }
            },
            Slice(slice) => slice.arity(),
            Str(..)
            | FloatRange(..)
            | IntRange(..)
            | NonExhaustive
            | Opaque
            | Missing { .. }
            | Wildcard => 0,
            Or => {
                never!("The `Or` constructor doesn't have a fixed arity");
                0
            }
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
        pcx: PatCtxt<'_, '_>,
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
            Slice(slice) => match slice._unimplemented {},
            // Any other constructor can be used unchanged.
            _ => smallvec![self.clone()],
        }
    }

    /// Returns whether `self` is covered by `other`, i.e. whether `self` is a subset of `other`.
    /// For the simple cases, this is simply checking for equality. For the "grouped" constructors,
    /// this checks for inclusion.
    // We inline because this has a single call site in `Matrix::specialize_constructor`.
    #[inline]
    pub(super) fn is_covered_by(&self, _pcx: PatCtxt<'_, '_>, other: &Self) -> bool {
        // This must be kept in sync with `is_covered_by_any`.
        match (self, other) {
            // Wildcards cover anything
            (_, Wildcard) => true,
            // The missing ctors are not covered by anything in the matrix except wildcards.
            (Missing { .. } | Wildcard, _) => false,

            (Single, Single) => true,
            (Variant(self_id), Variant(other_id)) => self_id == other_id,

            (IntRange(self_range), IntRange(other_range)) => self_range.is_covered_by(other_range),
            (FloatRange(void), FloatRange(..)) => match *void {},
            (Str(void), Str(..)) => match *void {},
            (Slice(self_slice), Slice(other_slice)) => self_slice.is_covered_by(*other_slice),

            // We are trying to inspect an opaque constant. Thus we skip the row.
            (Opaque, _) | (_, Opaque) => false,
            // Only a wildcard pattern can match the special extra constructor.
            (NonExhaustive, _) => false,

            _ => {
                never!("trying to compare incompatible constructors {:?} and {:?}", self, other);
                // Continue with 'whatever is covered' supposed to result in false no-error diagnostic.
                true
            }
        }
    }

    /// Faster version of `is_covered_by` when applied to many constructors. `used_ctors` is
    /// assumed to be built from `matrix.head_ctors()` with wildcards filtered out, and `self` is
    /// assumed to have been split from a wildcard.
    fn is_covered_by_any(&self, _pcx: PatCtxt<'_, '_>, used_ctors: &[Constructor]) -> bool {
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
            Str(..) | FloatRange(..) | Opaque | Missing { .. } | Wildcard | Or => {
                never!("found unexpected ctor in all_ctors: {:?}", self);
                true
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
    pub(super) fn new(pcx: PatCtxt<'_, '_>) -> Self {
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
        let all_ctors = match pcx.ty.kind(Interner) {
            TyKind::Scalar(Scalar::Bool) => smallvec![make_range(0, 1, Scalar::Bool)],
            // TyKind::Array(..) if ... => unhandled(),
            TyKind::Array(..) | TyKind::Slice(..) => unhandled(),
            TyKind::Adt(AdtId(hir_def::AdtId::EnumId(enum_id)), subst) => {
                let enum_data = cx.db.enum_data(*enum_id);

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
                let is_declared_nonexhaustive = cx.is_foreign_non_exhaustive_enum(pcx.ty);

                let is_exhaustive_pat_feature = cx.feature_exhaustive_patterns();

                // If `exhaustive_patterns` is disabled and our scrutinee is an empty enum, we treat it
                // as though it had an "unknown" constructor to avoid exposing its emptiness. The
                // exception is if the pattern is at the top level, because we want empty matches to be
                // considered exhaustive.
                let is_secretly_empty = enum_data.variants.is_empty()
                    && !is_exhaustive_pat_feature
                    && !pcx.is_top_level;

                let mut ctors: SmallVec<[_; 1]> = enum_data
                    .variants
                    .iter()
                    .map(|(local_id, _)| EnumVariantId { parent: *enum_id, local_id })
                    .filter(|&variant| {
                        // If `exhaustive_patterns` is enabled, we exclude variants known to be
                        // uninhabited.
                        let is_uninhabited = is_exhaustive_pat_feature
                            && is_enum_variant_uninhabited_from(variant, subst, cx.module, cx.db);
                        !is_uninhabited
                    })
                    .map(Variant)
                    .collect();

                if is_secretly_empty || is_declared_nonexhaustive {
                    ctors.push(NonExhaustive);
                }
                ctors
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
        pcx: PatCtxt<'_, '_>,
        ctors: impl Iterator<Item = &'a Constructor> + Clone,
    ) {
        // Since `all_ctors` never contains wildcards, this won't recurse further.
        self.all_ctors =
            self.all_ctors.iter().flat_map(|ctor| ctor.split(pcx, ctors.clone())).collect();
        self.matrix_ctors = ctors.filter(|c| !c.is_wildcard()).cloned().collect();
    }

    /// Whether there are any value constructors for this type that are not present in the matrix.
    fn any_missing(&self, pcx: PatCtxt<'_, '_>) -> bool {
        self.iter_missing(pcx).next().is_some()
    }

    /// Iterate over the constructors for this type that are not present in the matrix.
    pub(super) fn iter_missing<'a, 'p>(
        &'a self,
        pcx: PatCtxt<'a, 'p>,
    ) -> impl Iterator<Item = &'a Constructor> + Captures<'p> {
        self.all_ctors.iter().filter(move |ctor| !ctor.is_covered_by_any(pcx, &self.matrix_ctors))
    }

    /// Return the set of constructors resulting from splitting the wildcard. As explained at the
    /// top of the file, if any constructors are missing we can ignore the present ones.
    fn into_ctors(self, pcx: PatCtxt<'_, '_>) -> SmallVec<[Constructor; 1]> {
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
                if pcx.is_non_exhaustive {
                    Missing {
                        nonexhaustive_enum_missing_real_variants: self
                            .iter_missing(pcx)
                            .any(|c| !(c.is_non_exhaustive() || c.is_unstable_variant(pcx))),
                    }
                } else {
                    Missing { nonexhaustive_enum_missing_real_variants: false }
                }
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
///
/// This is constructed for a constructor using [`Fields::wildcards()`]. The idea is that
/// [`Fields::wildcards()`] constructs a list of fields where all entries are wildcards, and then
/// given a pattern we fill some of the fields with its subpatterns.
/// In the following example `Fields::wildcards` returns `[_, _, _, _]`. Then in
/// `extract_pattern_arguments` we fill some of the entries, and the result is
/// `[Some(0), _, _, _]`.
/// ```rust
/// let x: [Option<u8>; 4] = foo();
/// match x {
///     [Some(0), ..] => {}
/// }
/// ```
///
/// Note that the number of fields of a constructor may not match the fields declared in the
/// original struct/variant. This happens if a private or `non_exhaustive` field is uninhabited,
/// because the code mustn't observe that it is uninhabited. In that case that field is not
/// included in `fields`. For that reason, when you have a `mir::Field` you must use
/// `index_with_declared_idx`.
#[derive(Clone, Copy)]
pub(super) struct Fields<'p> {
    fields: &'p [DeconstructedPat<'p>],
}

impl<'p> Fields<'p> {
    fn empty() -> Self {
        Fields { fields: &[] }
    }

    fn singleton(cx: &MatchCheckCtx<'_, 'p>, field: DeconstructedPat<'p>) -> Self {
        let field = cx.pattern_arena.alloc(field);
        Fields { fields: std::slice::from_ref(field) }
    }

    pub(super) fn from_iter(
        cx: &MatchCheckCtx<'_, 'p>,
        fields: impl IntoIterator<Item = DeconstructedPat<'p>>,
    ) -> Self {
        let fields: &[_] = cx.pattern_arena.alloc_extend(fields);
        Fields { fields }
    }

    fn wildcards_from_tys(cx: &MatchCheckCtx<'_, 'p>, tys: impl IntoIterator<Item = Ty>) -> Self {
        Fields::from_iter(cx, tys.into_iter().map(DeconstructedPat::wildcard))
    }

    // In the cases of either a `#[non_exhaustive]` field list or a non-public field, we hide
    // uninhabited fields in order not to reveal the uninhabitedness of the whole variant.
    // This lists the fields we keep along with their types.
    fn list_variant_nonhidden_fields<'a>(
        cx: &'a MatchCheckCtx<'a, 'p>,
        ty: &'a Ty,
        variant: VariantId,
    ) -> impl Iterator<Item = (LocalFieldId, Ty)> + Captures<'a> + Captures<'p> {
        let (adt, substs) = ty.as_adt().unwrap();

        let adt_is_local = variant.module(cx.db.upcast()).krate() == cx.module.krate();
        // Whether we must not match the fields of this variant exhaustively.
        let is_non_exhaustive = is_field_list_non_exhaustive(variant, cx) && !adt_is_local;

        let visibility = cx.db.field_visibilities(variant);
        let field_ty = cx.db.field_types(variant);
        let fields_len = variant.variant_data(cx.db.upcast()).fields().len() as u32;

        (0..fields_len).map(|idx| LocalFieldId::from_raw(idx.into())).filter_map(move |fid| {
            let ty = field_ty[fid].clone().substitute(Interner, substs);
            let ty = normalize(cx.db, cx.db.trait_environment_for_body(cx.body), ty);
            let is_visible = matches!(adt, hir_def::AdtId::EnumId(..))
                || visibility[fid].is_visible_from(cx.db.upcast(), cx.module);
            let is_uninhabited = cx.is_uninhabited(&ty);

            if is_uninhabited && (!is_visible || is_non_exhaustive) {
                None
            } else {
                Some((fid, ty))
            }
        })
    }

    /// Creates a new list of wildcard fields for a given constructor. The result must have a
    /// length of `constructor.arity()`.
    pub(crate) fn wildcards(
        cx: &MatchCheckCtx<'_, 'p>,
        ty: &Ty,
        constructor: &Constructor,
    ) -> Self {
        let ret = match constructor {
            Single | Variant(_) => match ty.kind(Interner) {
                TyKind::Tuple(_, substs) => {
                    let tys = substs.iter(Interner).map(|ty| ty.assert_ty_ref(Interner));
                    Fields::wildcards_from_tys(cx, tys.cloned())
                }
                TyKind::Ref(.., rty) => Fields::wildcards_from_tys(cx, once(rty.clone())),
                &TyKind::Adt(AdtId(adt), ref substs) => {
                    if is_box(cx.db, adt) {
                        // The only legal patterns of type `Box` (outside `std`) are `_` and box
                        // patterns. If we're here we can assume this is a box pattern.
                        let subst_ty = substs.at(Interner, 0).assert_ty_ref(Interner).clone();
                        Fields::wildcards_from_tys(cx, once(subst_ty))
                    } else {
                        let variant = constructor.variant_id_for_adt(adt);
                        let tys = Fields::list_variant_nonhidden_fields(cx, ty, variant)
                            .map(|(_, ty)| ty);
                        Fields::wildcards_from_tys(cx, tys)
                    }
                }
                ty_kind => {
                    never!("Unexpected type for `Single` constructor: {:?}", ty_kind);
                    Fields::wildcards_from_tys(cx, once(ty.clone()))
                }
            },
            Slice(slice) => match slice._unimplemented {},
            Str(..)
            | FloatRange(..)
            | IntRange(..)
            | NonExhaustive
            | Opaque
            | Missing { .. }
            | Wildcard => Fields::empty(),
            Or => {
                never!("called `Fields::wildcards` on an `Or` ctor");
                Fields::empty()
            }
        };
        ret
    }

    /// Returns the list of patterns.
    pub(super) fn iter_patterns<'a>(
        &'a self,
    ) -> impl Iterator<Item = &'p DeconstructedPat<'p>> + Captures<'a> {
        self.fields.iter()
    }
}

/// Values and patterns can be represented as a constructor applied to some fields. This represents
/// a pattern in this form.
/// This also keeps track of whether the pattern has been found reachable during analysis. For this
/// reason we should be careful not to clone patterns for which we care about that. Use
/// `clone_and_forget_reachability` if you're sure.
pub(crate) struct DeconstructedPat<'p> {
    ctor: Constructor,
    fields: Fields<'p>,
    ty: Ty,
    reachable: Cell<bool>,
}

impl<'p> DeconstructedPat<'p> {
    pub(super) fn wildcard(ty: Ty) -> Self {
        Self::new(Wildcard, Fields::empty(), ty)
    }

    pub(super) fn new(ctor: Constructor, fields: Fields<'p>, ty: Ty) -> Self {
        DeconstructedPat { ctor, fields, ty, reachable: Cell::new(false) }
    }

    /// Construct a pattern that matches everything that starts with this constructor.
    /// For example, if `ctor` is a `Constructor::Variant` for `Option::Some`, we get the pattern
    /// `Some(_)`.
    pub(super) fn wild_from_ctor(pcx: PatCtxt<'_, 'p>, ctor: Constructor) -> Self {
        let fields = Fields::wildcards(pcx.cx, pcx.ty, &ctor);
        DeconstructedPat::new(ctor, fields, pcx.ty.clone())
    }

    /// Clone this value. This method emphasizes that cloning loses reachability information and
    /// should be done carefully.
    pub(super) fn clone_and_forget_reachability(&self) -> Self {
        DeconstructedPat::new(self.ctor.clone(), self.fields, self.ty.clone())
    }

    pub(crate) fn from_pat(cx: &MatchCheckCtx<'_, 'p>, pat: &Pat) -> Self {
        let mkpat = |pat| DeconstructedPat::from_pat(cx, pat);
        let ctor;
        let fields;
        match pat.kind.as_ref() {
            PatKind::Binding { subpattern: Some(subpat), .. } => return mkpat(subpat),
            PatKind::Binding { subpattern: None, .. } | PatKind::Wild => {
                ctor = Wildcard;
                fields = Fields::empty();
            }
            PatKind::Deref { subpattern } => {
                ctor = Single;
                fields = Fields::singleton(cx, mkpat(subpattern));
            }
            PatKind::Leaf { subpatterns } | PatKind::Variant { subpatterns, .. } => {
                match pat.ty.kind(Interner) {
                    TyKind::Tuple(_, substs) => {
                        ctor = Single;
                        let mut wilds: SmallVec<[_; 2]> = substs
                            .iter(Interner)
                            .map(|arg| arg.assert_ty_ref(Interner).clone())
                            .map(DeconstructedPat::wildcard)
                            .collect();
                        for pat in subpatterns {
                            let idx: u32 = pat.field.into_raw().into();
                            wilds[idx as usize] = mkpat(&pat.pattern);
                        }
                        fields = Fields::from_iter(cx, wilds)
                    }
                    TyKind::Adt(adt, substs) if is_box(cx.db, adt.0) => {
                        // The only legal patterns of type `Box` (outside `std`) are `_` and box
                        // patterns. If we're here we can assume this is a box pattern.
                        // FIXME(Nadrieril): A `Box` can in theory be matched either with `Box(_,
                        // _)` or a box pattern. As a hack to avoid an ICE with the former, we
                        // ignore other fields than the first one. This will trigger an error later
                        // anyway.
                        // See https://github.com/rust-lang/rust/issues/82772 ,
                        // explanation: https://github.com/rust-lang/rust/pull/82789#issuecomment-796921977
                        // The problem is that we can't know from the type whether we'll match
                        // normally or through box-patterns. We'll have to figure out a proper
                        // solution when we introduce generalized deref patterns. Also need to
                        // prevent mixing of those two options.
                        let pat =
                            subpatterns.iter().find(|pat| pat.field.into_raw() == 0u32.into());
                        let field = if let Some(pat) = pat {
                            mkpat(&pat.pattern)
                        } else {
                            let ty = substs.at(Interner, 0).assert_ty_ref(Interner).clone();
                            DeconstructedPat::wildcard(ty)
                        };
                        ctor = Single;
                        fields = Fields::singleton(cx, field)
                    }
                    &TyKind::Adt(adt, _) => {
                        ctor = match pat.kind.as_ref() {
                            PatKind::Leaf { .. } => Single,
                            PatKind::Variant { enum_variant, .. } => Variant(*enum_variant),
                            _ => {
                                never!();
                                Wildcard
                            }
                        };
                        let variant = ctor.variant_id_for_adt(adt.0);
                        let fields_len = variant.variant_data(cx.db.upcast()).fields().len();
                        // For each field in the variant, we store the relevant index into `self.fields` if any.
                        let mut field_id_to_id: Vec<Option<usize>> = vec![None; fields_len];
                        let tys = Fields::list_variant_nonhidden_fields(cx, &pat.ty, variant)
                            .enumerate()
                            .map(|(i, (fid, ty))| {
                                let field_idx: u32 = fid.into_raw().into();
                                field_id_to_id[field_idx as usize] = Some(i);
                                ty
                            });
                        let mut wilds: SmallVec<[_; 2]> =
                            tys.map(DeconstructedPat::wildcard).collect();
                        for pat in subpatterns {
                            let field_idx: u32 = pat.field.into_raw().into();
                            if let Some(i) = field_id_to_id[field_idx as usize] {
                                wilds[i] = mkpat(&pat.pattern);
                            }
                        }
                        fields = Fields::from_iter(cx, wilds);
                    }
                    _ => {
                        never!("pattern has unexpected type: pat: {:?}, ty: {:?}", pat, &pat.ty);
                        ctor = Wildcard;
                        fields = Fields::empty();
                    }
                }
            }
            &PatKind::LiteralBool { value } => {
                ctor = IntRange(IntRange::from_bool(value));
                fields = Fields::empty();
            }
            PatKind::Or { .. } => {
                ctor = Or;
                let pats: SmallVec<[_; 2]> = expand_or_pat(pat).into_iter().map(mkpat).collect();
                fields = Fields::from_iter(cx, pats)
            }
        }
        DeconstructedPat::new(ctor, fields, pat.ty.clone())
    }

    pub(crate) fn to_pat(&self, cx: &MatchCheckCtx<'_, 'p>) -> Pat {
        let mut subpatterns = self.iter_fields().map(|p| p.to_pat(cx));
        let pat = match &self.ctor {
            Single | Variant(_) => match self.ty.kind(Interner) {
                TyKind::Tuple(..) => PatKind::Leaf {
                    subpatterns: subpatterns
                        .zip(0u32..)
                        .map(|(p, i)| FieldPat {
                            field: LocalFieldId::from_raw(i.into()),
                            pattern: p,
                        })
                        .collect(),
                },
                TyKind::Adt(adt, _) if is_box(cx.db, adt.0) => {
                    // Without `box_patterns`, the only legal pattern of type `Box` is `_` (outside
                    // of `std`). So this branch is only reachable when the feature is enabled and
                    // the pattern is a box pattern.
                    PatKind::Deref { subpattern: subpatterns.next().unwrap() }
                }
                TyKind::Adt(adt, substs) => {
                    let variant = self.ctor.variant_id_for_adt(adt.0);
                    let subpatterns = Fields::list_variant_nonhidden_fields(cx, self.ty(), variant)
                        .zip(subpatterns)
                        .map(|((field, _ty), pattern)| FieldPat { field, pattern })
                        .collect();

                    if let VariantId::EnumVariantId(enum_variant) = variant {
                        PatKind::Variant { substs: substs.clone(), enum_variant, subpatterns }
                    } else {
                        PatKind::Leaf { subpatterns }
                    }
                }
                // Note: given the expansion of `&str` patterns done in `expand_pattern`, we should
                // be careful to reconstruct the correct constant pattern here. However a string
                // literal pattern will never be reported as a non-exhaustiveness witness, so we
                // ignore this issue.
                TyKind::Ref(..) => PatKind::Deref { subpattern: subpatterns.next().unwrap() },
                _ => {
                    never!("unexpected ctor for type {:?} {:?}", self.ctor, self.ty);
                    PatKind::Wild
                }
            },
            &Slice(slice) => match slice._unimplemented {},
            &Str(void) => match void {},
            &FloatRange(void) => match void {},
            IntRange(range) => return range.to_pat(cx, self.ty.clone()),
            Wildcard | NonExhaustive => PatKind::Wild,
            Missing { .. } => {
                never!(
                    "trying to convert a `Missing` constructor into a `Pat`; this is a bug, \
                    `Missing` should have been processed in `apply_constructors`"
                );
                PatKind::Wild
            }
            Opaque | Or => {
                never!("can't convert to pattern: {:?}", self.ctor);
                PatKind::Wild
            }
        };
        Pat { ty: self.ty.clone(), kind: Box::new(pat) }
    }

    pub(super) fn is_or_pat(&self) -> bool {
        matches!(self.ctor, Or)
    }

    pub(super) fn ctor(&self) -> &Constructor {
        &self.ctor
    }

    pub(super) fn ty(&self) -> &Ty {
        &self.ty
    }

    pub(super) fn iter_fields<'a>(&'a self) -> impl Iterator<Item = &'p DeconstructedPat<'p>> + 'a {
        self.fields.iter_patterns()
    }

    /// Specialize this pattern with a constructor.
    /// `other_ctor` can be different from `self.ctor`, but must be covered by it.
    pub(super) fn specialize<'a>(
        &'a self,
        cx: &MatchCheckCtx<'_, 'p>,
        other_ctor: &Constructor,
    ) -> SmallVec<[&'p DeconstructedPat<'p>; 2]> {
        match (&self.ctor, other_ctor) {
            (Wildcard, _) => {
                // We return a wildcard for each field of `other_ctor`.
                Fields::wildcards(cx, &self.ty, other_ctor).iter_patterns().collect()
            }
            (Slice(self_slice), Slice(other_slice))
                if self_slice.arity() != other_slice.arity() =>
            {
                match self_slice._unimplemented {}
            }
            _ => self.fields.iter_patterns().collect(),
        }
    }

    /// We keep track for each pattern if it was ever reachable during the analysis. This is used
    /// with `unreachable_spans` to report unreachable subpatterns arising from or patterns.
    pub(super) fn set_reachable(&self) {
        self.reachable.set(true)
    }
    pub(super) fn is_reachable(&self) -> bool {
        self.reachable.get()
    }
}

fn is_field_list_non_exhaustive(variant_id: VariantId, cx: &MatchCheckCtx<'_, '_>) -> bool {
    let attr_def_id = match variant_id {
        VariantId::EnumVariantId(id) => id.into(),
        VariantId::StructId(id) => id.into(),
        VariantId::UnionId(id) => id.into(),
    };
    cx.db.attrs(attr_def_id).by_key("non_exhaustive").exists()
}
