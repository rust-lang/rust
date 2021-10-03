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
//! wildcards, see [`SplitWildcard`]; for integer ranges, see [`SplitIntRange`]; for slices, see
//! [`SplitVarLenSlice`].

use self::Constructor::*;
use self::SliceKind::*;

use super::usefulness::{MatchCheckCtxt, PatCtxt};

use rustc_apfloat::ieee::{DoubleS, IeeeFloat, SingleS};
use rustc_data_structures::captures::Captures;

use rustc_hir::{HirId, RangeEnd};
use rustc_middle::ty::subst::GenericArg;
use rustc_middle::ty::Ty;
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::VariantIdx;

use smallvec::{smallvec, SmallVec};
use std::cell::Cell;
use std::cmp::{self, max, min, Ordering};
use std::fmt;
use std::iter::{once, IntoIterator};
use std::ops::RangeInclusive;

/// An inclusive interval, used for precise integer exhaustiveness checking.
/// `IntRange`s always store a contiguous range. This means that values are
/// encoded such that `0` encodes the minimum value for the integer,
/// regardless of the signedness.
/// For example, the pattern `-128..=127i8` is encoded as `0..=255`.
/// This makes comparisons and arithmetic on interval endpoints much more
/// straightforward. See `signed_bias` for details.
/// Only use for chars, ints and uints.
///
/// `IntRange` is never used to encode an empty range or a "range" that wraps
/// around the (offset) space: i.e., `range.lo <= range.hi`.
#[derive(Clone, PartialEq, Eq)]
pub(super) struct IntRange {
    range: RangeInclusive<u128>,
    /// Keeps the bias used for encoding the range. It depends on the type of the range and
    /// possibly the pointer size of the current architecture. The algorithm ensures we never
    /// compare `IntRange`s with different types/architectures.
    bias: u128,
}

impl IntRange {
    fn is_singleton(&self) -> bool {
        self.range.start() == self.range.end()
    }

    fn boundaries(&self) -> (u128, u128) {
        (*self.range.start(), *self.range.end())
    }

    #[inline]
    pub(super) fn from_bits<'tcx>(
        ty: Ty<'tcx>,
        ty_size: rustc_target::abi::Size,
        lo: u128,
        hi: u128,
        end: &RangeEnd,
    ) -> IntRange {
        let bias = if MatchCheckCtxt::is_signed_int(ty) {
            1u128 << (ty_size.bits() as u128 - 1)
        } else {
            0
        };
        // Perform a shift if the underlying types are signed,
        // which makes the interval arithmetic simpler.
        let (lo, hi) = (lo ^ bias, hi ^ bias);
        let offset = (*end == RangeEnd::Excluded) as u128;
        if lo > hi || (lo == hi && *end == RangeEnd::Excluded) {
            // This should have been caught earlier by E0030.
            bug!("malformed range pattern: {}..={}", lo, (hi - offset));
        }
        IntRange { range: lo..=(hi - offset), bias }
    }

    /// The reverse of `Self::from_bits`.
    pub(super) fn to_bits(&self) -> (u128, u128, RangeEnd) {
        let (lo, hi) = self.boundaries();
        let bias = self.bias;
        (lo ^ bias, hi ^ bias, RangeEnd::Included)
    }

    fn is_subrange(&self, other: &Self) -> bool {
        other.range.start() <= self.range.start() && self.range.end() <= other.range.end()
    }

    fn intersection(&self, other: &Self) -> Option<Self> {
        let (lo, hi) = self.boundaries();
        let (other_lo, other_hi) = other.boundaries();
        if lo <= other_hi && other_lo <= hi {
            Some(IntRange { range: max(lo, other_lo)..=min(hi, other_hi), bias: self.bias })
        } else {
            None
        }
    }

    /// Check whether two non-singleton ranges touch exactly on on of their endpoints.
    fn suspicious_intersection(&self, other: &Self) -> Option<Self> {
        if self.is_singleton() || other.is_singleton() {
            return None;
        }
        // `false` in the following cases:
        // 1     ----      // 1  ----------   // 1 ----        // 1       ----
        // 2  ----------   // 2     ----      // 2       ----  // 2 ----
        //
        // The following are currently `false`, but could be `true` in the future (#64007):
        // 1 ---------       // 1     ---------
        // 2     ----------  // 2 ----------
        //
        // `true` in the following cases:
        // 1 -------          // 1       -------
        // 2       --------   // 2 -------
        let (lo, hi) = self.boundaries();
        let (other_lo, other_hi) = other.boundaries();
        if lo == other_hi || hi == other_lo {
            Some(self.intersection(other).unwrap())
        } else {
            None
        }
    }

    /// Lint on likely incorrect range patterns (#63987)
    pub(super) fn lint_overlapping_range_endpoints<'a, 'p: 'a, 'tcx: 'a>(
        &self,
        pcx: PatCtxt<'_, 'p, 'tcx>,
        report: &mut Vec<(Span, HirId, Vec<DeconstructedPat<'p, 'tcx>>)>,
        pats: impl Iterator<Item = &'a DeconstructedPat<'p, 'tcx>>,
        column_count: usize,
        hir_id: HirId,
    ) {
        if self.is_singleton() {
            return;
        }

        if column_count != 1 {
            // FIXME: for now, only check for overlapping ranges on simple range
            // patterns. Otherwise with the current logic the following is detected
            // as overlapping:
            // ```
            // match (0u8, true) {
            //   (0 ..= 125, false) => {}
            //   (125 ..= 255, true) => {}
            //   _ => {}
            // }
            // ```
            return;
        }

        let mut overlaps = Vec::new();
        for pat in pats {
            if let IntRange(range) = pat.ctor() {
                if let Some(intersection) = self.suspicious_intersection(range) {
                    let intersection = DeconstructedPat::new(
                        IntRange(intersection),
                        Fields::empty(),
                        pcx.ty,
                        pat.span(),
                    );
                    overlaps.push(intersection);
                }
            }
        }
        if !overlaps.is_empty() {
            report.push((pcx.span, hir_id, overlaps));
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

/// Note: this is missing some information. It will render chars as numbers among other things. To
/// render properly, convert to a pattern first.
impl fmt::Debug for IntRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (lo, hi) = self.boundaries();
        let bias = self.bias;
        let (lo, hi) = (lo ^ bias, hi ^ bias);
        write!(f, "{}", lo)?;
        write!(f, "{}", RangeEnd::Included)?;
        write!(f, "{}", hi)
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
    fn iter<'a>(&'a self) -> impl Iterator<Item = IntRange> + Captures<'a> {
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
            .map(move |(prev_border, border)| {
                let range = match (prev_border, border) {
                    (JustBefore(n), JustBefore(m)) if n < m => n..=(m - 1),
                    (JustBefore(n), AfterMax) => n..=u128::MAX,
                    _ => unreachable!(), // Ruled out by the sorting and filtering we did
                };
                IntRange { range, bias: self.range.bias }
            })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum SliceKind {
    /// Patterns of length `n` (`[x, y]`).
    FixedLen(usize),
    /// Patterns using the `..` notation (`[x, .., y]`).
    /// Captures any array constructor of `length >= i + j`.
    /// In the case where `array_len` is `Some(_)`,
    /// this indicates that we only care about the first `i` and the last `j` values of the array,
    /// and everything in between is a wildcard `_`.
    VarLen(usize, usize),
}

impl SliceKind {
    fn arity(self) -> usize {
        match self {
            FixedLen(length) => length,
            VarLen(prefix, suffix) => prefix + suffix,
        }
    }

    /// Whether this pattern includes patterns of length `other_len`.
    fn covers_length(self, other_len: usize) -> bool {
        match self {
            FixedLen(len) => len == other_len,
            VarLen(prefix, suffix) => prefix + suffix <= other_len,
        }
    }
}

/// A constructor for array and slice patterns.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Slice {
    /// `None` if the matched value is a slice, `Some(n)` if it is an array of size `n`.
    pub(super) array_len: Option<usize>,
    /// The kind of pattern it is: fixed-length `[x, y]` or variable length `[x, .., y]`.
    pub(super) kind: SliceKind,
}

impl Slice {
    pub(super) fn new(array_len: Option<usize>, kind: SliceKind) -> Self {
        let kind = match (array_len, kind) {
            // If the middle `..` is empty, we effectively have a fixed-length pattern.
            (Some(len), VarLen(prefix, suffix)) if prefix + suffix >= len => FixedLen(len),
            _ => kind,
        };
        Slice { array_len, kind }
    }

    fn arity(self) -> usize {
        self.kind.arity()
    }

    /// See `Constructor::is_covered_by`
    fn is_covered_by(self, other: Self) -> bool {
        other.kind.covers_length(self.arity())
    }
}

/// This computes constructor splitting for variable-length slices, as explained at the top of the
/// file.
///
/// A slice pattern `[x, .., y]` behaves like the infinite or-pattern `[x, y] | [x, _, y] | [x, _,
/// _, y] | ...`. The corresponding value constructors are fixed-length array constructors above a
/// given minimum length. We obviously can't list this infinitude of constructors. Thankfully,
/// it turns out that for each finite set of slice patterns, all sufficiently large array lengths
/// are equivalent.
///
/// Let's look at an example, where we are trying to split the last pattern:
/// ```
/// match x {
///     [true, true, ..] => {}
///     [.., false, false] => {}
///     [..] => {}
/// }
/// ```
/// Here are the results of specialization for the first few lengths:
/// ```
/// // length 0
/// [] => {}
/// // length 1
/// [_] => {}
/// // length 2
/// [true, true] => {}
/// [false, false] => {}
/// [_, _] => {}
/// // length 3
/// [true, true,  _    ] => {}
/// [_,    false, false] => {}
/// [_,    _,     _    ] => {}
/// // length 4
/// [true, true, _,     _    ] => {}
/// [_,    _,    false, false] => {}
/// [_,    _,    _,     _    ] => {}
/// // length 5
/// [true, true, _, _,     _    ] => {}
/// [_,    _,    _, false, false] => {}
/// [_,    _,    _, _,     _    ] => {}
/// ```
///
/// If we went above length 5, we would simply be inserting more columns full of wildcards in the
/// middle. This means that the set of witnesses for length `l >= 5` if equivalent to the set for
/// any other `l' >= 5`: simply add or remove wildcards in the middle to convert between them.
///
/// This applies to any set of slice patterns: there will be a length `L` above which all lengths
/// behave the same. This is exactly what we need for constructor splitting. Therefore a
/// variable-length slice can be split into a variable-length slice of minimal length `L`, and many
/// fixed-length slices of lengths `< L`.
///
/// For each variable-length pattern `p` with a prefix of length `plₚ` and suffix of length `slₚ`,
/// only the first `plₚ` and the last `slₚ` elements are examined. Therefore, as long as `L` is
/// positive (to avoid concerns about empty types), all elements after the maximum prefix length
/// and before the maximum suffix length are not examined by any variable-length pattern, and
/// therefore can be added/removed without affecting them - creating equivalent patterns from any
/// sufficiently-large length.
///
/// Of course, if fixed-length patterns exist, we must be sure that our length is large enough to
/// miss them all, so we can pick `L = max(max(FIXED_LEN)+1, max(PREFIX_LEN) + max(SUFFIX_LEN))`
///
/// `max_slice` below will be made to have arity `L`.
#[derive(Debug)]
struct SplitVarLenSlice {
    /// If the type is an array, this is its size.
    array_len: Option<usize>,
    /// The arity of the input slice.
    arity: usize,
    /// The smallest slice bigger than any slice seen. `max_slice.arity()` is the length `L`
    /// described above.
    max_slice: SliceKind,
}

impl SplitVarLenSlice {
    fn new(prefix: usize, suffix: usize, array_len: Option<usize>) -> Self {
        SplitVarLenSlice { array_len, arity: prefix + suffix, max_slice: VarLen(prefix, suffix) }
    }

    /// Pass a set of slices relative to which to split this one.
    fn split(&mut self, slices: impl Iterator<Item = SliceKind>) {
        let (max_prefix_len, max_suffix_len) = match &mut self.max_slice {
            VarLen(prefix, suffix) => (prefix, suffix),
            FixedLen(_) => return, // No need to split
        };
        // We grow `self.max_slice` to be larger than all slices encountered, as described above.
        // For diagnostics, we keep the prefix and suffix lengths separate, but grow them so that
        // `L = max_prefix_len + max_suffix_len`.
        let mut max_fixed_len = 0;
        for slice in slices {
            match slice {
                FixedLen(len) => {
                    max_fixed_len = cmp::max(max_fixed_len, len);
                }
                VarLen(prefix, suffix) => {
                    *max_prefix_len = cmp::max(*max_prefix_len, prefix);
                    *max_suffix_len = cmp::max(*max_suffix_len, suffix);
                }
            }
        }
        // We want `L = max(L, max_fixed_len + 1)`, modulo the fact that we keep prefix and
        // suffix separate.
        if max_fixed_len + 1 >= *max_prefix_len + *max_suffix_len {
            // The subtraction can't overflow thanks to the above check.
            // The new `max_prefix_len` is larger than its previous value.
            *max_prefix_len = max_fixed_len + 1 - *max_suffix_len;
        }

        // We cap the arity of `max_slice` at the array size.
        match self.array_len {
            Some(len) if self.max_slice.arity() >= len => self.max_slice = FixedLen(len),
            _ => {}
        }
    }

    /// Iterate over the partition of this slice.
    fn iter<'a>(&'a self) -> impl Iterator<Item = Slice> + Captures<'a> {
        let smaller_lengths = match self.array_len {
            // The only admissible fixed-length slice is one of the array size. Whether `max_slice`
            // is fixed-length or variable-length, it will be the only relevant slice to output
            // here.
            Some(_) => (0..0), // empty range
            // We cover all arities in the range `(self.arity..infinity)`. We split that range into
            // two: lengths smaller than `max_slice.arity()` are treated independently as
            // fixed-lengths slices, and lengths above are captured by `max_slice`.
            None => self.arity..self.max_slice.arity(),
        };
        smaller_lengths
            .map(FixedLen)
            .chain(once(self.max_slice))
            .map(move |kind| Slice::new(self.array_len, kind))
    }
}

/// A value can be decomposed into a constructor applied to some fields. This struct represents
/// the constructor. See also `Fields`.
///
/// `pat_constructor` retrieves the constructor corresponding to a pattern.
/// `specialize_constructor` returns the list of fields corresponding to a pattern, given a
/// constructor. `Constructor::apply` reconstructs the pattern from a pair of `Constructor` and
/// `Fields`.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Constructor<'tcx> {
    /// Structs.
    Single,
    /// Enum variants.
    Variant(VariantIdx),
    /// Tuple patterns.
    Tuple(&'tcx [GenericArg<'tcx>]),
    /// Ref patterns (`&_` and `&mut _`).
    Ref(Ty<'tcx>),
    /// Box patterns.
    BoxPat(Ty<'tcx>),
    /// Booleans.
    Bool(bool),
    /// Ranges of integer literal values (`2`, `2..=5` or `2..5`).
    IntRange(IntRange),
    /// Ranges of floating-point literal values (`2.0..=5.2`).
    F32Range(IeeeFloat<SingleS>, IeeeFloat<SingleS>, RangeEnd),
    F64Range(IeeeFloat<DoubleS>, IeeeFloat<DoubleS>, RangeEnd),
    /// String literals. Strings are not quite the same as `&[u8]` so we treat them separately.
    Str(&'tcx [u8]),
    /// Array and slice patterns.
    Slice(Slice, Ty<'tcx>),
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
    Missing {
        nonexhaustive_enum_missing_real_variants: bool,
    },
    /// Wildcard pattern.
    Wildcard,
    /// Or-pattern.
    Or,
}

impl<'tcx> Constructor<'tcx> {
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
            Slice(slice, _) => Some(*slice),
            _ => None,
        }
    }

    /// The number of fields for this constructor. This must be kept in sync with
    /// `Fields::wildcards`.
    pub(super) fn arity(&self, pcx: PatCtxt<'_, '_, 'tcx>) -> usize {
        match self {
            Single | Variant(_) => pcx.cx.list_variant_nonhidden_fields(pcx.ty, self).count(),
            Tuple(fs) => fs.len(),
            Ref(_) | BoxPat(_) => 1,
            Slice(slice, _) => slice.arity(),
            Str(..)
            | Bool(..)
            | IntRange(..)
            | F32Range(..)
            | F64Range(..)
            | NonExhaustive
            | Opaque
            | Missing { .. }
            | Wildcard => 0,
            Or => bug!("The `Or` constructor doesn't have a fixed arity"),
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
        pcx: PatCtxt<'_, '_, 'tcx>,
        ctors: impl Iterator<Item = &'a Constructor<'tcx>> + Clone,
    ) -> SmallVec<[Self; 1]>
    where
        'tcx: 'a,
    {
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
            &Slice(Slice { kind: VarLen(self_prefix, self_suffix), array_len }, ty) => {
                let mut split_self = SplitVarLenSlice::new(self_prefix, self_suffix, array_len);
                let slices = ctors.filter_map(|c| c.as_slice()).map(|s| s.kind);
                split_self.split(slices);
                split_self.iter().map(|s| Slice(s, ty)).collect()
            }
            // Any other constructor can be used unchanged.
            _ => smallvec![self.clone()],
        }
    }

    /// Returns whether `self` is covered by `other`, i.e. whether `self` is a subset of `other`.
    /// For the simple cases, this is simply checking for equality. For the "grouped" constructors,
    /// this checks for inclusion.
    // We inline because this has a single call site in `Matrix::specialize_constructor`.
    #[inline]
    pub(super) fn is_covered_by<'p>(&self, pcx: PatCtxt<'_, 'p, 'tcx>, other: &Self) -> bool {
        // This must be kept in sync with `is_covered_by_any`.
        match (self, other) {
            // Wildcards cover anything
            (_, Wildcard) => true,
            // The missing ctors are not covered by anything in the matrix except wildcards.
            (Missing { .. } | Wildcard, _) => false,

            (Single, Single) => true,
            (Tuple(_), Tuple(_)) => true,
            (Ref(_), Ref(_)) => true,
            (BoxPat(_), BoxPat(_)) => true,
            (Variant(self_id), Variant(other_id)) => self_id == other_id,
            (Bool(self_b), Bool(other_b)) => self_b == other_b,

            (IntRange(self_range), IntRange(other_range)) => self_range.is_covered_by(other_range),
            (F32Range(self_from, self_to, self_end), F32Range(other_from, other_to, other_end)) => {
                match (self_to.partial_cmp(other_to), self_from.partial_cmp(other_from)) {
                    (Some(to), Some(from)) => {
                        (from == Ordering::Greater || from == Ordering::Equal)
                            && (to == Ordering::Less
                                || (other_end == self_end && to == Ordering::Equal))
                    }
                    _ => false,
                }
            }
            (F64Range(self_from, self_to, self_end), F64Range(other_from, other_to, other_end)) => {
                match (self_to.partial_cmp(other_to), self_from.partial_cmp(other_from)) {
                    (Some(to), Some(from)) => {
                        (from == Ordering::Greater || from == Ordering::Equal)
                            && (to == Ordering::Less
                                || (other_end == self_end && to == Ordering::Equal))
                    }
                    _ => false,
                }
            }
            (Str(self_val), Str(other_val)) => self_val == other_val,
            (Slice(self_slice, _), Slice(other_slice, _)) => self_slice.is_covered_by(*other_slice),

            // We are trying to inspect an opaque constant. Thus we skip the row.
            (Opaque, _) | (_, Opaque) => false,
            // Only a wildcard pattern can match the special extra constructor.
            (NonExhaustive, _) => false,

            _ => span_bug!(
                pcx.span,
                "trying to compare incompatible constructors {:?} and {:?}",
                self,
                other
            ),
        }
    }

    /// Faster version of `is_covered_by` when applied to many constructors. `used_ctors` is
    /// assumed to be built from `matrix.head_ctors()` with wildcards filtered out, and `self` is
    /// assumed to have been split from a wildcard.
    fn is_covered_by_any<'p>(
        &self,
        pcx: PatCtxt<'_, 'p, 'tcx>,
        used_ctors: &[Constructor<'tcx>],
    ) -> bool {
        if used_ctors.is_empty() {
            return false;
        }

        // This must be kept in sync with `is_covered_by`.
        match self {
            // If `self` is one of these the type has exactly one constructor so the check is
            // simpler.
            Single | Tuple(_) | Ref(_) | BoxPat(_) => !used_ctors.is_empty(),
            Variant(vid) => used_ctors.iter().any(|c| matches!(c, Variant(i) if i == vid)),
            Bool(self_b) => {
                used_ctors.iter().any(|c| matches!(c, Bool(other_b) if self_b == other_b))
            }
            IntRange(range) => used_ctors
                .iter()
                .filter_map(|c| c.as_int_range())
                .any(|other| range.is_covered_by(other)),
            Slice(slice, _) => used_ctors
                .iter()
                .filter_map(|c| c.as_slice())
                .any(|other| slice.is_covered_by(other)),
            // This constructor is never covered by anything else
            NonExhaustive => false,
            Str(..) | F32Range(..) | F64Range(..) | Opaque | Missing { .. } | Wildcard | Or => {
                span_bug!(pcx.span, "found unexpected ctor in all_ctors: {:?}", self)
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
pub(super) struct SplitWildcard<'tcx> {
    /// Constructors seen in the matrix.
    matrix_ctors: Vec<Constructor<'tcx>>,
    /// All the constructors for this type
    all_ctors: SmallVec<[Constructor<'tcx>; 1]>,
}

impl<'tcx> SplitWildcard<'tcx> {
    pub(super) fn new<'p>(pcx: PatCtxt<'_, 'p, 'tcx>) -> Self {
        let all_ctors = pcx.cx.list_constructors_for_type(pcx.ty, pcx.is_top_level);
        SplitWildcard { matrix_ctors: Vec::new(), all_ctors }
    }

    /// Pass a set of constructors relative to which to split this one. Don't call twice, it won't
    /// do what you want.
    pub(super) fn split<'a>(
        &mut self,
        pcx: PatCtxt<'_, '_, 'tcx>,
        ctors: impl Iterator<Item = &'a Constructor<'tcx>> + Clone,
    ) where
        'tcx: 'a,
    {
        // Since `all_ctors` never contains wildcards, this won't recurse further.
        self.all_ctors =
            self.all_ctors.iter().flat_map(|ctor| ctor.split(pcx, ctors.clone())).collect();
        self.matrix_ctors = ctors.filter(|c| !c.is_wildcard()).cloned().collect();
    }

    /// Whether there are any value constructors for this type that are not present in the matrix.
    fn any_missing(&self, pcx: PatCtxt<'_, '_, 'tcx>) -> bool {
        self.iter_missing(pcx).next().is_some()
    }

    /// Iterate over the constructors for this type that are not present in the matrix.
    pub(super) fn iter_missing<'a, 'p>(
        &'a self,
        pcx: PatCtxt<'a, 'p, 'tcx>,
    ) -> impl Iterator<Item = &'a Constructor<'tcx>> + Captures<'p> {
        self.all_ctors.iter().filter(move |ctor| !ctor.is_covered_by_any(pcx, &self.matrix_ctors))
    }

    /// Return the set of constructors resulting from splitting the wildcard. As explained at the
    /// top of the file, if any constructors are missing we can ignore the present ones.
    fn into_ctors(self, pcx: PatCtxt<'_, '_, 'tcx>) -> SmallVec<[Constructor<'tcx>; 1]> {
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
            let report_when_all_missing = pcx.is_top_level && !MatchCheckCtxt::is_numeric(pcx.ty);
            let ctor = if !self.matrix_ctors.is_empty() || report_when_all_missing {
                if pcx.is_non_exhaustive {
                    Missing {
                        nonexhaustive_enum_missing_real_variants: self
                            .iter_missing(pcx)
                            .filter(|c| !c.is_non_exhaustive())
                            .next()
                            .is_some(),
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
#[derive(Debug, Clone, Copy)]
pub(super) struct Fields<'p, 'tcx> {
    fields: &'p [DeconstructedPat<'p, 'tcx>],
}

impl<'p, 'tcx> Fields<'p, 'tcx> {
    pub(super) fn empty() -> Self {
        Fields { fields: &[] }
    }

    pub(super) fn singleton(
        cx: &MatchCheckCtxt<'p, 'tcx>,
        field: DeconstructedPat<'p, 'tcx>,
    ) -> Self {
        let field: &_ = cx.pattern_arena.alloc(field);
        Fields { fields: std::slice::from_ref(field) }
    }

    pub(super) fn from_iter(
        cx: &MatchCheckCtxt<'p, 'tcx>,
        fields: impl IntoIterator<Item = DeconstructedPat<'p, 'tcx>>,
    ) -> Self {
        let fields: &[_] = cx.pattern_arena.alloc_from_iter(fields);
        Fields { fields }
    }

    fn wildcards_from_tys(
        cx: &MatchCheckCtxt<'p, 'tcx>,
        tys: impl IntoIterator<Item = Ty<'tcx>>,
    ) -> Self {
        Fields::from_iter(cx, tys.into_iter().map(DeconstructedPat::wildcard))
    }

    /// Creates a new list of wildcard fields for a given constructor. The result must have a
    /// length of `constructor.arity()`.
    pub(super) fn wildcards(
        cx: &MatchCheckCtxt<'p, 'tcx>,
        ty: Ty<'tcx>,
        constructor: &Constructor<'tcx>,
    ) -> Self {
        let ret = match constructor {
            Single | Variant(_) => {
                let tys = cx.list_variant_nonhidden_fields(ty, constructor).map(|(_, ty)| ty);
                Fields::wildcards_from_tys(cx, tys)
            }
            Tuple(fs) => Fields::wildcards_from_tys(cx, fs.iter().map(|ty| ty.expect_ty())),
            Ref(ty) | BoxPat(ty) => Fields::wildcards_from_tys(cx, once(*ty)),
            Slice(slice, ty) => {
                let arity = slice.arity();
                Fields::wildcards_from_tys(cx, (0..arity).map(|_| *ty))
            }
            Str(..)
            | Bool(..)
            | IntRange(..)
            | F32Range(..)
            | F64Range(..)
            | NonExhaustive
            | Opaque
            | Missing { .. }
            | Wildcard => Fields::empty(),
            Or => {
                bug!("called `Fields::wildcards` on an `Or` ctor")
            }
        };
        debug!("Fields::wildcards({:?}, {:?}) = {:#?}", constructor, ty, ret);
        ret
    }

    /// Returns the list of patterns.
    pub(super) fn iter_patterns<'a>(
        &'a self,
    ) -> impl Iterator<Item = &'p DeconstructedPat<'p, 'tcx>> + Captures<'a> {
        self.fields.iter()
    }
}

/// Values and patterns can be represented as a constructor applied to some fields. This represents
/// a pattern in this form.
/// This also keeps track of whether the pattern has been foundreachable during analysis. For this
/// reason we should be careful not to clone patterns for which we care about that. Use
/// `clone_and_forget_reachability` is you're sure.
pub(crate) struct DeconstructedPat<'p, 'tcx> {
    ctor: Constructor<'tcx>,
    fields: Fields<'p, 'tcx>,
    ty: Ty<'tcx>,
    span: Span,
    reachable: Cell<bool>,
}

impl<'p, 'tcx> DeconstructedPat<'p, 'tcx> {
    pub(super) fn wildcard(ty: Ty<'tcx>) -> Self {
        Self::new(Wildcard, Fields::empty(), ty, DUMMY_SP)
    }

    pub(super) fn new(
        ctor: Constructor<'tcx>,
        fields: Fields<'p, 'tcx>,
        ty: Ty<'tcx>,
        span: Span,
    ) -> Self {
        DeconstructedPat { ctor, fields, ty, span, reachable: Cell::new(false) }
    }

    /// Construct a pattern that matches everything that starts with this constructor.
    /// For example, if `ctor` is a `Constructor::Variant` for `Option::Some`, we get the pattern
    /// `Some(_)`.
    pub(super) fn wild_from_ctor(pcx: PatCtxt<'_, 'p, 'tcx>, ctor: Constructor<'tcx>) -> Self {
        let fields = Fields::wildcards(pcx.cx, pcx.ty, &ctor);
        DeconstructedPat::new(ctor, fields, pcx.ty, DUMMY_SP)
    }

    /// Clone this value. This method emphasizes that cloning loses reachability information and
    /// should be done carefully.
    pub(super) fn clone_and_forget_reachability(&self) -> Self {
        DeconstructedPat::new(self.ctor.clone(), self.fields, self.ty, self.span)
    }

    pub(super) fn is_or_pat(&self) -> bool {
        matches!(self.ctor, Or)
    }

    pub(super) fn ctor(&self) -> &Constructor<'tcx> {
        &self.ctor
    }
    pub(super) fn ty(&self) -> Ty<'tcx> {
        self.ty
    }
    pub(super) fn span(&self) -> Span {
        self.span
    }

    pub(super) fn iter_fields<'a>(
        &'a self,
    ) -> impl Iterator<Item = &'p DeconstructedPat<'p, 'tcx>> + Captures<'a> {
        self.fields.iter_patterns()
    }

    /// Specialize this pattern with a constructor.
    /// `other_ctor` can be different from `self.ctor`, but must be covered by it.
    pub(super) fn specialize<'a>(
        &'a self,
        cx: &MatchCheckCtxt<'p, 'tcx>,
        other_ctor: &Constructor<'tcx>,
    ) -> SmallVec<[&'p DeconstructedPat<'p, 'tcx>; 2]> {
        match (&self.ctor, other_ctor) {
            (Wildcard, _) => {
                // We return a wildcard for each field of `other_ctor`.
                Fields::wildcards(cx, self.ty, other_ctor).iter_patterns().collect()
            }
            (Slice(self_slice, inner_ty), Slice(other_slice, _))
                if self_slice.arity() != other_slice.arity() =>
            {
                // The only tricky case: two slices of different arity. Since `self_slice` covers
                // `other_slice`, `self_slice` must be `VarLen`, i.e. of the form
                // `[prefix, .., suffix]`. Moreover `other_slice` is guaranteed to have a larger
                // arity. So we fill the middle part with enough wildcards to reach the length of
                // the new, larger slice.
                match self_slice.kind {
                    FixedLen(_) => bug!("{:?} doesn't cover {:?}", self_slice, other_slice),
                    VarLen(prefix, suffix) => {
                        let prefix = &self.fields.fields[..prefix];
                        let suffix = &self.fields.fields[self_slice.arity() - suffix..];
                        let wildcard: &_ =
                            cx.pattern_arena.alloc(DeconstructedPat::wildcard(inner_ty));
                        let extra_wildcards = other_slice.arity() - self_slice.arity();
                        let extra_wildcards = (0..extra_wildcards).map(|_| wildcard);
                        prefix.iter().chain(extra_wildcards).chain(suffix).collect()
                    }
                }
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

    /// Report the spans of subpatterns that were not reachable, if any.
    pub(super) fn unreachable_spans(&self) -> Vec<Span> {
        let mut spans = Vec::new();
        self.collect_unreachable_spans(&mut spans);
        spans
    }

    fn collect_unreachable_spans(&self, spans: &mut Vec<Span>) {
        // We don't look at subpatterns if we already reported the whole pattern as unreachable.
        if !self.is_reachable() {
            spans.push(self.span);
        } else {
            for p in self.iter_fields() {
                p.collect_unreachable_spans(spans);
            }
        }
    }
}

/// This is mostly copied from the `Pat` impl. This is best effort and not good enough for a
/// `Display` impl.
impl<'p, 'tcx> fmt::Debug for DeconstructedPat<'p, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Printing lists is a chore.
        let mut first = true;
        let mut start_or_continue = |s| {
            if first {
                first = false;
                ""
            } else {
                s
            }
        };
        let mut start_or_comma = || start_or_continue(", ");

        match &self.ctor {
            Single | Variant(_) | Tuple(_) => {
                if let Some(ident) = MatchCheckCtxt::variant_ident(self.ty(), self.ctor()) {
                    write!(f, "{}", ident)?;
                }

                // Without `cx`, we can't know which field corresponds to which, so we can't
                // get the names of the fields. Instead we just display everything as a suple
                // struct, which should be good enough.
                write!(f, "(")?;
                for p in self.iter_fields() {
                    write!(f, "{}", start_or_comma())?;
                    write!(f, "{:?}", p)?;
                }
                write!(f, ")")
            }
            Ref(_) => {
                let subpattern = self.iter_fields().next().unwrap();
                write!(f, "&{:?}", subpattern)
            }
            BoxPat(_) => {
                let subpattern = self.iter_fields().next().unwrap();
                write!(f, "box {:?}", subpattern)
            }
            Slice(slice, _) => {
                let mut subpatterns = self.fields.iter_patterns();
                write!(f, "[")?;
                match slice.kind {
                    FixedLen(_) => {
                        for p in subpatterns {
                            write!(f, "{}{:?}", start_or_comma(), p)?;
                        }
                    }
                    VarLen(prefix_len, _) => {
                        for p in subpatterns.by_ref().take(prefix_len) {
                            write!(f, "{}{:?}", start_or_comma(), p)?;
                        }
                        write!(f, "{}", start_or_comma())?;
                        write!(f, "..")?;
                        for p in subpatterns {
                            write!(f, "{}{:?}", start_or_comma(), p)?;
                        }
                    }
                }
                write!(f, "]")
            }
            Bool(b) => write!(f, "{}", b),
            IntRange(range) => write!(f, "{:?}", range), // Best-effort, will render chars as ranges etc.
            F32Range(lo, hi, end) => {
                write!(f, "{}", lo)?;
                write!(f, "{}", end)?;
                write!(f, "{}", hi)
            }
            F64Range(lo, hi, end) => {
                write!(f, "{}", lo)?;
                write!(f, "{}", end)?;
                write!(f, "{}", hi)
            }
            Wildcard | Missing { .. } | NonExhaustive => write!(f, "_ : {:?}", self.ty),
            Or => {
                for pat in self.iter_fields() {
                    write!(f, "{}{:?}", start_or_continue(" | "), pat)?;
                }
                Ok(())
            }
            Str(bytes) => write!(f, "{}", String::from_utf8(bytes.to_vec()).unwrap()),
            Opaque => write!(f, "<constant pattern>"),
        }
    }
}
