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

use super::compare_const_vals;
use super::usefulness::{MatchCheckCtxt, PatCtxt};

use rustc_data_structures::captures::Captures;
use rustc_index::vec::Idx;

use rustc_hir::{HirId, RangeEnd};
use rustc_middle::mir::Field;
use rustc_middle::thir::{FieldPat, Pat, PatKind, PatRange};
use rustc_middle::ty::layout::IntegerExt;
use rustc_middle::ty::{self, Const, Ty, TyCtxt, VariantDef};
use rustc_middle::{middle::stability::EvalResult, mir::interpret::ConstValue};
use rustc_session::lint;
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::{Integer, Size, VariantIdx};

use smallvec::{smallvec, SmallVec};
use std::cell::Cell;
use std::cmp::{self, max, min, Ordering};
use std::fmt;
use std::iter::{once, IntoIterator};
use std::ops::RangeInclusive;

/// Recursively expand this pattern into its subpatterns. Only useful for or-patterns.
fn expand_or_pat<'p, 'tcx>(pat: &'p Pat<'tcx>) -> Vec<&'p Pat<'tcx>> {
    fn expand<'p, 'tcx>(pat: &'p Pat<'tcx>, vec: &mut Vec<&'p Pat<'tcx>>) {
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
#[derive(Clone, PartialEq, Eq)]
pub(super) struct IntRange {
    range: RangeInclusive<u128>,
    /// Keeps the bias used for encoding the range. It depends on the type of the range and
    /// possibly the pointer size of the current architecture. The algorithm ensures we never
    /// compare `IntRange`s with different types/architectures.
    bias: u128,
}

impl IntRange {
    #[inline]
    fn is_integral(ty: Ty<'_>) -> bool {
        matches!(ty.kind(), ty::Char | ty::Int(_) | ty::Uint(_) | ty::Bool)
    }

    fn is_singleton(&self) -> bool {
        self.range.start() == self.range.end()
    }

    fn boundaries(&self) -> (u128, u128) {
        (*self.range.start(), *self.range.end())
    }

    #[inline]
    fn integral_size_and_signed_bias(tcx: TyCtxt<'_>, ty: Ty<'_>) -> Option<(Size, u128)> {
        match *ty.kind() {
            ty::Bool => Some((Size::from_bytes(1), 0)),
            ty::Char => Some((Size::from_bytes(4), 0)),
            ty::Int(ity) => {
                let size = Integer::from_int_ty(&tcx, ity).size();
                Some((size, 1u128 << (size.bits() as u128 - 1)))
            }
            ty::Uint(uty) => Some((Integer::from_uint_ty(&tcx, uty).size(), 0)),
            _ => None,
        }
    }

    #[inline]
    fn from_const<'tcx>(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        value: Const<'tcx>,
    ) -> Option<IntRange> {
        let ty = value.ty();
        if let Some((target_size, bias)) = Self::integral_size_and_signed_bias(tcx, ty) {
            let val = (|| {
                if let ty::ConstKind::Value(ConstValue::Scalar(scalar)) = value.val() {
                    // For this specific pattern we can skip a lot of effort and go
                    // straight to the result, after doing a bit of checking. (We
                    // could remove this branch and just fall through, which
                    // is more general but much slower.)
                    if let Ok(bits) = scalar.to_bits_or_ptr_internal(target_size) {
                        return Some(bits);
                    }
                }
                // This is a more general form of the previous case.
                value.try_eval_bits(tcx, param_env, ty)
            })()?;
            let val = val ^ bias;
            Some(IntRange { range: val..=val, bias })
        } else {
            None
        }
    }

    #[inline]
    fn from_range<'tcx>(
        tcx: TyCtxt<'tcx>,
        lo: u128,
        hi: u128,
        ty: Ty<'tcx>,
        end: &RangeEnd,
    ) -> Option<IntRange> {
        if Self::is_integral(ty) {
            // Perform a shift if the underlying types are signed,
            // which makes the interval arithmetic simpler.
            let bias = IntRange::signed_bias(tcx, ty);
            let (lo, hi) = (lo ^ bias, hi ^ bias);
            let offset = (*end == RangeEnd::Excluded) as u128;
            if lo > hi || (lo == hi && *end == RangeEnd::Excluded) {
                // This should have been caught earlier by E0030.
                bug!("malformed range pattern: {}..={}", lo, (hi - offset));
            }
            Some(IntRange { range: lo..=(hi - offset), bias })
        } else {
            None
        }
    }

    // The return value of `signed_bias` should be XORed with an endpoint to encode/decode it.
    fn signed_bias(tcx: TyCtxt<'_>, ty: Ty<'_>) -> u128 {
        match *ty.kind() {
            ty::Int(ity) => {
                let bits = Integer::from_int_ty(&tcx, ity).size().bits() as u128;
                1u128 << (bits - 1)
            }
            _ => 0,
        }
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

    fn suspicious_intersection(&self, other: &Self) -> bool {
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
        (lo == other_hi || hi == other_lo) && !self.is_singleton() && !other.is_singleton()
    }

    /// Only used for displaying the range properly.
    fn to_pat<'tcx>(&self, tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Pat<'tcx> {
        let (lo, hi) = self.boundaries();

        let bias = self.bias;
        let (lo, hi) = (lo ^ bias, hi ^ bias);

        let env = ty::ParamEnv::empty().and(ty);
        let lo_const = ty::Const::from_bits(tcx, lo, env);
        let hi_const = ty::Const::from_bits(tcx, hi, env);

        let kind = if lo == hi {
            PatKind::Constant { value: lo_const }
        } else {
            PatKind::Range(PatRange { lo: lo_const, hi: hi_const, end: RangeEnd::Included })
        };

        Pat { ty, span: DUMMY_SP, kind: Box::new(kind) }
    }

    /// Lint on likely incorrect range patterns (#63987)
    pub(super) fn lint_overlapping_range_endpoints<'a, 'p: 'a, 'tcx: 'a>(
        &self,
        pcx: PatCtxt<'_, 'p, 'tcx>,
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

        let overlaps: Vec<_> = pats
            .filter_map(|pat| Some((pat.ctor().as_int_range()?, pat.span())))
            .filter(|(range, _)| self.suspicious_intersection(range))
            .map(|(range, span)| (self.intersection(&range).unwrap(), span))
            .collect();

        if !overlaps.is_empty() {
            pcx.cx.tcx.struct_span_lint_hir(
                lint::builtin::OVERLAPPING_RANGE_ENDPOINTS,
                hir_id,
                pcx.span,
                |lint| {
                    let mut err = lint.build("multiple patterns overlap on their endpoints");
                    for (int_range, span) in overlaps {
                        err.span_label(
                            span,
                            &format!(
                                "this range overlaps on `{}`...",
                                int_range.to_pat(pcx.cx.tcx, pcx.ty)
                            ),
                        );
                    }
                    err.span_label(pcx.span, "... with this range");
                    err.note("you likely meant to write mutually exclusive ranges");
                    err.emit();
                },
            );
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

/// Note: this is often not what we want: e.g. `false` is converted into the range `0..=0` and
/// would be displayed as such. To render properly, convert to a pattern first.
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
enum SliceKind {
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
    array_len: Option<usize>,
    /// The kind of pattern it is: fixed-length `[x, y]` or variable length `[x, .., y]`.
    kind: SliceKind,
}

impl Slice {
    fn new(array_len: Option<usize>, kind: SliceKind) -> Self {
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
    /// The constructor for patterns that have a single constructor, like tuples, struct patterns
    /// and fixed-length arrays.
    Single,
    /// Enum variants.
    Variant(VariantIdx),
    /// Ranges of integer literal values (`2`, `2..=5` or `2..5`).
    IntRange(IntRange),
    /// Ranges of floating-point literal values (`2.0..=5.2`).
    FloatRange(ty::Const<'tcx>, ty::Const<'tcx>, RangeEnd),
    /// String literals. Strings are not quite the same as `&[u8]` so we treat them separately.
    Str(ty::Const<'tcx>),
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
            Slice(slice) => Some(*slice),
            _ => None,
        }
    }

    /// Checks if the `Constructor` is a variant and `TyCtxt::eval_stability` returns
    /// `EvalResult::Deny { .. }`.
    ///
    /// This means that the variant has a stdlib unstable feature marking it.
    pub(super) fn is_unstable_variant(&self, pcx: PatCtxt<'_, '_, 'tcx>) -> bool {
        if let Constructor::Variant(idx) = self {
            if let ty::Adt(adt, _) = pcx.ty.kind() {
                let variant_def_id = adt.variants[*idx].def_id;
                // Filter variants that depend on a disabled unstable feature.
                return matches!(
                    pcx.cx.tcx.eval_stability(variant_def_id, None, DUMMY_SP, None),
                    EvalResult::Deny { .. }
                );
            }
        }
        false
    }

    /// Checks if the `Constructor` is a `Constructor::Variant` with a `#[doc(hidden)]`
    /// attribute.
    pub(super) fn is_doc_hidden_variant(&self, pcx: PatCtxt<'_, '_, 'tcx>) -> bool {
        if let Constructor::Variant(idx) = self {
            if let ty::Adt(adt, _) = pcx.ty.kind() {
                let variant_def_id = adt.variants[*idx].def_id;
                return pcx.cx.tcx.is_doc_hidden(variant_def_id);
            }
        }
        false
    }

    fn variant_index_for_adt(&self, adt: &'tcx ty::AdtDef) -> VariantIdx {
        match *self {
            Variant(idx) => idx,
            Single => {
                assert!(!adt.is_enum());
                VariantIdx::new(0)
            }
            _ => bug!("bad constructor {:?} for adt {:?}", self, adt),
        }
    }

    /// The number of fields for this constructor. This must be kept in sync with
    /// `Fields::wildcards`.
    pub(super) fn arity(&self, pcx: PatCtxt<'_, '_, 'tcx>) -> usize {
        match self {
            Single | Variant(_) => match pcx.ty.kind() {
                ty::Tuple(fs) => fs.len(),
                ty::Ref(..) => 1,
                ty::Adt(adt, ..) => {
                    if adt.is_box() {
                        // The only legal patterns of type `Box` (outside `std`) are `_` and box
                        // patterns. If we're here we can assume this is a box pattern.
                        1
                    } else {
                        let variant = &adt.variants[self.variant_index_for_adt(adt)];
                        Fields::list_variant_nonhidden_fields(pcx.cx, pcx.ty, variant).count()
                    }
                }
                _ => bug!("Unexpected type for `Single` constructor: {:?}", pcx.ty),
            },
            Slice(slice) => slice.arity(),
            Str(..)
            | FloatRange(..)
            | IntRange(..)
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
            &Slice(Slice { kind: VarLen(self_prefix, self_suffix), array_len }) => {
                let mut split_self = SplitVarLenSlice::new(self_prefix, self_suffix, array_len);
                let slices = ctors.filter_map(|c| c.as_slice()).map(|s| s.kind);
                split_self.split(slices);
                split_self.iter().map(Slice).collect()
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
            (Variant(self_id), Variant(other_id)) => self_id == other_id,

            (IntRange(self_range), IntRange(other_range)) => self_range.is_covered_by(other_range),
            (
                FloatRange(self_from, self_to, self_end),
                FloatRange(other_from, other_to, other_end),
            ) => {
                match (
                    compare_const_vals(pcx.cx.tcx, *self_to, *other_to, pcx.cx.param_env, pcx.ty),
                    compare_const_vals(
                        pcx.cx.tcx,
                        *self_from,
                        *other_from,
                        pcx.cx.param_env,
                        pcx.ty,
                    ),
                ) {
                    (Some(to), Some(from)) => {
                        (from == Ordering::Greater || from == Ordering::Equal)
                            && (to == Ordering::Less
                                || (other_end == self_end && to == Ordering::Equal))
                    }
                    _ => false,
                }
            }
            (Str(self_val), Str(other_val)) => {
                // FIXME: there's probably a more direct way of comparing for equality
                match compare_const_vals(
                    pcx.cx.tcx,
                    *self_val,
                    *other_val,
                    pcx.cx.param_env,
                    pcx.ty,
                ) {
                    Some(comparison) => comparison == Ordering::Equal,
                    None => false,
                }
            }
            (Slice(self_slice), Slice(other_slice)) => self_slice.is_covered_by(*other_slice),

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
            // If `self` is `Single`, `used_ctors` cannot contain anything else than `Single`s.
            Single => !used_ctors.is_empty(),
            Variant(vid) => used_ctors.iter().any(|c| matches!(c, Variant(i) if i == vid)),
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
        debug!("SplitWildcard::new({:?})", pcx.ty);
        let cx = pcx.cx;
        let make_range = |start, end| {
            IntRange(
                // `unwrap()` is ok because we know the type is an integer.
                IntRange::from_range(cx.tcx, start, end, pcx.ty, &RangeEnd::Included).unwrap(),
            )
        };
        // This determines the set of all possible constructors for the type `pcx.ty`. For numbers,
        // arrays and slices we use ranges and variable-length slices when appropriate.
        //
        // If the `exhaustive_patterns` feature is enabled, we make sure to omit constructors that
        // are statically impossible. E.g., for `Option<!>`, we do not include `Some(_)` in the
        // returned list of constructors.
        // Invariant: this is empty if and only if the type is uninhabited (as determined by
        // `cx.is_uninhabited()`).
        let all_ctors = match pcx.ty.kind() {
            ty::Bool => smallvec![make_range(0, 1)],
            ty::Array(sub_ty, len) if len.try_eval_usize(cx.tcx, cx.param_env).is_some() => {
                let len = len.eval_usize(cx.tcx, cx.param_env) as usize;
                if len != 0 && cx.is_uninhabited(*sub_ty) {
                    smallvec![]
                } else {
                    smallvec![Slice(Slice::new(Some(len), VarLen(0, 0)))]
                }
            }
            // Treat arrays of a constant but unknown length like slices.
            ty::Array(sub_ty, _) | ty::Slice(sub_ty) => {
                let kind = if cx.is_uninhabited(*sub_ty) { FixedLen(0) } else { VarLen(0, 0) };
                smallvec![Slice(Slice::new(None, kind))]
            }
            ty::Adt(def, substs) if def.is_enum() => {
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

                let is_exhaustive_pat_feature = cx.tcx.features().exhaustive_patterns;

                // If `exhaustive_patterns` is disabled and our scrutinee is an empty enum, we treat it
                // as though it had an "unknown" constructor to avoid exposing its emptiness. The
                // exception is if the pattern is at the top level, because we want empty matches to be
                // considered exhaustive.
                let is_secretly_empty =
                    def.variants.is_empty() && !is_exhaustive_pat_feature && !pcx.is_top_level;

                let mut ctors: SmallVec<[_; 1]> = def
                    .variants
                    .iter_enumerated()
                    .filter(|(_, v)| {
                        // If `exhaustive_patterns` is enabled, we exclude variants known to be
                        // uninhabited.
                        let is_uninhabited = is_exhaustive_pat_feature
                            && v.uninhabited_from(cx.tcx, substs, def.adt_kind(), cx.param_env)
                                .contains(cx.tcx, cx.module);
                        !is_uninhabited
                    })
                    .map(|(idx, _)| Variant(idx))
                    .collect();

                if is_secretly_empty || is_declared_nonexhaustive {
                    ctors.push(NonExhaustive);
                }
                ctors
            }
            ty::Char => {
                smallvec![
                    // The valid Unicode Scalar Value ranges.
                    make_range('\u{0000}' as u128, '\u{D7FF}' as u128),
                    make_range('\u{E000}' as u128, '\u{10FFFF}' as u128),
                ]
            }
            ty::Int(_) | ty::Uint(_)
                if pcx.ty.is_ptr_sized_integral()
                    && !cx.tcx.features().precise_pointer_size_matching =>
            {
                // `usize`/`isize` are not allowed to be matched exhaustively unless the
                // `precise_pointer_size_matching` feature is enabled. So we treat those types like
                // `#[non_exhaustive]` enums by returning a special unmatcheable constructor.
                smallvec![NonExhaustive]
            }
            &ty::Int(ity) => {
                let bits = Integer::from_int_ty(&cx.tcx, ity).size().bits() as u128;
                let min = 1u128 << (bits - 1);
                let max = min - 1;
                smallvec![make_range(min, max)]
            }
            &ty::Uint(uty) => {
                let size = Integer::from_uint_ty(&cx.tcx, uty).size();
                let max = size.truncate(u128::MAX);
                smallvec![make_range(0, max)]
            }
            // If `exhaustive_patterns` is disabled and our scrutinee is the never type, we cannot
            // expose its emptiness. The exception is if the pattern is at the top level, because we
            // want empty matches to be considered exhaustive.
            ty::Never if !cx.tcx.features().exhaustive_patterns && !pcx.is_top_level => {
                smallvec![NonExhaustive]
            }
            ty::Never => smallvec![],
            _ if cx.is_uninhabited(pcx.ty) => smallvec![],
            ty::Adt(..) | ty::Tuple(..) | ty::Ref(..) => smallvec![Single],
            // This type is one for which we cannot list constructors, like `str` or `f64`.
            _ => smallvec![NonExhaustive],
        };

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
#[derive(Debug, Clone, Copy)]
pub(super) struct Fields<'p, 'tcx> {
    fields: &'p [DeconstructedPat<'p, 'tcx>],
}

impl<'p, 'tcx> Fields<'p, 'tcx> {
    fn empty() -> Self {
        Fields { fields: &[] }
    }

    fn singleton(cx: &MatchCheckCtxt<'p, 'tcx>, field: DeconstructedPat<'p, 'tcx>) -> Self {
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

    // In the cases of either a `#[non_exhaustive]` field list or a non-public field, we hide
    // uninhabited fields in order not to reveal the uninhabitedness of the whole variant.
    // This lists the fields we keep along with their types.
    fn list_variant_nonhidden_fields<'a>(
        cx: &'a MatchCheckCtxt<'p, 'tcx>,
        ty: Ty<'tcx>,
        variant: &'a VariantDef,
    ) -> impl Iterator<Item = (Field, Ty<'tcx>)> + Captures<'a> + Captures<'p> {
        let (adt, substs) = match ty.kind() {
            ty::Adt(adt, substs) => (adt, substs),
            _ => bug!(),
        };
        // Whether we must not match the fields of this variant exhaustively.
        let is_non_exhaustive = variant.is_field_list_non_exhaustive() && !adt.did.is_local();

        variant.fields.iter().enumerate().filter_map(move |(i, field)| {
            let ty = field.ty(cx.tcx, substs);
            // `field.ty()` doesn't normalize after substituting.
            let ty = cx.tcx.normalize_erasing_regions(cx.param_env, ty);
            let is_visible = adt.is_enum() || field.vis.is_accessible_from(cx.module, cx.tcx);
            let is_uninhabited = cx.is_uninhabited(ty);

            if is_uninhabited && (!is_visible || is_non_exhaustive) {
                None
            } else {
                Some((Field::new(i), ty))
            }
        })
    }

    /// Creates a new list of wildcard fields for a given constructor. The result must have a
    /// length of `constructor.arity()`.
    pub(super) fn wildcards(
        cx: &MatchCheckCtxt<'p, 'tcx>,
        ty: Ty<'tcx>,
        constructor: &Constructor<'tcx>,
    ) -> Self {
        let ret = match constructor {
            Single | Variant(_) => match ty.kind() {
                ty::Tuple(fs) => Fields::wildcards_from_tys(cx, fs.iter().map(|ty| ty.expect_ty())),
                ty::Ref(_, rty, _) => Fields::wildcards_from_tys(cx, once(*rty)),
                ty::Adt(adt, substs) => {
                    if adt.is_box() {
                        // The only legal patterns of type `Box` (outside `std`) are `_` and box
                        // patterns. If we're here we can assume this is a box pattern.
                        Fields::wildcards_from_tys(cx, once(substs.type_at(0)))
                    } else {
                        let variant = &adt.variants[constructor.variant_index_for_adt(adt)];
                        let tys = Fields::list_variant_nonhidden_fields(cx, ty, variant)
                            .map(|(_, ty)| ty);
                        Fields::wildcards_from_tys(cx, tys)
                    }
                }
                _ => bug!("Unexpected type for `Single` constructor: {:?}", ty),
            },
            Slice(slice) => match *ty.kind() {
                ty::Slice(ty) | ty::Array(ty, _) => {
                    let arity = slice.arity();
                    Fields::wildcards_from_tys(cx, (0..arity).map(|_| ty))
                }
                _ => bug!("bad slice pattern {:?} {:?}", constructor, ty),
            },
            Str(..)
            | FloatRange(..)
            | IntRange(..)
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
/// This also keeps track of whether the pattern has been found reachable during analysis. For this
/// reason we should be careful not to clone patterns for which we care about that. Use
/// `clone_and_forget_reachability` if you're sure.
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

    pub(crate) fn from_pat(cx: &MatchCheckCtxt<'p, 'tcx>, pat: &Pat<'tcx>) -> Self {
        let mkpat = |pat| DeconstructedPat::from_pat(cx, pat);
        let ctor;
        let fields;
        match pat.kind.as_ref() {
            PatKind::AscribeUserType { subpattern, .. } => return mkpat(subpattern),
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
                match pat.ty.kind() {
                    ty::Tuple(fs) => {
                        ctor = Single;
                        let mut wilds: SmallVec<[_; 2]> = fs
                            .iter()
                            .map(|ty| ty.expect_ty())
                            .map(DeconstructedPat::wildcard)
                            .collect();
                        for pat in subpatterns {
                            wilds[pat.field.index()] = mkpat(&pat.pattern);
                        }
                        fields = Fields::from_iter(cx, wilds);
                    }
                    ty::Adt(adt, substs) if adt.is_box() => {
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
                        let pat = subpatterns.into_iter().find(|pat| pat.field.index() == 0);
                        let pat = if let Some(pat) = pat {
                            mkpat(&pat.pattern)
                        } else {
                            DeconstructedPat::wildcard(substs.type_at(0))
                        };
                        ctor = Single;
                        fields = Fields::singleton(cx, pat);
                    }
                    ty::Adt(adt, _) => {
                        ctor = match pat.kind.as_ref() {
                            PatKind::Leaf { .. } => Single,
                            PatKind::Variant { variant_index, .. } => Variant(*variant_index),
                            _ => bug!(),
                        };
                        let variant = &adt.variants[ctor.variant_index_for_adt(adt)];
                        // For each field in the variant, we store the relevant index into `self.fields` if any.
                        let mut field_id_to_id: Vec<Option<usize>> =
                            (0..variant.fields.len()).map(|_| None).collect();
                        let tys = Fields::list_variant_nonhidden_fields(cx, pat.ty, variant)
                            .enumerate()
                            .map(|(i, (field, ty))| {
                                field_id_to_id[field.index()] = Some(i);
                                ty
                            });
                        let mut wilds: SmallVec<[_; 2]> =
                            tys.map(DeconstructedPat::wildcard).collect();
                        for pat in subpatterns {
                            if let Some(i) = field_id_to_id[pat.field.index()] {
                                wilds[i] = mkpat(&pat.pattern);
                            }
                        }
                        fields = Fields::from_iter(cx, wilds);
                    }
                    _ => bug!("pattern has unexpected type: pat: {:?}, ty: {:?}", pat, pat.ty),
                }
            }
            PatKind::Constant { value } => {
                if let Some(int_range) = IntRange::from_const(cx.tcx, cx.param_env, *value) {
                    ctor = IntRange(int_range);
                    fields = Fields::empty();
                } else {
                    match pat.ty.kind() {
                        ty::Float(_) => {
                            ctor = FloatRange(*value, *value, RangeEnd::Included);
                            fields = Fields::empty();
                        }
                        ty::Ref(_, t, _) if t.is_str() => {
                            // We want a `&str` constant to behave like a `Deref` pattern, to be compatible
                            // with other `Deref` patterns. This could have been done in `const_to_pat`,
                            // but that causes issues with the rest of the matching code.
                            // So here, the constructor for a `"foo"` pattern is `&` (represented by
                            // `Single`), and has one field. That field has constructor `Str(value)` and no
                            // fields.
                            // Note: `t` is `str`, not `&str`.
                            let subpattern =
                                DeconstructedPat::new(Str(*value), Fields::empty(), *t, pat.span);
                            ctor = Single;
                            fields = Fields::singleton(cx, subpattern)
                        }
                        // All constants that can be structurally matched have already been expanded
                        // into the corresponding `Pat`s by `const_to_pat`. Constants that remain are
                        // opaque.
                        _ => {
                            ctor = Opaque;
                            fields = Fields::empty();
                        }
                    }
                }
            }
            &PatKind::Range(PatRange { lo, hi, end }) => {
                let ty = lo.ty();
                ctor = if let Some(int_range) = IntRange::from_range(
                    cx.tcx,
                    lo.eval_bits(cx.tcx, cx.param_env, lo.ty()),
                    hi.eval_bits(cx.tcx, cx.param_env, hi.ty()),
                    ty,
                    &end,
                ) {
                    IntRange(int_range)
                } else {
                    FloatRange(lo, hi, end)
                };
                fields = Fields::empty();
            }
            PatKind::Array { prefix, slice, suffix } | PatKind::Slice { prefix, slice, suffix } => {
                let array_len = match pat.ty.kind() {
                    ty::Array(_, length) => Some(length.eval_usize(cx.tcx, cx.param_env) as usize),
                    ty::Slice(_) => None,
                    _ => span_bug!(pat.span, "bad ty {:?} for slice pattern", pat.ty),
                };
                let kind = if slice.is_some() {
                    VarLen(prefix.len(), suffix.len())
                } else {
                    FixedLen(prefix.len() + suffix.len())
                };
                ctor = Slice(Slice::new(array_len, kind));
                fields = Fields::from_iter(cx, prefix.iter().chain(suffix).map(mkpat));
            }
            PatKind::Or { .. } => {
                ctor = Or;
                let pats = expand_or_pat(pat);
                fields = Fields::from_iter(cx, pats.into_iter().map(mkpat));
            }
        }
        DeconstructedPat::new(ctor, fields, pat.ty, pat.span)
    }

    pub(crate) fn to_pat(&self, cx: &MatchCheckCtxt<'p, 'tcx>) -> Pat<'tcx> {
        let is_wildcard = |pat: &Pat<'_>| {
            matches!(*pat.kind, PatKind::Binding { subpattern: None, .. } | PatKind::Wild)
        };
        let mut subpatterns = self.iter_fields().map(|p| p.to_pat(cx));
        let pat = match &self.ctor {
            Single | Variant(_) => match self.ty.kind() {
                ty::Tuple(..) => PatKind::Leaf {
                    subpatterns: subpatterns
                        .enumerate()
                        .map(|(i, p)| FieldPat { field: Field::new(i), pattern: p })
                        .collect(),
                },
                ty::Adt(adt_def, _) if adt_def.is_box() => {
                    // Without `box_patterns`, the only legal pattern of type `Box` is `_` (outside
                    // of `std`). So this branch is only reachable when the feature is enabled and
                    // the pattern is a box pattern.
                    PatKind::Deref { subpattern: subpatterns.next().unwrap() }
                }
                ty::Adt(adt_def, substs) => {
                    let variant_index = self.ctor.variant_index_for_adt(adt_def);
                    let variant = &adt_def.variants[variant_index];
                    let subpatterns = Fields::list_variant_nonhidden_fields(cx, self.ty, variant)
                        .zip(subpatterns)
                        .map(|((field, _ty), pattern)| FieldPat { field, pattern })
                        .collect();

                    if adt_def.is_enum() {
                        PatKind::Variant { adt_def, substs, variant_index, subpatterns }
                    } else {
                        PatKind::Leaf { subpatterns }
                    }
                }
                // Note: given the expansion of `&str` patterns done in `expand_pattern`, we should
                // be careful to reconstruct the correct constant pattern here. However a string
                // literal pattern will never be reported as a non-exhaustiveness witness, so we
                // ignore this issue.
                ty::Ref(..) => PatKind::Deref { subpattern: subpatterns.next().unwrap() },
                _ => bug!("unexpected ctor for type {:?} {:?}", self.ctor, self.ty),
            },
            Slice(slice) => {
                match slice.kind {
                    FixedLen(_) => PatKind::Slice {
                        prefix: subpatterns.collect(),
                        slice: None,
                        suffix: vec![],
                    },
                    VarLen(prefix, _) => {
                        let mut subpatterns = subpatterns.peekable();
                        let mut prefix: Vec<_> = subpatterns.by_ref().take(prefix).collect();
                        if slice.array_len.is_some() {
                            // Improves diagnostics a bit: if the type is a known-size array, instead
                            // of reporting `[x, _, .., _, y]`, we prefer to report `[x, .., y]`.
                            // This is incorrect if the size is not known, since `[_, ..]` captures
                            // arrays of lengths `>= 1` whereas `[..]` captures any length.
                            while !prefix.is_empty() && is_wildcard(prefix.last().unwrap()) {
                                prefix.pop();
                            }
                            while subpatterns.peek().is_some()
                                && is_wildcard(subpatterns.peek().unwrap())
                            {
                                subpatterns.next();
                            }
                        }
                        let suffix: Vec<_> = subpatterns.collect();
                        let wild = Pat::wildcard_from_ty(self.ty);
                        PatKind::Slice { prefix, slice: Some(wild), suffix }
                    }
                }
            }
            &Str(value) => PatKind::Constant { value },
            &FloatRange(lo, hi, end) => PatKind::Range(PatRange { lo, hi, end }),
            IntRange(range) => return range.to_pat(cx.tcx, self.ty),
            Wildcard | NonExhaustive => PatKind::Wild,
            Missing { .. } => bug!(
                "trying to convert a `Missing` constructor into a `Pat`; this is probably a bug,
                `Missing` should have been processed in `apply_constructors`"
            ),
            Opaque | Or => {
                bug!("can't convert to pattern: {:?}", self)
            }
        };

        Pat { ty: self.ty, span: DUMMY_SP, kind: Box::new(pat) }
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
            (Slice(self_slice), Slice(other_slice))
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
                        let inner_ty = match *self.ty.kind() {
                            ty::Slice(ty) | ty::Array(ty, _) => ty,
                            _ => bug!("bad slice pattern {:?} {:?}", self.ctor, self.ty),
                        };
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
            Single | Variant(_) => match self.ty.kind() {
                ty::Adt(def, _) if def.is_box() => {
                    // Without `box_patterns`, the only legal pattern of type `Box` is `_` (outside
                    // of `std`). So this branch is only reachable when the feature is enabled and
                    // the pattern is a box pattern.
                    let subpattern = self.iter_fields().next().unwrap();
                    write!(f, "box {:?}", subpattern)
                }
                ty::Adt(..) | ty::Tuple(..) => {
                    let variant = match self.ty.kind() {
                        ty::Adt(adt, _) => {
                            Some(&adt.variants[self.ctor.variant_index_for_adt(adt)])
                        }
                        ty::Tuple(_) => None,
                        _ => unreachable!(),
                    };

                    if let Some(variant) = variant {
                        write!(f, "{}", variant.name)?;
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
                // Note: given the expansion of `&str` patterns done in `expand_pattern`, we should
                // be careful to detect strings here. However a string literal pattern will never
                // be reported as a non-exhaustiveness witness, so we can ignore this issue.
                ty::Ref(_, _, mutbl) => {
                    let subpattern = self.iter_fields().next().unwrap();
                    write!(f, "&{}{:?}", mutbl.prefix_str(), subpattern)
                }
                _ => write!(f, "_"),
            },
            Slice(slice) => {
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
            &FloatRange(lo, hi, end) => {
                write!(f, "{}", lo)?;
                write!(f, "{}", end)?;
                write!(f, "{}", hi)
            }
            IntRange(range) => write!(f, "{:?}", range), // Best-effort, will render e.g. `false` as `0..=0`
            Wildcard | Missing { .. } | NonExhaustive => write!(f, "_ : {:?}", self.ty),
            Or => {
                for pat in self.iter_fields() {
                    write!(f, "{}{:?}", start_or_continue(" | "), pat)?;
                }
                Ok(())
            }
            Str(value) => write!(f, "{}", value),
            Opaque => write!(f, "<constant pattern>"),
        }
    }
}
