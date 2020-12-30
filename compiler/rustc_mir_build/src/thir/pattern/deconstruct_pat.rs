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
use super::{FieldPat, Pat, PatKind, PatRange};

use rustc_data_structures::captures::Captures;
use rustc_index::vec::Idx;

use rustc_attr::{SignedInt, UnsignedInt};
use rustc_hir::def_id::DefId;
use rustc_hir::{HirId, RangeEnd};
use rustc_middle::mir::interpret::ConstValue;
use rustc_middle::mir::Field;
use rustc_middle::ty::layout::IntegerExt;
use rustc_middle::ty::{self, Const, Ty, TyCtxt};
use rustc_session::lint;
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::{Integer, Size, VariantIdx};

use smallvec::{smallvec, SmallVec};
use std::cmp::{self, max, min, Ordering};
use std::iter::{once, IntoIterator};
use std::ops::RangeInclusive;

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
                let size = Integer::from_attr(&tcx, SignedInt(ity)).size();
                Some((size, 1u128 << (size.bits() as u128 - 1)))
            }
            ty::Uint(uty) => Some((Integer::from_attr(&tcx, UnsignedInt(uty)).size(), 0)),
            _ => None,
        }
    }

    #[inline]
    fn from_const<'tcx>(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        value: &Const<'tcx>,
    ) -> Option<IntRange> {
        if let Some((target_size, bias)) = Self::integral_size_and_signed_bias(tcx, value.ty) {
            let ty = value.ty;
            let val = (|| {
                if let ty::ConstKind::Value(ConstValue::Scalar(scalar)) = value.val {
                    // For this specific pattern we can skip a lot of effort and go
                    // straight to the result, after doing a bit of checking. (We
                    // could remove this branch and just fall through, which
                    // is more general but much slower.)
                    if let Ok(bits) = scalar.to_bits_or_ptr(target_size, &tcx) {
                        return Some(bits);
                    }
                }
                // This is a more general form of the previous case.
                value.try_eval_bits(tcx, param_env, ty)
            })()?;
            let val = val ^ bias;
            Some(IntRange { range: val..=val })
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
            Some(IntRange { range: lo..=(hi - offset) })
        } else {
            None
        }
    }

    // The return value of `signed_bias` should be XORed with an endpoint to encode/decode it.
    fn signed_bias(tcx: TyCtxt<'_>, ty: Ty<'_>) -> u128 {
        match *ty.kind() {
            ty::Int(ity) => {
                let bits = Integer::from_attr(&tcx, SignedInt(ity)).size().bits() as u128;
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
            Some(IntRange { range: max(lo, other_lo)..=min(hi, other_hi) })
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

    fn to_pat<'tcx>(&self, tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Pat<'tcx> {
        let (lo, hi) = self.boundaries();

        let bias = IntRange::signed_bias(tcx, ty);
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
    pub(super) fn lint_overlapping_range_endpoints<'a, 'tcx: 'a>(
        &self,
        pcx: PatCtxt<'_, '_, 'tcx>,
        ctors: impl Iterator<Item = (&'a Constructor<'tcx>, Span)>,
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

        let overlaps: Vec<_> = ctors
            .filter_map(|(ctor, span)| Some((ctor.as_int_range()?, span)))
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SliceKind {
    /// Patterns of length `n` (`[x, y]`).
    FixedLen(u64),
    /// Patterns using the `..` notation (`[x, .., y]`).
    /// Captures any array constructor of `length >= i + j`.
    /// In the case where `array_len` is `Some(_)`,
    /// this indicates that we only care about the first `i` and the last `j` values of the array,
    /// and everything in between is a wildcard `_`.
    VarLen(u64, u64),
}

impl SliceKind {
    fn arity(self) -> u64 {
        match self {
            FixedLen(length) => length,
            VarLen(prefix, suffix) => prefix + suffix,
        }
    }

    /// Whether this pattern includes patterns of length `other_len`.
    fn covers_length(self, other_len: u64) -> bool {
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
    array_len: Option<u64>,
    /// The kind of pattern it is: fixed-length `[x, y]` or variable length `[x, .., y]`.
    kind: SliceKind,
}

impl Slice {
    fn new(array_len: Option<u64>, kind: SliceKind) -> Self {
        let kind = match (array_len, kind) {
            // If the middle `..` is empty, we effectively have a fixed-length pattern.
            (Some(len), VarLen(prefix, suffix)) if prefix + suffix >= len => FixedLen(len),
            _ => kind,
        };
        Slice { array_len, kind }
    }

    fn arity(self) -> u64 {
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
    array_len: Option<u64>,
    /// The arity of the input slice.
    arity: u64,
    /// The smallest slice bigger than any slice seen. `max_slice.arity()` is the length `L`
    /// described above.
    max_slice: SliceKind,
}

impl SplitVarLenSlice {
    fn new(prefix: u64, suffix: u64, array_len: Option<u64>) -> Self {
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
    Variant(DefId),
    /// Ranges of integer literal values (`2`, `2..=5` or `2..5`).
    IntRange(IntRange),
    /// Ranges of floating-point literal values (`2.0..=5.2`).
    FloatRange(&'tcx ty::Const<'tcx>, &'tcx ty::Const<'tcx>, RangeEnd),
    /// String literals. Strings are not quite the same as `&[u8]` so we treat them separately.
    Str(&'tcx ty::Const<'tcx>),
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

impl<'tcx> Constructor<'tcx> {
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

    fn variant_index_for_adt(&self, adt: &'tcx ty::AdtDef) -> VariantIdx {
        match *self {
            Variant(id) => adt.variant_index_with_id(id),
            Single => {
                assert!(!adt.is_enum());
                VariantIdx::new(0)
            }
            _ => bug!("bad constructor {:?} for adt {:?}", self, adt),
        }
    }

    /// Determines the constructor that the given pattern can be specialized to.
    pub(super) fn from_pat<'p>(cx: &MatchCheckCtxt<'p, 'tcx>, pat: &'p Pat<'tcx>) -> Self {
        match pat.kind.as_ref() {
            PatKind::AscribeUserType { .. } => bug!(), // Handled by `expand_pattern`
            PatKind::Binding { .. } | PatKind::Wild => Wildcard,
            PatKind::Leaf { .. } | PatKind::Deref { .. } => Single,
            &PatKind::Variant { adt_def, variant_index, .. } => {
                Variant(adt_def.variants[variant_index].def_id)
            }
            PatKind::Constant { value } => {
                if let Some(int_range) = IntRange::from_const(cx.tcx, cx.param_env, value) {
                    IntRange(int_range)
                } else {
                    match pat.ty.kind() {
                        ty::Float(_) => FloatRange(value, value, RangeEnd::Included),
                        // In `expand_pattern`, we convert string literals to `&CONST` patterns with
                        // `CONST` a pattern of type `str`. In truth this contains a constant of type
                        // `&str`.
                        ty::Str => Str(value),
                        // All constants that can be structurally matched have already been expanded
                        // into the corresponding `Pat`s by `const_to_pat`. Constants that remain are
                        // opaque.
                        _ => Opaque,
                    }
                }
            }
            &PatKind::Range(PatRange { lo, hi, end }) => {
                let ty = lo.ty;
                if let Some(int_range) = IntRange::from_range(
                    cx.tcx,
                    lo.eval_bits(cx.tcx, cx.param_env, lo.ty),
                    hi.eval_bits(cx.tcx, cx.param_env, hi.ty),
                    ty,
                    &end,
                ) {
                    IntRange(int_range)
                } else {
                    FloatRange(lo, hi, end)
                }
            }
            PatKind::Array { prefix, slice, suffix } | PatKind::Slice { prefix, slice, suffix } => {
                let array_len = match pat.ty.kind() {
                    ty::Array(_, length) => Some(length.eval_usize(cx.tcx, cx.param_env)),
                    ty::Slice(_) => None,
                    _ => span_bug!(pat.span, "bad ty {:?} for slice pattern", pat.ty),
                };
                let prefix = prefix.len() as u64;
                let suffix = suffix.len() as u64;
                let kind = if slice.is_some() {
                    VarLen(prefix, suffix)
                } else {
                    FixedLen(prefix + suffix)
                };
                Slice(Slice::new(array_len, kind))
            }
            PatKind::Or { .. } => bug!("Or-pattern should have been expanded earlier on."),
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
        debug!("Constructor::split({:#?})", self);

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
            (Missing | Wildcard, _) => false,

            (Single, Single) => true,
            (Variant(self_id), Variant(other_id)) => self_id == other_id,

            (IntRange(self_range), IntRange(other_range)) => self_range.is_covered_by(other_range),
            (
                FloatRange(self_from, self_to, self_end),
                FloatRange(other_from, other_to, other_end),
            ) => {
                match (
                    compare_const_vals(pcx.cx.tcx, self_to, other_to, pcx.cx.param_env, pcx.ty),
                    compare_const_vals(pcx.cx.tcx, self_from, other_from, pcx.cx.param_env, pcx.ty),
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
                match compare_const_vals(pcx.cx.tcx, self_val, other_val, pcx.cx.param_env, pcx.ty)
                {
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
                let len = len.eval_usize(cx.tcx, cx.param_env);
                if len != 0 && cx.is_uninhabited(sub_ty) {
                    smallvec![]
                } else {
                    smallvec![Slice(Slice::new(Some(len), VarLen(0, 0)))]
                }
            }
            // Treat arrays of a constant but unknown length like slices.
            ty::Array(sub_ty, _) | ty::Slice(sub_ty) => {
                let kind = if cx.is_uninhabited(sub_ty) { FixedLen(0) } else { VarLen(0, 0) };
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

                // If `exhaustive_patterns` is disabled and our scrutinee is an empty enum, we treat it
                // as though it had an "unknown" constructor to avoid exposing its emptiness. The
                // exception is if the pattern is at the top level, because we want empty matches to be
                // considered exhaustive.
                let is_secretly_empty = def.variants.is_empty()
                    && !cx.tcx.features().exhaustive_patterns
                    && !pcx.is_top_level;

                if is_secretly_empty || is_declared_nonexhaustive {
                    smallvec![NonExhaustive]
                } else if cx.tcx.features().exhaustive_patterns {
                    // If `exhaustive_patterns` is enabled, we exclude variants known to be
                    // uninhabited.
                    def.variants
                        .iter()
                        .filter(|v| {
                            !v.uninhabited_from(cx.tcx, substs, def.adt_kind(), cx.param_env)
                                .contains(cx.tcx, cx.module)
                        })
                        .map(|v| Variant(v.def_id))
                        .collect()
                } else {
                    def.variants.iter().map(|v| Variant(v.def_id)).collect()
                }
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
                let bits = Integer::from_attr(&cx.tcx, SignedInt(ity)).size().bits() as u128;
                let min = 1u128 << (bits - 1);
                let max = min - 1;
                smallvec![make_range(min, max)]
            }
            &ty::Uint(uty) => {
                let size = Integer::from_attr(&cx.tcx, UnsignedInt(uty)).size();
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

/// Some fields need to be explicitly hidden away in certain cases; see the comment above the
/// `Fields` struct. This struct represents such a potentially-hidden field.
#[derive(Debug, Copy, Clone)]
pub(super) enum FilteredField<'p, 'tcx> {
    Kept(&'p Pat<'tcx>),
    Hidden,
}

impl<'p, 'tcx> FilteredField<'p, 'tcx> {
    fn kept(self) -> Option<&'p Pat<'tcx>> {
        match self {
            FilteredField::Kept(p) => Some(p),
            FilteredField::Hidden => None,
        }
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
pub(super) enum Fields<'p, 'tcx> {
    /// Lists of patterns that don't contain any filtered fields.
    /// `Slice` and `Vec` behave the same; the difference is only to avoid allocating and
    /// triple-dereferences when possible. Frankly this is premature optimization, I (Nadrieril)
    /// have not measured if it really made a difference.
    Slice(&'p [Pat<'tcx>]),
    Vec(SmallVec<[&'p Pat<'tcx>; 2]>),
    /// Patterns where some of the fields need to be hidden. For all intents and purposes we only
    /// care about the non-hidden fields. We need to keep the real field index for those fields;
    /// we're morally storing a `Vec<(usize, &Pat)>` but what we do is more convenient.
    /// `len` counts the number of non-hidden fields
    Filtered {
        fields: SmallVec<[FilteredField<'p, 'tcx>; 2]>,
        len: usize,
    },
}

impl<'p, 'tcx> Fields<'p, 'tcx> {
    /// Internal use. Use `Fields::wildcards()` instead.
    /// Must not be used if the pattern is a field of a struct/tuple/variant.
    fn from_single_pattern(pat: &'p Pat<'tcx>) -> Self {
        Fields::Slice(std::slice::from_ref(pat))
    }

    /// Convenience; internal use.
    fn wildcards_from_tys(
        cx: &MatchCheckCtxt<'p, 'tcx>,
        tys: impl IntoIterator<Item = Ty<'tcx>>,
    ) -> Self {
        let wilds = tys.into_iter().map(Pat::wildcard_from_ty);
        let pats = cx.pattern_arena.alloc_from_iter(wilds);
        Fields::Slice(pats)
    }

    /// Creates a new list of wildcard fields for a given constructor.
    pub(super) fn wildcards(pcx: PatCtxt<'_, 'p, 'tcx>, constructor: &Constructor<'tcx>) -> Self {
        let ty = pcx.ty;
        let cx = pcx.cx;
        let wildcard_from_ty = |ty| &*cx.pattern_arena.alloc(Pat::wildcard_from_ty(ty));

        let ret = match constructor {
            Single | Variant(_) => match ty.kind() {
                ty::Tuple(ref fs) => {
                    Fields::wildcards_from_tys(cx, fs.into_iter().map(|ty| ty.expect_ty()))
                }
                ty::Ref(_, rty, _) => Fields::from_single_pattern(wildcard_from_ty(rty)),
                ty::Adt(adt, substs) => {
                    if adt.is_box() {
                        // Use T as the sub pattern type of Box<T>.
                        Fields::from_single_pattern(wildcard_from_ty(substs.type_at(0)))
                    } else {
                        let variant = &adt.variants[constructor.variant_index_for_adt(adt)];
                        // Whether we must not match the fields of this variant exhaustively.
                        let is_non_exhaustive =
                            variant.is_field_list_non_exhaustive() && !adt.did.is_local();
                        let field_tys = variant.fields.iter().map(|field| field.ty(cx.tcx, substs));
                        // In the following cases, we don't need to filter out any fields. This is
                        // the vast majority of real cases, since uninhabited fields are uncommon.
                        let has_no_hidden_fields = (adt.is_enum() && !is_non_exhaustive)
                            || !field_tys.clone().any(|ty| cx.is_uninhabited(ty));

                        if has_no_hidden_fields {
                            Fields::wildcards_from_tys(cx, field_tys)
                        } else {
                            let mut len = 0;
                            let fields = variant
                                .fields
                                .iter()
                                .map(|field| {
                                    let ty = field.ty(cx.tcx, substs);
                                    let is_visible = adt.is_enum()
                                        || field.vis.is_accessible_from(cx.module, cx.tcx);
                                    let is_uninhabited = cx.is_uninhabited(ty);

                                    // In the cases of either a `#[non_exhaustive]` field list
                                    // or a non-public field, we hide uninhabited fields in
                                    // order not to reveal the uninhabitedness of the whole
                                    // variant.
                                    if is_uninhabited && (!is_visible || is_non_exhaustive) {
                                        FilteredField::Hidden
                                    } else {
                                        len += 1;
                                        FilteredField::Kept(wildcard_from_ty(ty))
                                    }
                                })
                                .collect();
                            Fields::Filtered { fields, len }
                        }
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
            Str(..) | FloatRange(..) | IntRange(..) | NonExhaustive | Opaque | Missing
            | Wildcard => Fields::Slice(&[]),
        };
        debug!("Fields::wildcards({:?}, {:?}) = {:#?}", constructor, ty, ret);
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
    pub(super) fn apply(self, pcx: PatCtxt<'_, 'p, 'tcx>, ctor: &Constructor<'tcx>) -> Pat<'tcx> {
        let subpatterns_and_indices = self.patterns_and_indices();
        let mut subpatterns = subpatterns_and_indices.iter().map(|&(_, p)| p).cloned();

        let pat = match ctor {
            Single | Variant(_) => match pcx.ty.kind() {
                ty::Adt(..) | ty::Tuple(..) => {
                    // We want the real indices here.
                    let subpatterns = subpatterns_and_indices
                        .iter()
                        .map(|&(field, p)| FieldPat { field, pattern: p.clone() })
                        .collect();

                    if let ty::Adt(adt, substs) = pcx.ty.kind() {
                        if adt.is_enum() {
                            PatKind::Variant {
                                adt_def: adt,
                                substs,
                                variant_index: ctor.variant_index_for_adt(adt),
                                subpatterns,
                            }
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
                ty::Ref(..) => PatKind::Deref { subpattern: subpatterns.next().unwrap() },
                ty::Slice(_) | ty::Array(..) => bug!("bad slice pattern {:?} {:?}", ctor, pcx.ty),
                _ => PatKind::Wild,
            },
            Slice(slice) => match slice.kind {
                FixedLen(_) => {
                    PatKind::Slice { prefix: subpatterns.collect(), slice: None, suffix: vec![] }
                }
                VarLen(prefix, _) => {
                    let mut prefix: Vec<_> = subpatterns.by_ref().take(prefix as usize).collect();
                    if slice.array_len.is_some() {
                        // Improves diagnostics a bit: if the type is a known-size array, instead
                        // of reporting `[x, _, .., _, y]`, we prefer to report `[x, .., y]`.
                        // This is incorrect if the size is not known, since `[_, ..]` captures
                        // arrays of lengths `>= 1` whereas `[..]` captures any length.
                        while !prefix.is_empty() && prefix.last().unwrap().is_wildcard() {
                            prefix.pop();
                        }
                    }
                    let suffix: Vec<_> = if slice.array_len.is_some() {
                        // Same as above.
                        subpatterns.skip_while(Pat::is_wildcard).collect()
                    } else {
                        subpatterns.collect()
                    };
                    let wild = Pat::wildcard_from_ty(pcx.ty);
                    PatKind::Slice { prefix, slice: Some(wild), suffix }
                }
            },
            &Str(value) => PatKind::Constant { value },
            &FloatRange(lo, hi, end) => PatKind::Range(PatRange { lo, hi, end }),
            IntRange(range) => return range.to_pat(pcx.cx.tcx, pcx.ty),
            NonExhaustive => PatKind::Wild,
            Wildcard => return Pat::wildcard_from_ty(pcx.ty),
            Opaque => bug!("we should not try to apply an opaque constructor"),
            Missing => bug!(
                "trying to apply the `Missing` constructor; this should have been done in `apply_constructors`"
            ),
        };

        Pat { ty: pcx.ty, span: DUMMY_SP, kind: Box::new(pat) }
    }

    /// Returns the number of patterns. This is the same as the arity of the constructor used to
    /// construct `self`.
    pub(super) fn len(&self) -> usize {
        match self {
            Fields::Slice(pats) => pats.len(),
            Fields::Vec(pats) => pats.len(),
            Fields::Filtered { len, .. } => *len,
        }
    }

    /// Returns the list of patterns along with the corresponding field indices.
    fn patterns_and_indices(&self) -> SmallVec<[(Field, &'p Pat<'tcx>); 2]> {
        match self {
            Fields::Slice(pats) => {
                pats.iter().enumerate().map(|(i, p)| (Field::new(i), p)).collect()
            }
            Fields::Vec(pats) => {
                pats.iter().copied().enumerate().map(|(i, p)| (Field::new(i), p)).collect()
            }
            Fields::Filtered { fields, .. } => {
                // Indices must be relative to the full list of patterns
                fields
                    .iter()
                    .enumerate()
                    .filter_map(|(i, p)| Some((Field::new(i), p.kept()?)))
                    .collect()
            }
        }
    }

    /// Returns the list of patterns.
    pub(super) fn into_patterns(self) -> SmallVec<[&'p Pat<'tcx>; 2]> {
        match self {
            Fields::Slice(pats) => pats.iter().collect(),
            Fields::Vec(pats) => pats,
            Fields::Filtered { fields, .. } => fields.iter().filter_map(|p| p.kept()).collect(),
        }
    }

    /// Overrides some of the fields with the provided patterns. Exactly like
    /// `replace_fields_indexed`, except that it takes `FieldPat`s as input.
    fn replace_with_fieldpats(
        &self,
        new_pats: impl IntoIterator<Item = &'p FieldPat<'tcx>>,
    ) -> Self {
        self.replace_fields_indexed(
            new_pats.into_iter().map(|pat| (pat.field.index(), &pat.pattern)),
        )
    }

    /// Overrides some of the fields with the provided patterns. This is used when a pattern
    /// defines some fields but not all, for example `Foo { field1: Some(_), .. }`: here we start
    /// with a `Fields` that is just one wildcard per field of the `Foo` struct, and override the
    /// entry corresponding to `field1` with the pattern `Some(_)`. This is also used for slice
    /// patterns for the same reason.
    fn replace_fields_indexed(
        &self,
        new_pats: impl IntoIterator<Item = (usize, &'p Pat<'tcx>)>,
    ) -> Self {
        let mut fields = self.clone();
        if let Fields::Slice(pats) = fields {
            fields = Fields::Vec(pats.iter().collect());
        }

        match &mut fields {
            Fields::Vec(pats) => {
                for (i, pat) in new_pats {
                    pats[i] = pat
                }
            }
            Fields::Filtered { fields, .. } => {
                for (i, pat) in new_pats {
                    if let FilteredField::Kept(p) = &mut fields[i] {
                        *p = pat
                    }
                }
            }
            Fields::Slice(_) => unreachable!(),
        }
        fields
    }

    /// Replaces contained fields with the given list of patterns. There must be `len()` patterns
    /// in `pats`.
    pub(super) fn replace_fields(
        &self,
        cx: &MatchCheckCtxt<'p, 'tcx>,
        pats: impl IntoIterator<Item = Pat<'tcx>>,
    ) -> Self {
        let pats: &[_] = cx.pattern_arena.alloc_from_iter(pats);

        match self {
            Fields::Filtered { fields, len } => {
                let mut pats = pats.iter();
                let mut fields = fields.clone();
                for f in &mut fields {
                    if let FilteredField::Kept(p) = f {
                        // We take one input pattern for each `Kept` field, in order.
                        *p = pats.next().unwrap();
                    }
                }
                Fields::Filtered { fields, len: *len }
            }
            _ => Fields::Slice(pats),
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
    pub(super) fn replace_with_pattern_arguments(&self, pat: &'p Pat<'tcx>) -> Self {
        match pat.kind.as_ref() {
            PatKind::Deref { subpattern } => {
                assert_eq!(self.len(), 1);
                Fields::from_single_pattern(subpattern)
            }
            PatKind::Leaf { subpatterns } | PatKind::Variant { subpatterns, .. } => {
                self.replace_with_fieldpats(subpatterns)
            }
            PatKind::Array { prefix, suffix, .. } | PatKind::Slice { prefix, suffix, .. } => {
                // Number of subpatterns for the constructor
                let ctor_arity = self.len();

                // Replace the prefix and the suffix with the given patterns, leaving wildcards in
                // the middle if there was a subslice pattern `..`.
                let prefix = prefix.iter().enumerate();
                let suffix =
                    suffix.iter().enumerate().map(|(i, p)| (ctor_arity - suffix.len() + i, p));
                self.replace_fields_indexed(prefix.chain(suffix))
            }
            _ => self.clone(),
        }
    }
}
