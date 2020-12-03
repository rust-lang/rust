//! This module provides functions to deconstruct and reconstruct patterns into a constructor
//! applied to some fields. This is used by the `_match` module to compute pattern
//! usefulness/exhaustiveness.
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
use std::iter::IntoIterator;
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
        lo == other_hi || hi == other_lo
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

    /// For exhaustive integer matching, some constructors are grouped within other constructors
    /// (namely integer typed values are grouped within ranges). However, when specialising these
    /// constructors, we want to be specialising for the underlying constructors (the integers), not
    /// the groups (the ranges). Thus we need to split the groups up. Splitting them up naïvely would
    /// mean creating a separate constructor for every single value in the range, which is clearly
    /// impractical. However, observe that for some ranges of integers, the specialisation will be
    /// identical across all values in that range (i.e., there are equivalence classes of ranges of
    /// constructors based on their `U(S(c, P), S(c, p))` outcome). These classes are grouped by
    /// the patterns that apply to them (in the matrix `P`). We can split the range whenever the
    /// patterns that apply to that range (specifically: the patterns that *intersect* with that range)
    /// change.
    /// Our solution, therefore, is to split the range constructor into subranges at every single point
    /// the group of intersecting patterns changes (using the method described below).
    /// And voilà! We're testing precisely those ranges that we need to, without any exhaustive matching
    /// on actual integers. The nice thing about this is that the number of subranges is linear in the
    /// number of rows in the matrix (i.e., the number of cases in the `match` statement), so we don't
    /// need to be worried about matching over gargantuan ranges.
    ///
    /// Essentially, given the first column of a matrix representing ranges, looking like the following:
    ///
    /// |------|  |----------| |-------|    ||
    ///    |-------| |-------|            |----| ||
    ///       |---------|
    ///
    /// We split the ranges up into equivalence classes so the ranges are no longer overlapping:
    ///
    /// |--|--|||-||||--||---|||-------|  |-|||| ||
    ///
    /// The logic for determining how to split the ranges is fairly straightforward: we calculate
    /// boundaries for each interval range, sort them, then create constructors for each new interval
    /// between every pair of boundary points. (This essentially sums up to performing the intuitive
    /// merging operation depicted above.)
    fn split<'p, 'tcx>(
        &self,
        pcx: PatCtxt<'_, 'p, 'tcx>,
        hir_id: Option<HirId>,
    ) -> SmallVec<[Constructor<'tcx>; 1]> {
        /// Represents a border between 2 integers. Because the intervals spanning borders
        /// must be able to cover every integer, we need to be able to represent
        /// 2^128 + 1 such borders.
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
        enum Border {
            JustBefore(u128),
            AfterMax,
        }

        // A function for extracting the borders of an integer interval.
        fn range_borders(r: IntRange) -> impl Iterator<Item = Border> {
            let (lo, hi) = r.range.into_inner();
            let from = Border::JustBefore(lo);
            let to = match hi.checked_add(1) {
                Some(m) => Border::JustBefore(m),
                None => Border::AfterMax,
            };
            vec![from, to].into_iter()
        }

        // Collect the span and range of all the intersecting ranges to lint on likely
        // incorrect range patterns. (#63987)
        let mut overlaps = vec![];
        let row_len = pcx.matrix.column_count().unwrap_or(0);
        // `borders` is the set of borders between equivalence classes: each equivalence
        // class lies between 2 borders.
        let row_borders = pcx
            .matrix
            .head_ctors_and_spans(pcx.cx)
            .filter_map(|(ctor, span)| Some((ctor.as_int_range()?, span)))
            .filter_map(|(range, span)| {
                let intersection = self.intersection(&range);
                let should_lint = self.suspicious_intersection(&range);
                if let (Some(range), 1, true) = (&intersection, row_len, should_lint) {
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
                    overlaps.push((range.clone(), span));
                }
                intersection
            })
            .flat_map(range_borders);
        let self_borders = range_borders(self.clone());
        let mut borders: Vec<_> = row_borders.chain(self_borders).collect();
        borders.sort_unstable();

        self.lint_overlapping_patterns(pcx, hir_id, overlaps);

        // We're going to iterate through every adjacent pair of borders, making sure that
        // each represents an interval of nonnegative length, and convert each such
        // interval into a constructor.
        borders
            .array_windows()
            .filter_map(|&pair| match pair {
                [Border::JustBefore(n), Border::JustBefore(m)] => {
                    if n < m {
                        Some(n..=(m - 1))
                    } else {
                        None
                    }
                }
                [Border::JustBefore(n), Border::AfterMax] => Some(n..=u128::MAX),
                [Border::AfterMax, _] => None,
            })
            .map(|range| IntRange { range })
            .map(IntRange)
            .collect()
    }

    fn lint_overlapping_patterns(
        &self,
        pcx: PatCtxt<'_, '_, '_>,
        hir_id: Option<HirId>,
        overlaps: Vec<(IntRange, Span)>,
    ) {
        if let (true, Some(hir_id)) = (!overlaps.is_empty(), hir_id) {
            pcx.cx.tcx.struct_span_lint_hir(
                lint::builtin::OVERLAPPING_PATTERNS,
                hir_id,
                pcx.span,
                |lint| {
                    let mut err = lint.build("multiple patterns covering the same range");
                    err.span_label(pcx.span, "overlapping patterns");
                    for (int_range, span) in overlaps {
                        // Use the real type for user display of the ranges:
                        err.span_label(
                            span,
                            &format!(
                                "this range overlaps on `{}`",
                                int_range.to_pat(pcx.cx.tcx, pcx.ty),
                            ),
                        );
                    }
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

    /// The exhaustiveness-checking paper does not include any details on
    /// checking variable-length slice patterns. However, they may be
    /// matched by an infinite collection of fixed-length array patterns.
    ///
    /// Checking the infinite set directly would take an infinite amount
    /// of time. However, it turns out that for each finite set of
    /// patterns `P`, all sufficiently large array lengths are equivalent:
    ///
    /// Each slice `s` with a "sufficiently-large" length `l ≥ L` that applies
    /// to exactly the subset `Pₜ` of `P` can be transformed to a slice
    /// `sₘ` for each sufficiently-large length `m` that applies to exactly
    /// the same subset of `P`.
    ///
    /// Because of that, each witness for reachability-checking of one
    /// of the sufficiently-large lengths can be transformed to an
    /// equally-valid witness of any other length, so we only have
    /// to check slices of the "minimal sufficiently-large length"
    /// and less.
    ///
    /// Note that the fact that there is a *single* `sₘ` for each `m`
    /// not depending on the specific pattern in `P` is important: if
    /// you look at the pair of patterns
    ///     `[true, ..]`
    ///     `[.., false]`
    /// Then any slice of length ≥1 that matches one of these two
    /// patterns can be trivially turned to a slice of any
    /// other length ≥1 that matches them and vice-versa,
    /// but the slice of length 2 `[false, true]` that matches neither
    /// of these patterns can't be turned to a slice from length 1 that
    /// matches neither of these patterns, so we have to consider
    /// slices from length 2 there.
    ///
    /// Now, to see that that length exists and find it, observe that slice
    /// patterns are either "fixed-length" patterns (`[_, _, _]`) or
    /// "variable-length" patterns (`[_, .., _]`).
    ///
    /// For fixed-length patterns, all slices with lengths *longer* than
    /// the pattern's length have the same outcome (of not matching), so
    /// as long as `L` is greater than the pattern's length we can pick
    /// any `sₘ` from that length and get the same result.
    ///
    /// For variable-length patterns, the situation is more complicated,
    /// because as seen above the precise value of `sₘ` matters.
    ///
    /// However, for each variable-length pattern `p` with a prefix of length
    /// `plₚ` and suffix of length `slₚ`, only the first `plₚ` and the last
    /// `slₚ` elements are examined.
    ///
    /// Therefore, as long as `L` is positive (to avoid concerns about empty
    /// types), all elements after the maximum prefix length and before
    /// the maximum suffix length are not examined by any variable-length
    /// pattern, and therefore can be added/removed without affecting
    /// them - creating equivalent patterns from any sufficiently-large
    /// length.
    ///
    /// Of course, if fixed-length patterns exist, we must be sure
    /// that our length is large enough to miss them all, so
    /// we can pick `L = max(max(FIXED_LEN)+1, max(PREFIX_LEN) + max(SUFFIX_LEN))`
    ///
    /// for example, with the above pair of patterns, all elements
    /// but the first and last can be added/removed, so any
    /// witness of length ≥2 (say, `[false, false, true]`) can be
    /// turned to a witness from any other length ≥2.
    fn split<'p, 'tcx>(self, pcx: PatCtxt<'_, 'p, 'tcx>) -> SmallVec<[Constructor<'tcx>; 1]> {
        let (self_prefix, self_suffix) = match self.kind {
            VarLen(self_prefix, self_suffix) => (self_prefix, self_suffix),
            _ => return smallvec![Slice(self)],
        };

        let head_ctors = pcx.matrix.head_ctors(pcx.cx).filter(|c| !c.is_wildcard());

        let mut max_prefix_len = self_prefix;
        let mut max_suffix_len = self_suffix;
        let mut max_fixed_len = 0;

        for ctor in head_ctors {
            if let Slice(slice) = ctor {
                match slice.kind {
                    FixedLen(len) => {
                        max_fixed_len = cmp::max(max_fixed_len, len);
                    }
                    VarLen(prefix, suffix) => {
                        max_prefix_len = cmp::max(max_prefix_len, prefix);
                        max_suffix_len = cmp::max(max_suffix_len, suffix);
                    }
                }
            } else {
                bug!("unexpected ctor for slice type: {:?}", ctor);
            }
        }

        // For diagnostics, we keep the prefix and suffix lengths separate, so in the case
        // where `max_fixed_len + 1` is the largest, we adapt `max_prefix_len` accordingly,
        // so that `L = max_prefix_len + max_suffix_len`.
        if max_fixed_len + 1 >= max_prefix_len + max_suffix_len {
            // The subtraction can't overflow thanks to the above check.
            // The new `max_prefix_len` is also guaranteed to be larger than its previous
            // value.
            max_prefix_len = max_fixed_len + 1 - max_suffix_len;
        }

        let final_slice = VarLen(max_prefix_len, max_suffix_len);
        let final_slice = Slice::new(self.array_len, final_slice);
        match self.array_len {
            Some(_) => smallvec![Slice(final_slice)],
            None => {
                // `self` originally covered the range `(self.arity()..infinity)`. We split that
                // range into two: lengths smaller than `final_slice.arity()` are treated
                // independently as fixed-lengths slices, and lengths above are captured by
                // `final_slice`.
                let smaller_lengths = (self.arity()..final_slice.arity()).map(FixedLen);
                smaller_lengths
                    .map(|kind| Slice::new(self.array_len, kind))
                    .chain(Some(final_slice))
                    .map(Slice)
                    .collect()
            }
        }
    }

    /// See `Constructor::is_covered_by`
    fn is_covered_by(self, other: Self) -> bool {
        other.kind.covers_length(self.arity())
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
    ///
    /// `hir_id` is `None` when we're evaluating the wildcard pattern. In that case we do not want
    /// to lint for overlapping ranges.
    pub(super) fn split<'p>(
        &self,
        pcx: PatCtxt<'_, 'p, 'tcx>,
        hir_id: Option<HirId>,
    ) -> SmallVec<[Self; 1]> {
        debug!("Constructor::split({:#?}, {:#?})", self, pcx.matrix);

        match self {
            Wildcard => Constructor::split_wildcard(pcx),
            // Fast-track if the range is trivial. In particular, we don't do the overlapping
            // ranges check.
            IntRange(ctor_range) if !ctor_range.is_singleton() => ctor_range.split(pcx, hir_id),
            Slice(slice @ Slice { kind: VarLen(..), .. }) => slice.split(pcx),
            // Any other constructor can be used unchanged.
            _ => smallvec![self.clone()],
        }
    }

    /// For wildcards, there are two groups of constructors: there are the constructors actually
    /// present in the matrix (`head_ctors`), and the constructors not present (`missing_ctors`).
    /// Two constructors that are not in the matrix will either both be caught (by a wildcard), or
    /// both not be caught. Therefore we can keep the missing constructors grouped together.
    fn split_wildcard<'p>(pcx: PatCtxt<'_, 'p, 'tcx>) -> SmallVec<[Self; 1]> {
        // Missing constructors are those that are not matched by any non-wildcard patterns in the
        // current column. We only fully construct them on-demand, because they're rarely used and
        // can be big.
        let missing_ctors = MissingConstructors::new(pcx);
        if missing_ctors.is_empty(pcx) {
            // All the constructors are present in the matrix, so we just go through them all.
            // We must also split them first.
            missing_ctors.all_ctors
        } else {
            // Some constructors are missing, thus we can specialize with the wildcard constructor,
            // which will stand for those constructors that are missing, and behaves like any of
            // them.
            smallvec![Wildcard]
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
            // Wildcards are only covered by wildcards
            (Wildcard, _) => false,

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
            Str(..) | FloatRange(..) | Opaque | Wildcard => {
                span_bug!(pcx.span, "found unexpected ctor in all_ctors: {:?}", self)
            }
        }
    }
}

/// This determines the set of all possible constructors of a pattern matching
/// values of type `left_ty`. For vectors, this would normally be an infinite set
/// but is instead bounded by the maximum fixed length of slice patterns in
/// the column of patterns being analyzed.
///
/// We make sure to omit constructors that are statically impossible. E.g., for
/// `Option<!>`, we do not include `Some(_)` in the returned list of constructors.
/// Invariant: this returns an empty `Vec` if and only if the type is uninhabited (as determined by
/// `cx.is_uninhabited()`).
fn all_constructors<'p, 'tcx>(pcx: PatCtxt<'_, 'p, 'tcx>) -> Vec<Constructor<'tcx>> {
    debug!("all_constructors({:?})", pcx.ty);
    let cx = pcx.cx;
    let make_range = |start, end| {
        IntRange(
            // `unwrap()` is ok because we know the type is an integer.
            IntRange::from_range(cx.tcx, start, end, pcx.ty, &RangeEnd::Included).unwrap(),
        )
    };
    match pcx.ty.kind() {
        ty::Bool => vec![make_range(0, 1)],
        ty::Array(sub_ty, len) if len.try_eval_usize(cx.tcx, cx.param_env).is_some() => {
            let len = len.eval_usize(cx.tcx, cx.param_env);
            if len != 0 && cx.is_uninhabited(sub_ty) {
                vec![]
            } else {
                vec![Slice(Slice::new(Some(len), VarLen(0, 0)))]
            }
        }
        // Treat arrays of a constant but unknown length like slices.
        ty::Array(sub_ty, _) | ty::Slice(sub_ty) => {
            let kind = if cx.is_uninhabited(sub_ty) { FixedLen(0) } else { VarLen(0, 0) };
            vec![Slice(Slice::new(None, kind))]
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
                vec![NonExhaustive]
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
            vec![
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
            vec![NonExhaustive]
        }
        &ty::Int(ity) => {
            let bits = Integer::from_attr(&cx.tcx, SignedInt(ity)).size().bits() as u128;
            let min = 1u128 << (bits - 1);
            let max = min - 1;
            vec![make_range(min, max)]
        }
        &ty::Uint(uty) => {
            let size = Integer::from_attr(&cx.tcx, UnsignedInt(uty)).size();
            let max = size.truncate(u128::MAX);
            vec![make_range(0, max)]
        }
        // If `exhaustive_patterns` is disabled and our scrutinee is the never type, we cannot
        // expose its emptiness. The exception is if the pattern is at the top level, because we
        // want empty matches to be considered exhaustive.
        ty::Never if !cx.tcx.features().exhaustive_patterns && !pcx.is_top_level => {
            vec![NonExhaustive]
        }
        ty::Never => vec![],
        _ if cx.is_uninhabited(pcx.ty) => vec![],
        ty::Adt(..) | ty::Tuple(..) | ty::Ref(..) => vec![Single],
        // This type is one for which we cannot list constructors, like `str` or `f64`.
        _ => vec![NonExhaustive],
    }
}

// A struct to compute a set of constructors equivalent to `all_ctors \ used_ctors`.
#[derive(Debug)]
pub(super) struct MissingConstructors<'tcx> {
    all_ctors: SmallVec<[Constructor<'tcx>; 1]>,
    used_ctors: Vec<Constructor<'tcx>>,
}

impl<'tcx> MissingConstructors<'tcx> {
    pub(super) fn new<'p>(pcx: PatCtxt<'_, 'p, 'tcx>) -> Self {
        let used_ctors: Vec<Constructor<'_>> =
            pcx.matrix.head_ctors(pcx.cx).cloned().filter(|c| !c.is_wildcard()).collect();
        // Since `all_ctors` never contains wildcards, this won't recurse further.
        let all_ctors =
            all_constructors(pcx).into_iter().flat_map(|ctor| ctor.split(pcx, None)).collect();

        MissingConstructors { all_ctors, used_ctors }
    }

    fn is_empty<'p>(&self, pcx: PatCtxt<'_, 'p, 'tcx>) -> bool {
        self.iter(pcx).next().is_none()
    }

    /// Iterate over all_ctors \ used_ctors
    fn iter<'a, 'p>(
        &'a self,
        pcx: PatCtxt<'a, 'p, 'tcx>,
    ) -> impl Iterator<Item = &'a Constructor<'tcx>> + Captures<'p> {
        self.all_ctors.iter().filter(move |ctor| !ctor.is_covered_by_any(pcx, &self.used_ctors))
    }

    /// List the patterns corresponding to the missing constructors. In some cases, instead of
    /// listing all constructors of a given type, we prefer to simply report a wildcard.
    pub(super) fn report_patterns<'p>(
        &self,
        pcx: PatCtxt<'_, 'p, 'tcx>,
    ) -> SmallVec<[Pat<'tcx>; 1]> {
        // There are 2 ways we can report a witness here.
        // Commonly, we can report all the "free"
        // constructors as witnesses, e.g., if we have:
        //
        // ```
        //     enum Direction { N, S, E, W }
        //     let Direction::N = ...;
        // ```
        //
        // we can report 3 witnesses: `S`, `E`, and `W`.
        //
        // However, there is a case where we don't want
        // to do this and instead report a single `_` witness:
        // if the user didn't actually specify a constructor
        // in this arm, e.g., in
        //
        // ```
        //     let x: (Direction, Direction, bool) = ...;
        //     let (_, _, false) = x;
        // ```
        //
        // we don't want to show all 16 possible witnesses
        // `(<direction-1>, <direction-2>, true)` - we are
        // satisfied with `(_, _, true)`. In this case,
        // `used_ctors` is empty.
        // The exception is: if we are at the top-level, for example in an empty match, we
        // sometimes prefer reporting the list of constructors instead of just `_`.
        let report_when_all_missing = pcx.is_top_level && !IntRange::is_integral(pcx.ty);
        if self.used_ctors.is_empty() && !report_when_all_missing {
            // All constructors are unused. Report only a wildcard
            // rather than each individual constructor.
            smallvec![Pat::wildcard_from_ty(pcx.ty)]
        } else {
            // Construct for each missing constructor a "wild" version of this
            // constructor, that matches everything that can be built with
            // it. For example, if `ctor` is a `Constructor::Variant` for
            // `Option::Some`, we get the pattern `Some(_)`.
            self.iter(pcx)
                .map(|missing_ctor| Fields::wildcards(pcx, &missing_ctor).apply(pcx, missing_ctor))
                .collect()
        }
    }
}

/// Some fields need to be explicitly hidden away in certain cases; see the comment above the
/// `Fields` struct. This struct represents such a potentially-hidden field. When a field is hidden
/// we still keep its type around.
#[derive(Debug, Copy, Clone)]
pub(super) enum FilteredField<'p, 'tcx> {
    Kept(&'p Pat<'tcx>),
    Hidden(Ty<'tcx>),
}

impl<'p, 'tcx> FilteredField<'p, 'tcx> {
    fn kept(self) -> Option<&'p Pat<'tcx>> {
        match self {
            FilteredField::Kept(p) => Some(p),
            FilteredField::Hidden(_) => None,
        }
    }

    fn to_pattern(self) -> Pat<'tcx> {
        match self {
            FilteredField::Kept(p) => p.clone(),
            FilteredField::Hidden(ty) => Pat::wildcard_from_ty(ty),
        }
    }
}

/// A value can be decomposed into a constructor applied to some fields. This struct represents
/// those fields, generalized to allow patterns in each field. See also `Constructor`.
///
/// If a private or `non_exhaustive` field is uninhabited, the code mustn't observe that it is
/// uninhabited. For that, we filter these fields out of the matrix. This is subtle because we
/// still need to have those fields back when going to/from a `Pat`. Most of this is handled
/// automatically in `Fields`, but when constructing or deconstructing `Fields` you need to be
/// careful. As a rule, when going to/from the matrix, use the filtered field list; when going
/// to/from `Pat`, use the full field list.
/// This filtering is uncommon in practice, because uninhabited fields are rarely used, so we avoid
/// it when possible to preserve performance.
#[derive(Debug, Clone)]
pub(super) enum Fields<'p, 'tcx> {
    /// Lists of patterns that don't contain any filtered fields.
    /// `Slice` and `Vec` behave the same; the difference is only to avoid allocating and
    /// triple-dereferences when possible. Frankly this is premature optimization, I (Nadrieril)
    /// have not measured if it really made a difference.
    Slice(&'p [Pat<'tcx>]),
    Vec(SmallVec<[&'p Pat<'tcx>; 2]>),
    /// Patterns where some of the fields need to be hidden. `kept_count` caches the number of
    /// non-hidden fields.
    Filtered {
        fields: SmallVec<[FilteredField<'p, 'tcx>; 2]>,
        kept_count: usize,
    },
}

impl<'p, 'tcx> Fields<'p, 'tcx> {
    fn empty() -> Self {
        Fields::Slice(&[])
    }

    /// Construct a new `Fields` from the given pattern. Must not be used if the pattern is a field
    /// of a struct/tuple/variant.
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
                            let mut kept_count = 0;
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
                                        FilteredField::Hidden(ty)
                                    } else {
                                        kept_count += 1;
                                        FilteredField::Kept(wildcard_from_ty(ty))
                                    }
                                })
                                .collect();
                            Fields::Filtered { fields, kept_count }
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
            Str(..) | FloatRange(..) | IntRange(..) | NonExhaustive | Opaque | Wildcard => {
                Fields::empty()
            }
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
        let mut subpatterns = self.all_patterns();

        let pat = match ctor {
            Single | Variant(_) => match pcx.ty.kind() {
                ty::Adt(..) | ty::Tuple(..) => {
                    let subpatterns = subpatterns
                        .enumerate()
                        .map(|(i, p)| FieldPat { field: Field::new(i), pattern: p })
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
            Opaque => bug!("we should not try to apply an opaque constructor"),
            Wildcard => bug!(
                "trying to apply a wildcard constructor; this should have been done in `apply_constructors`"
            ),
        };

        Pat { ty: pcx.ty, span: DUMMY_SP, kind: Box::new(pat) }
    }

    /// Returns the number of patterns from the viewpoint of match-checking, i.e. excluding hidden
    /// fields. This is what we want in most cases in this file, the only exception being
    /// conversion to/from `Pat`.
    pub(super) fn len(&self) -> usize {
        match self {
            Fields::Slice(pats) => pats.len(),
            Fields::Vec(pats) => pats.len(),
            Fields::Filtered { kept_count, .. } => *kept_count,
        }
    }

    /// Returns the complete list of patterns, including hidden fields.
    fn all_patterns(self) -> impl Iterator<Item = Pat<'tcx>> {
        let pats: SmallVec<[_; 2]> = match self {
            Fields::Slice(pats) => pats.iter().cloned().collect(),
            Fields::Vec(pats) => pats.into_iter().cloned().collect(),
            Fields::Filtered { fields, .. } => {
                // We don't skip any fields here.
                fields.into_iter().map(|p| p.to_pattern()).collect()
            }
        };
        pats.into_iter()
    }

    /// Returns the filtered list of patterns, not including hidden fields.
    pub(super) fn filtered_patterns(self) -> SmallVec<[&'p Pat<'tcx>; 2]> {
        match self {
            Fields::Slice(pats) => pats.iter().collect(),
            Fields::Vec(pats) => pats,
            Fields::Filtered { fields, .. } => {
                // We skip hidden fields here
                fields.into_iter().filter_map(|p| p.kept()).collect()
            }
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
    /// defines some fields but not all, for example `Foo { field1: Some(_), .. }`: here we start with a
    /// `Fields` that is just one wildcard per field of the `Foo` struct, and override the entry
    /// corresponding to `field1` with the pattern `Some(_)`. This is also used for slice patterns
    /// for the same reason.
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

    /// Replaces contained fields with the given filtered list of patterns, e.g. taken from the
    /// matrix. There must be `len()` patterns in `pats`.
    pub(super) fn replace_fields(
        &self,
        cx: &MatchCheckCtxt<'p, 'tcx>,
        pats: impl IntoIterator<Item = Pat<'tcx>>,
    ) -> Self {
        let pats: &[_] = cx.pattern_arena.alloc_from_iter(pats);

        match self {
            Fields::Filtered { fields, kept_count } => {
                let mut pats = pats.iter();
                let mut fields = fields.clone();
                for f in &mut fields {
                    if let FilteredField::Kept(p) = f {
                        // We take one input pattern for each `Kept` field, in order.
                        *p = pats.next().unwrap();
                    }
                }
                Fields::Filtered { fields, kept_count: *kept_count }
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
