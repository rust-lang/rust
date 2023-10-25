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
//! ```compile_fail,E0004
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
//! wildcards, see [`Constructor::split`]; for integer ranges, see
//! [`IntRange::split`]; for slices, see [`Slice::split`].

use std::cell::Cell;
use std::cmp::{self, max, min, Ordering};
use std::fmt;
use std::iter::once;
use std::ops::RangeInclusive;

use smallvec::{smallvec, SmallVec};

use rustc_apfloat::ieee::{DoubleS, IeeeFloat, SingleS};
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::{HirId, RangeEnd};
use rustc_index::Idx;
use rustc_middle::middle::stability::EvalResult;
use rustc_middle::mir;
use rustc_middle::thir::{FieldPat, Pat, PatKind, PatRange};
use rustc_middle::ty::layout::IntegerExt;
use rustc_middle::ty::{self, Ty, TyCtxt, VariantDef};
use rustc_session::lint;
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::{FieldIdx, Integer, VariantIdx, FIRST_VARIANT};

use self::Constructor::*;
use self::SliceKind::*;

use super::usefulness::{MatchCheckCtxt, PatCtxt};
use crate::errors::{Overlap, OverlappingRangeEndpoints};

/// Recursively expand this pattern into its subpatterns. Only useful for or-patterns.
fn expand_or_pat<'p, 'tcx>(pat: &'p Pat<'tcx>) -> Vec<&'p Pat<'tcx>> {
    fn expand<'p, 'tcx>(pat: &'p Pat<'tcx>, vec: &mut Vec<&'p Pat<'tcx>>) {
        if let PatKind::Or { pats } = &pat.kind {
            for pat in pats.iter() {
                expand(&pat, vec);
            }
        } else {
            vec.push(pat)
        }
    }

    let mut pats = Vec::new();
    expand(pat, &mut pats);
    pats
}

/// Whether we have seen a constructor in the column or not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Presence {
    Unseen,
    Seen,
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
pub(crate) struct IntRange {
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
    fn from_bits<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, bits: u128) -> IntRange {
        let bias = IntRange::signed_bias(tcx, ty);
        // Perform a shift if the underlying types are signed,
        // which makes the interval arithmetic simpler.
        let val = bits ^ bias;
        IntRange { range: val..=val }
    }

    #[inline]
    fn from_range<'tcx>(
        tcx: TyCtxt<'tcx>,
        lo: u128,
        hi: u128,
        ty: Ty<'tcx>,
        end: RangeEnd,
    ) -> IntRange {
        // Perform a shift if the underlying types are signed,
        // which makes the interval arithmetic simpler.
        let bias = IntRange::signed_bias(tcx, ty);
        let (lo, hi) = (lo ^ bias, hi ^ bias);
        let offset = (end == RangeEnd::Excluded) as u128;
        if lo > hi || (lo == hi && end == RangeEnd::Excluded) {
            // This should have been caught earlier by E0030.
            bug!("malformed range pattern: {}..={}", lo, (hi - offset));
        }
        IntRange { range: lo..=(hi - offset) }
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

    /// Partition a range of integers into disjoint subranges. This does constructor splitting for
    /// integer ranges as explained at the top of the file.
    ///
    /// This returns an output that covers `self`. The output is split so that the only
    /// intersections between an output range and a column range are inclusions. No output range
    /// straddles the boundary of one of the inputs.
    ///
    /// Additionally, we track for each output range whether it is covered by one of the column ranges or not.
    ///
    /// The following input:
    /// ```text
    ///   (--------------------------) // `self`
    /// (------) (----------)    (-)
    ///     (------) (--------)
    /// ```
    /// is first intersected with `self`:
    /// ```text
    ///   (--------------------------) // `self`
    ///   (----) (----------)    (-)
    ///     (------) (--------)
    /// ```
    /// and then iterated over as follows:
    /// ```text
    ///   (-(--)-(-)-(------)-)--(-)-
    /// ```
    /// where each sequence of dashes is an output range, and dashes outside parentheses are marked
    /// as `Presence::Missing`.
    fn split(
        &self,
        column_ranges: impl Iterator<Item = IntRange>,
    ) -> impl Iterator<Item = (Presence, IntRange)> {
        /// Represents a boundary between 2 integers. Because the intervals spanning boundaries must be
        /// able to cover every integer, we need to be able to represent 2^128 + 1 such boundaries.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        enum IntBoundary {
            JustBefore(u128),
            AfterMax,
        }

        fn unpack_intrange(range: IntRange) -> [IntBoundary; 2] {
            use IntBoundary::*;
            let (lo, hi) = range.boundaries();
            let lo = JustBefore(lo);
            let hi = match hi.checked_add(1) {
                Some(m) => JustBefore(m),
                None => AfterMax,
            };
            [lo, hi]
        }

        // The boundaries of ranges in `column_ranges` intersected with `self`.
        // We do parenthesis matching for input ranges. A boundary counts as +1 if it starts
        // a range and -1 if it ends it. When the count is > 0 between two boundaries, we
        // are within an input range.
        let mut boundaries: Vec<(IntBoundary, isize)> = column_ranges
            .filter_map(|r| self.intersection(&r))
            .map(unpack_intrange)
            .flat_map(|[lo, hi]| [(lo, 1), (hi, -1)])
            .collect();
        // We sort by boundary, and for each boundary we sort the "closing parentheses" first. The
        // order of +1/-1 for a same boundary value is actually irrelevant, because we only look at
        // the accumulated count between distinct boundary values.
        boundaries.sort_unstable();

        let [self_start, self_end] = unpack_intrange(self.clone());
        // Accumulate parenthesis counts.
        let mut paren_counter = 0isize;
        // Gather pairs of adjacent boundaries.
        let mut prev_bdy = self_start;
        boundaries
            .into_iter()
            // End with the end of the range. The count is ignored.
            .chain(once((self_end, 0)))
            // List pairs of adjacent boundaries and the count between them.
            .map(move |(bdy, delta)| {
                // `delta` affects the count as we cross `bdy`, so the relevant count between
                // `prev_bdy` and `bdy` is untouched by `delta`.
                let ret = (prev_bdy, paren_counter, bdy);
                prev_bdy = bdy;
                paren_counter += delta;
                ret
            })
            // Skip empty ranges.
            .filter(|&(prev_bdy, _, bdy)| prev_bdy != bdy)
            // Convert back to ranges.
            .map(move |(prev_bdy, paren_count, bdy)| {
                use IntBoundary::*;
                use Presence::*;
                let presence = if paren_count > 0 { Seen } else { Unseen };
                let range = match (prev_bdy, bdy) {
                    (JustBefore(n), JustBefore(m)) if n < m => n..=(m - 1),
                    (JustBefore(n), AfterMax) => n..=u128::MAX,
                    _ => unreachable!(), // Ruled out by the sorting and filtering we did
                };
                (presence, IntRange { range })
            })
    }

    /// Only used for displaying the range.
    fn to_pat<'tcx>(&self, tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Pat<'tcx> {
        let (lo, hi) = self.boundaries();

        let bias = IntRange::signed_bias(tcx, ty);
        let (lo, hi) = (lo ^ bias, hi ^ bias);

        let env = ty::ParamEnv::empty().and(ty);
        let lo_const = mir::Const::from_bits(tcx, lo, env);
        let hi_const = mir::Const::from_bits(tcx, hi, env);

        let kind = if lo == hi {
            PatKind::Constant { value: lo_const }
        } else {
            PatKind::Range(Box::new(PatRange {
                lo: lo_const,
                hi: hi_const,
                end: RangeEnd::Included,
            }))
        };

        Pat { ty, span: DUMMY_SP, kind }
    }

    /// Lint on likely incorrect range patterns (#63987)
    pub(super) fn lint_overlapping_range_endpoints<'a, 'p: 'a, 'tcx: 'a>(
        &self,
        pcx: &PatCtxt<'_, 'p, 'tcx>,
        pats: impl Iterator<Item = &'a DeconstructedPat<'p, 'tcx>>,
        column_count: usize,
        lint_root: HirId,
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

        let overlap: Vec<_> = pats
            .filter_map(|pat| Some((pat.ctor().as_int_range()?, pat.span())))
            .filter(|(range, _)| self.suspicious_intersection(range))
            .map(|(range, span)| Overlap {
                range: self.intersection(&range).unwrap().to_pat(pcx.cx.tcx, pcx.ty),
                span,
            })
            .collect();

        if !overlap.is_empty() {
            pcx.cx.tcx.emit_spanned_lint(
                lint::builtin::OVERLAPPING_RANGE_ENDPOINTS,
                lint_root,
                pcx.span,
                OverlappingRangeEndpoints { overlap, range: pcx.span },
            );
        }
    }
}

/// Note: this is often not what we want: e.g. `false` is converted into the range `0..=0` and
/// would be displayed as such. To render properly, convert to a pattern first.
impl fmt::Debug for IntRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (lo, hi) = self.boundaries();
        write!(f, "{lo}")?;
        write!(f, "{}", RangeEnd::Included)?;
        write!(f, "{hi}")
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

    /// This computes constructor splitting for variable-length slices, as explained at the top of
    /// the file.
    ///
    /// A slice pattern `[x, .., y]` behaves like the infinite or-pattern `[x, y] | [x, _, y] | [x,
    /// _, _, y] | etc`. The corresponding value constructors are fixed-length array constructors of
    /// corresponding lengths. We obviously can't list this infinitude of constructors.
    /// Thankfully, it turns out that for each finite set of slice patterns, all sufficiently large
    /// array lengths are equivalent.
    ///
    /// Let's look at an example, where we are trying to split the last pattern:
    /// ```
    /// # fn foo(x: &[bool]) {
    /// match x {
    ///     [true, true, ..] => {}
    ///     [.., false, false] => {}
    ///     [..] => {}
    /// }
    /// # }
    /// ```
    /// Here are the results of specialization for the first few lengths:
    /// ```
    /// # fn foo(x: &[bool]) { match x {
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
    /// # _ => {}
    /// # }}
    /// ```
    ///
    /// We see that above length 4, we are simply inserting columns full of wildcards in the middle.
    /// This means that specialization and witness computation with slices of length `l >= 4` will
    /// give equivalent results regardless of `l`. This applies to any set of slice patterns: there
    /// will be a length `L` above which all lengths behave the same. This is exactly what we need
    /// for constructor splitting.
    ///
    /// A variable-length slice pattern covers all lengths from its arity up to infinity. As we just
    /// saw, we can split this in two: lengths below `L` are treated individually with a
    /// fixed-length slice each; lengths above `L` are grouped into a single variable-length slice
    /// constructor.
    ///
    /// For each variable-length slice pattern `p` with a prefix of length `plₚ` and suffix of
    /// length `slₚ`, only the first `plₚ` and the last `slₚ` elements are examined. Therefore, as
    /// long as `L` is positive (to avoid concerns about empty types), all elements after the
    /// maximum prefix length and before the maximum suffix length are not examined by any
    /// variable-length pattern, and therefore can be ignored. This gives us a way to compute `L`.
    ///
    /// Additionally, if fixed-length patterns exist, we must pick an `L` large enough to miss them,
    /// so we can pick `L = max(max(FIXED_LEN)+1, max(PREFIX_LEN) + max(SUFFIX_LEN))`.
    /// `max_slice` below will be made to have this arity `L`.
    ///
    /// If `self` is fixed-length, it is returned as-is.
    ///
    /// Additionally, we track for each output slice whether it is covered by one of the column slices or not.
    fn split(
        self,
        column_slices: impl Iterator<Item = Slice>,
    ) -> impl Iterator<Item = (Presence, Slice)> {
        // Range of lengths below `L`.
        let smaller_lengths;
        let arity = self.arity();
        let mut max_slice = self.kind;
        // Tracks the smallest variable-length slice we've seen. Any slice arity above it is
        // therefore `Presence::Seen` in the column.
        let mut min_var_len = usize::MAX;
        // Tracks the fixed-length slices we've seen, to mark them as `Presence::Seen`.
        let mut seen_fixed_lens = FxHashSet::default();
        match &mut max_slice {
            VarLen(max_prefix_len, max_suffix_len) => {
                // We grow `max_slice` to be larger than all slices encountered, as described above.
                // For diagnostics, we keep the prefix and suffix lengths separate, but grow them so that
                // `L = max_prefix_len + max_suffix_len`.
                let mut max_fixed_len = 0;
                for slice in column_slices {
                    match slice.kind {
                        FixedLen(len) => {
                            max_fixed_len = cmp::max(max_fixed_len, len);
                            if arity <= len {
                                seen_fixed_lens.insert(len);
                            }
                        }
                        VarLen(prefix, suffix) => {
                            *max_prefix_len = cmp::max(*max_prefix_len, prefix);
                            *max_suffix_len = cmp::max(*max_suffix_len, suffix);
                            min_var_len = cmp::min(min_var_len, prefix + suffix);
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
                    Some(len) if max_slice.arity() >= len => max_slice = FixedLen(len),
                    _ => {}
                }

                smaller_lengths = match self.array_len {
                    // The only admissible fixed-length slice is one of the array size. Whether `max_slice`
                    // is fixed-length or variable-length, it will be the only relevant slice to output
                    // here.
                    Some(_) => 0..0, // empty range
                    // We need to cover all arities in the range `(arity..infinity)`. We split that
                    // range into two: lengths smaller than `max_slice.arity()` are treated
                    // independently as fixed-lengths slices, and lengths above are captured by
                    // `max_slice`.
                    None => self.arity()..max_slice.arity(),
                };
            }
            FixedLen(_) => {
                // No need to split here. We only track presence.
                for slice in column_slices {
                    match slice.kind {
                        FixedLen(len) => {
                            if len == arity {
                                seen_fixed_lens.insert(len);
                            }
                        }
                        VarLen(prefix, suffix) => {
                            min_var_len = cmp::min(min_var_len, prefix + suffix);
                        }
                    }
                }
                smaller_lengths = 0..0;
            }
        };

        smaller_lengths.map(FixedLen).chain(once(max_slice)).map(move |kind| {
            let arity = kind.arity();
            let seen = if min_var_len <= arity || seen_fixed_lens.contains(&arity) {
                Presence::Seen
            } else {
                Presence::Unseen
            };
            (seen, Slice::new(self.array_len, kind))
        })
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
    F32Range(IeeeFloat<SingleS>, IeeeFloat<SingleS>, RangeEnd),
    F64Range(IeeeFloat<DoubleS>, IeeeFloat<DoubleS>, RangeEnd),
    /// String literals. Strings are not quite the same as `&[u8]` so we treat them separately.
    Str(mir::Const<'tcx>),
    /// Array and slice patterns.
    Slice(Slice),
    /// Constants that must not be matched structurally. They are treated as black
    /// boxes for the purposes of exhaustiveness: we must not inspect them, and they
    /// don't count towards making a match exhaustive.
    Opaque,
    /// Or-pattern.
    Or,
    /// Wildcard pattern.
    Wildcard,
    /// Fake extra constructor for enums that aren't allowed to be matched exhaustively. Also used
    /// for those types for which we cannot list constructors explicitly, like `f64` and `str`.
    NonExhaustive,
    /// Fake extra constructor for variants that should not be mentioned in diagnostics.
    /// We use this for variants behind an unstable gate as well as
    /// `#[doc(hidden)]` ones.
    Hidden,
    /// Fake extra constructor for constructors that are not seen in the matrix, as explained in the
    /// code for [`Constructor::split`].
    Missing,
}

impl<'tcx> Constructor<'tcx> {
    pub(super) fn is_non_exhaustive(&self) -> bool {
        matches!(self, NonExhaustive)
    }

    pub(super) fn as_variant(&self) -> Option<VariantIdx> {
        match self {
            Variant(i) => Some(*i),
            _ => None,
        }
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

    fn variant_index_for_adt(&self, adt: ty::AdtDef<'tcx>) -> VariantIdx {
        match *self {
            Variant(idx) => idx,
            Single => {
                assert!(!adt.is_enum());
                FIRST_VARIANT
            }
            _ => bug!("bad constructor {:?} for adt {:?}", self, adt),
        }
    }

    /// The number of fields for this constructor. This must be kept in sync with
    /// `Fields::wildcards`.
    pub(super) fn arity(&self, pcx: &PatCtxt<'_, '_, 'tcx>) -> usize {
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
                        let variant = &adt.variant(self.variant_index_for_adt(*adt));
                        Fields::list_variant_nonhidden_fields(pcx.cx, pcx.ty, variant).count()
                    }
                }
                _ => bug!("Unexpected type for `Single` constructor: {:?}", pcx.ty),
            },
            Slice(slice) => slice.arity(),
            Str(..)
            | F32Range(..)
            | F64Range(..)
            | IntRange(..)
            | Opaque
            | NonExhaustive
            | Hidden
            | Missing { .. }
            | Wildcard => 0,
            Or => bug!("The `Or` constructor doesn't have a fixed arity"),
        }
    }

    /// Some constructors (namely `Wildcard`, `IntRange` and `Slice`) actually stand for a set of
    /// actual constructors (like variants, integers or fixed-sized slices). When specializing for
    /// these constructors, we want to be specialising for the actual underlying constructors.
    /// Naively, we would simply return the list of constructors they correspond to. We instead are
    /// more clever: if there are constructors that we know will behave the same w.r.t. the current
    /// matrix, we keep them grouped. For example, all slices of a sufficiently large length will
    /// either be all useful or all non-useful with a given matrix.
    ///
    /// See the branches for details on how the splitting is done.
    ///
    /// This function may discard some irrelevant constructors if this preserves behavior. Eg. for
    /// the `_` case, we ignore the constructors already present in the column, unless all of them
    /// are.
    pub(super) fn split<'a>(
        &self,
        pcx: &PatCtxt<'_, '_, 'tcx>,
        ctors: impl Iterator<Item = &'a Constructor<'tcx>> + Clone,
    ) -> SmallVec<[Self; 1]>
    where
        'tcx: 'a,
    {
        match self {
            Wildcard => {
                let split_set = ConstructorSet::for_ty(pcx.cx, pcx.ty).split(pcx, ctors);
                if !split_set.missing.is_empty() {
                    // We are splitting a wildcard in order to compute its usefulness. Some constructors are
                    // not present in the column. The first thing we note is that specializing with any of
                    // the missing constructors would select exactly the rows with wildcards. Moreover, they
                    // would all return equivalent results. We can therefore group them all into a
                    // fictitious `Missing` constructor.
                    //
                    // As an important optimization, this function will skip all the present constructors.
                    // This is correct because specializing with any of the present constructors would
                    // select a strict superset of the wildcard rows, and thus would only find witnesses
                    // already found with the `Missing` constructor.
                    // This does mean that diagnostics are incomplete: in
                    // ```
                    // match x {
                    //   Some(true) => {}
                    // }
                    // ```
                    // we report `None` as missing but not `Some(false)`.
                    //
                    // When all the constructors are missing we can equivalently return the `Wildcard`
                    // constructor on its own. The difference between `Wildcard` and `Missing` will then
                    // only be in diagnostics.

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
                    // usually prefer to report the full list of constructors.
                    let all_missing = split_set.present.is_empty();
                    let report_when_all_missing =
                        pcx.is_top_level && !IntRange::is_integral(pcx.ty);
                    let ctor =
                        if all_missing && !report_when_all_missing { Wildcard } else { Missing };
                    smallvec![ctor]
                } else {
                    split_set.present
                }
            }
            // Fast-track if the range is trivial.
            IntRange(this_range) if !this_range.is_singleton() => {
                let column_ranges = ctors.filter_map(|ctor| ctor.as_int_range()).cloned();
                this_range.split(column_ranges).map(|(_, range)| IntRange(range)).collect()
            }
            Slice(this_slice @ Slice { kind: VarLen(..), .. }) => {
                let column_slices = ctors.filter_map(|c| c.as_slice());
                this_slice.split(column_slices).map(|(_, slice)| Slice(slice)).collect()
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
    pub(super) fn is_covered_by<'p>(&self, pcx: &PatCtxt<'_, 'p, 'tcx>, other: &Self) -> bool {
        // This must be kept in sync with `is_covered_by_any`.
        match (self, other) {
            // Wildcards cover anything
            (_, Wildcard) => true,
            // Only a wildcard pattern can match these special constructors.
            (Wildcard | Missing { .. } | NonExhaustive | Hidden, _) => false,

            (Single, Single) => true,
            (Variant(self_id), Variant(other_id)) => self_id == other_id,

            (IntRange(self_range), IntRange(other_range)) => self_range.is_subrange(other_range),
            (F32Range(self_from, self_to, self_end), F32Range(other_from, other_to, other_end)) => {
                self_from.ge(other_from)
                    && match self_to.partial_cmp(other_to) {
                        Some(Ordering::Less) => true,
                        Some(Ordering::Equal) => other_end == self_end,
                        _ => false,
                    }
            }
            (F64Range(self_from, self_to, self_end), F64Range(other_from, other_to, other_end)) => {
                self_from.ge(other_from)
                    && match self_to.partial_cmp(other_to) {
                        Some(Ordering::Less) => true,
                        Some(Ordering::Equal) => other_end == self_end,
                        _ => false,
                    }
            }
            (Str(self_val), Str(other_val)) => {
                // FIXME Once valtrees are available we can directly use the bytes
                // in the `Str` variant of the valtree for the comparison here.
                self_val == other_val
            }
            (Slice(self_slice), Slice(other_slice)) => self_slice.is_covered_by(*other_slice),

            // We are trying to inspect an opaque constant. Thus we skip the row.
            (Opaque, _) | (_, Opaque) => false,

            _ => span_bug!(
                pcx.span,
                "trying to compare incompatible constructors {:?} and {:?}",
                self,
                other
            ),
        }
    }
}

/// Describes the set of all constructors for a type.
#[derive(Debug)]
pub(super) enum ConstructorSet {
    /// The type has a single constructor, e.g. `&T` or a struct.
    Single,
    /// This type has the following list of constructors.
    /// Some variants are hidden, which means they won't be mentioned in diagnostics unless the user
    /// mentioned them first. We use this for variants behind an unstable gate as well as
    /// `#[doc(hidden)]` ones.
    Variants {
        visible_variants: Vec<VariantIdx>,
        hidden_variants: Vec<VariantIdx>,
        non_exhaustive: bool,
    },
    /// The type is spanned by integer values. The range or ranges give the set of allowed values.
    /// The second range is only useful for `char`.
    /// This is reused for bool. FIXME: don't.
    /// `non_exhaustive` is used when the range is not allowed to be matched exhaustively (that's
    /// for usize/isize).
    Integers { range_1: IntRange, range_2: Option<IntRange>, non_exhaustive: bool },
    /// The type is matched by slices. The usize is the compile-time length of the array, if known.
    Slice(Option<usize>),
    /// The type is matched by slices whose elements are uninhabited.
    SliceOfEmpty,
    /// The constructors cannot be listed, and the type cannot be matched exhaustively. E.g. `str`,
    /// floats.
    Unlistable,
    /// The type has no inhabitants.
    Uninhabited,
}

/// Describes the result of analyzing the constructors in a column of a match.
///
/// `present` is morally the set of constructors present in the column, and `missing` is the set of
/// constructors that exist in the type but are not present in the column.
///
/// More formally, they respect the following constraints:
/// - the union of `present` and `missing` covers the whole type
/// - `present` and `missing` are disjoint
/// - neither contains wildcards
/// - each constructor in `present` is covered by some non-wildcard constructor in the column
/// - together, the constructors in `present` cover all the non-wildcard constructor in the column
/// - non-wildcards in the column do no cover anything in `missing`
/// - constructors in `present` and `missing` are split for the column; in other words, they are
///     either fully included in or disjoint from each constructor in the column. This avoids
///     non-trivial intersections like between `0..10` and `5..15`.
#[derive(Debug)]
pub(super) struct SplitConstructorSet<'tcx> {
    pub(super) present: SmallVec<[Constructor<'tcx>; 1]>,
    pub(super) missing: Vec<Constructor<'tcx>>,
}

impl ConstructorSet {
    #[instrument(level = "debug", skip(cx), ret)]
    pub(super) fn for_ty<'p, 'tcx>(cx: &MatchCheckCtxt<'p, 'tcx>, ty: Ty<'tcx>) -> Self {
        let make_range =
            |start, end| IntRange::from_range(cx.tcx, start, end, ty, RangeEnd::Included);
        // This determines the set of all possible constructors for the type `ty`. For numbers,
        // arrays and slices we use ranges and variable-length slices when appropriate.
        //
        // If the `exhaustive_patterns` feature is enabled, we make sure to omit constructors that
        // are statically impossible. E.g., for `Option<!>`, we do not include `Some(_)` in the
        // returned list of constructors.
        // Invariant: this is `Uninhabited` if and only if the type is uninhabited (as determined by
        // `cx.is_uninhabited()`).
        match ty.kind() {
            ty::Bool => {
                Self::Integers { range_1: make_range(0, 1), range_2: None, non_exhaustive: false }
            }
            ty::Char => {
                // The valid Unicode Scalar Value ranges.
                Self::Integers {
                    range_1: make_range('\u{0000}' as u128, '\u{D7FF}' as u128),
                    range_2: Some(make_range('\u{E000}' as u128, '\u{10FFFF}' as u128)),
                    non_exhaustive: false,
                }
            }
            &ty::Int(ity) => {
                // `usize`/`isize` are not allowed to be matched exhaustively unless the
                // `precise_pointer_size_matching` feature is enabled.
                let non_exhaustive =
                    ty.is_ptr_sized_integral() && !cx.tcx.features().precise_pointer_size_matching;
                let bits = Integer::from_int_ty(&cx.tcx, ity).size().bits() as u128;
                let min = 1u128 << (bits - 1);
                let max = min - 1;
                Self::Integers { range_1: make_range(min, max), non_exhaustive, range_2: None }
            }
            &ty::Uint(uty) => {
                // `usize`/`isize` are not allowed to be matched exhaustively unless the
                // `precise_pointer_size_matching` feature is enabled.
                let non_exhaustive =
                    ty.is_ptr_sized_integral() && !cx.tcx.features().precise_pointer_size_matching;
                let size = Integer::from_uint_ty(&cx.tcx, uty).size();
                let max = size.truncate(u128::MAX);
                Self::Integers { range_1: make_range(0, max), non_exhaustive, range_2: None }
            }
            ty::Array(sub_ty, len) if len.try_eval_target_usize(cx.tcx, cx.param_env).is_some() => {
                let len = len.eval_target_usize(cx.tcx, cx.param_env) as usize;
                if len != 0 && cx.is_uninhabited(*sub_ty) {
                    Self::Uninhabited
                } else {
                    Self::Slice(Some(len))
                }
            }
            // Treat arrays of a constant but unknown length like slices.
            ty::Array(sub_ty, _) | ty::Slice(sub_ty) => {
                if cx.is_uninhabited(*sub_ty) {
                    Self::SliceOfEmpty
                } else {
                    Self::Slice(None)
                }
            }
            ty::Adt(def, args) if def.is_enum() => {
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
                let is_declared_nonexhaustive = cx.is_foreign_non_exhaustive_enum(ty);

                if def.variants().is_empty() && !is_declared_nonexhaustive {
                    Self::Uninhabited
                } else {
                    let is_exhaustive_pat_feature = cx.tcx.features().exhaustive_patterns;
                    let (hidden_variants, visible_variants) = def
                        .variants()
                        .iter_enumerated()
                        .filter(|(_, v)| {
                            // If `exhaustive_patterns` is enabled, we exclude variants known to be
                            // uninhabited.
                            !is_exhaustive_pat_feature
                                || v.inhabited_predicate(cx.tcx, *def)
                                    .instantiate(cx.tcx, args)
                                    .apply(cx.tcx, cx.param_env, cx.module)
                        })
                        .map(|(idx, _)| idx)
                        .partition(|idx| {
                            let variant_def_id = def.variant(*idx).def_id;
                            // Filter variants that depend on a disabled unstable feature.
                            let is_unstable = matches!(
                                cx.tcx.eval_stability(variant_def_id, None, DUMMY_SP, None),
                                EvalResult::Deny { .. }
                            );
                            // Filter foreign `#[doc(hidden)]` variants.
                            let is_doc_hidden =
                                cx.tcx.is_doc_hidden(variant_def_id) && !variant_def_id.is_local();
                            is_unstable || is_doc_hidden
                        });

                    Self::Variants {
                        visible_variants,
                        hidden_variants,
                        non_exhaustive: is_declared_nonexhaustive,
                    }
                }
            }
            ty::Never => Self::Uninhabited,
            _ if cx.is_uninhabited(ty) => Self::Uninhabited,
            ty::Adt(..) | ty::Tuple(..) | ty::Ref(..) => Self::Single,
            // This type is one for which we cannot list constructors, like `str` or `f64`.
            _ => Self::Unlistable,
        }
    }

    /// This is the core logical operation of exhaustiveness checking. This analyzes a column a
    /// constructors to 1/ determine which constructors of the type (if any) are missing; 2/ split
    /// constructors to handle non-trivial intersections e.g. on ranges or slices.
    #[instrument(level = "debug", skip(self, pcx, ctors), ret)]
    pub(super) fn split<'a, 'tcx>(
        &self,
        pcx: &PatCtxt<'_, '_, 'tcx>,
        ctors: impl Iterator<Item = &'a Constructor<'tcx>> + Clone,
    ) -> SplitConstructorSet<'tcx>
    where
        'tcx: 'a,
    {
        let mut present: SmallVec<[_; 1]> = SmallVec::new();
        let mut missing = Vec::new();
        // Constructors in `ctors`, except wildcards.
        let mut seen = ctors.filter(|c| !(matches!(c, Opaque | Wildcard)));
        match self {
            ConstructorSet::Single => {
                if seen.next().is_none() {
                    missing.push(Single);
                } else {
                    present.push(Single);
                }
            }
            ConstructorSet::Variants { visible_variants, hidden_variants, non_exhaustive } => {
                let seen_set: FxHashSet<_> = seen.map(|c| c.as_variant().unwrap()).collect();
                let mut skipped_a_hidden_variant = false;

                for variant in visible_variants {
                    let ctor = Variant(*variant);
                    if seen_set.contains(&variant) {
                        present.push(ctor);
                    } else {
                        missing.push(ctor);
                    }
                }

                for variant in hidden_variants {
                    let ctor = Variant(*variant);
                    if seen_set.contains(&variant) {
                        present.push(ctor);
                    } else {
                        skipped_a_hidden_variant = true;
                    }
                }
                if skipped_a_hidden_variant {
                    missing.push(Hidden);
                }

                if *non_exhaustive {
                    missing.push(NonExhaustive);
                }
            }
            ConstructorSet::Integers { range_1, range_2, non_exhaustive } => {
                let seen_ranges: Vec<_> =
                    seen.map(|ctor| ctor.as_int_range().unwrap().clone()).collect();
                for (seen, splitted_range) in range_1.split(seen_ranges.iter().cloned()) {
                    match seen {
                        Presence::Unseen => missing.push(IntRange(splitted_range)),
                        Presence::Seen => present.push(IntRange(splitted_range)),
                    }
                }
                if let Some(range_2) = range_2 {
                    for (seen, splitted_range) in range_2.split(seen_ranges.into_iter()) {
                        match seen {
                            Presence::Unseen => missing.push(IntRange(splitted_range)),
                            Presence::Seen => present.push(IntRange(splitted_range)),
                        }
                    }
                }

                if *non_exhaustive {
                    missing.push(NonExhaustive);
                }
            }
            &ConstructorSet::Slice(array_len) => {
                let seen_slices = seen.map(|c| c.as_slice().unwrap());
                let base_slice = Slice::new(array_len, VarLen(0, 0));
                for (seen, splitted_slice) in base_slice.split(seen_slices) {
                    let ctor = Slice(splitted_slice);
                    match seen {
                        Presence::Unseen => missing.push(ctor),
                        Presence::Seen => present.push(ctor),
                    }
                }
            }
            ConstructorSet::SliceOfEmpty => {
                // This one is tricky because even though there's only one possible value of this
                // type (namely `[]`), slice patterns of all lengths are allowed, they're just
                // unreachable if length != 0.
                // We still gather the seen constructors in `present`, but the only slice that can
                // go in `missing` is `[]`.
                let seen_slices = seen.map(|c| c.as_slice().unwrap());
                let base_slice = Slice::new(None, VarLen(0, 0));
                for (seen, splitted_slice) in base_slice.split(seen_slices) {
                    let ctor = Slice(splitted_slice);
                    match seen {
                        Presence::Seen => present.push(ctor),
                        Presence::Unseen if splitted_slice.arity() == 0 => {
                            missing.push(Slice(Slice::new(None, FixedLen(0))))
                        }
                        Presence::Unseen => {}
                    }
                }
            }
            ConstructorSet::Unlistable => {
                // Since we can't list constructors, we take the ones in the column. This might list
                // some constructors several times but there's not much we can do.
                present.extend(seen.cloned());
                missing.push(NonExhaustive);
            }
            // If `exhaustive_patterns` is disabled and our scrutinee is an empty type, we cannot
            // expose its emptiness. The exception is if the pattern is at the top level, because we
            // want empty matches to be considered exhaustive.
            ConstructorSet::Uninhabited
                if !pcx.cx.tcx.features().exhaustive_patterns && !pcx.is_top_level =>
            {
                missing.push(NonExhaustive);
            }
            ConstructorSet::Uninhabited => {}
        }

        SplitConstructorSet { present, missing }
    }

    /// Compute the set of constructors missing from this column.
    /// This is only used for reporting to the user.
    pub(super) fn compute_missing<'a, 'tcx>(
        &self,
        pcx: &PatCtxt<'_, '_, 'tcx>,
        ctors: impl Iterator<Item = &'a Constructor<'tcx>> + Clone,
    ) -> Vec<Constructor<'tcx>>
    where
        'tcx: 'a,
    {
        self.split(pcx, ctors).missing
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
/// ```compile_fail,E0004
/// # fn foo() -> [Option<u8>; 4] { [None; 4] }
/// let x: [Option<u8>; 4] = foo();
/// match x {
///     [Some(0), ..] => {}
/// }
/// ```
///
/// Note that the number of fields of a constructor may not match the fields declared in the
/// original struct/variant. This happens if a private or `non_exhaustive` field is uninhabited,
/// because the code mustn't observe that it is uninhabited. In that case that field is not
/// included in `fields`. For that reason, when you have a `FieldIdx` you must use
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
        span: Span,
    ) -> Self {
        Fields::from_iter(cx, tys.into_iter().map(|ty| DeconstructedPat::wildcard(ty, span)))
    }

    // In the cases of either a `#[non_exhaustive]` field list or a non-public field, we hide
    // uninhabited fields in order not to reveal the uninhabitedness of the whole variant.
    // This lists the fields we keep along with their types.
    fn list_variant_nonhidden_fields<'a>(
        cx: &'a MatchCheckCtxt<'p, 'tcx>,
        ty: Ty<'tcx>,
        variant: &'a VariantDef,
    ) -> impl Iterator<Item = (FieldIdx, Ty<'tcx>)> + Captures<'a> + Captures<'p> {
        let ty::Adt(adt, args) = ty.kind() else { bug!() };
        // Whether we must not match the fields of this variant exhaustively.
        let is_non_exhaustive = variant.is_field_list_non_exhaustive() && !adt.did().is_local();

        variant.fields.iter().enumerate().filter_map(move |(i, field)| {
            let ty = field.ty(cx.tcx, args);
            // `field.ty()` doesn't normalize after substituting.
            let ty = cx.tcx.normalize_erasing_regions(cx.param_env, ty);
            let is_visible = adt.is_enum() || field.vis.is_accessible_from(cx.module, cx.tcx);
            let is_uninhabited = cx.is_uninhabited(ty);

            if is_uninhabited && (!is_visible || is_non_exhaustive) {
                None
            } else {
                Some((FieldIdx::new(i), ty))
            }
        })
    }

    /// Creates a new list of wildcard fields for a given constructor. The result must have a
    /// length of `constructor.arity()`.
    #[instrument(level = "trace")]
    pub(super) fn wildcards(pcx: &PatCtxt<'_, 'p, 'tcx>, constructor: &Constructor<'tcx>) -> Self {
        let ret = match constructor {
            Single | Variant(_) => match pcx.ty.kind() {
                ty::Tuple(fs) => Fields::wildcards_from_tys(pcx.cx, fs.iter(), pcx.span),
                ty::Ref(_, rty, _) => Fields::wildcards_from_tys(pcx.cx, once(*rty), pcx.span),
                ty::Adt(adt, args) => {
                    if adt.is_box() {
                        // The only legal patterns of type `Box` (outside `std`) are `_` and box
                        // patterns. If we're here we can assume this is a box pattern.
                        Fields::wildcards_from_tys(pcx.cx, once(args.type_at(0)), pcx.span)
                    } else {
                        let variant = &adt.variant(constructor.variant_index_for_adt(*adt));
                        let tys = Fields::list_variant_nonhidden_fields(pcx.cx, pcx.ty, variant)
                            .map(|(_, ty)| ty);
                        Fields::wildcards_from_tys(pcx.cx, tys, pcx.span)
                    }
                }
                _ => bug!("Unexpected type for `Single` constructor: {:?}", pcx),
            },
            Slice(slice) => match *pcx.ty.kind() {
                ty::Slice(ty) | ty::Array(ty, _) => {
                    let arity = slice.arity();
                    Fields::wildcards_from_tys(pcx.cx, (0..arity).map(|_| ty), pcx.span)
                }
                _ => bug!("bad slice pattern {:?} {:?}", constructor, pcx),
            },
            Str(..)
            | F32Range(..)
            | F64Range(..)
            | IntRange(..)
            | Opaque
            | NonExhaustive
            | Hidden
            | Missing { .. }
            | Wildcard => Fields::empty(),
            Or => {
                bug!("called `Fields::wildcards` on an `Or` ctor")
            }
        };
        debug!(?ret);
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
/// This also uses interior mutability to keep track of whether the pattern has been found reachable
/// during analysis. For this reason they cannot be cloned.
/// A `DeconstructedPat` will almost always come from user input; the only exception are some
/// `Wildcard`s introduced during specialization.
pub(crate) struct DeconstructedPat<'p, 'tcx> {
    ctor: Constructor<'tcx>,
    fields: Fields<'p, 'tcx>,
    ty: Ty<'tcx>,
    span: Span,
    reachable: Cell<bool>,
}

impl<'p, 'tcx> DeconstructedPat<'p, 'tcx> {
    pub(super) fn wildcard(ty: Ty<'tcx>, span: Span) -> Self {
        Self::new(Wildcard, Fields::empty(), ty, span)
    }

    pub(super) fn new(
        ctor: Constructor<'tcx>,
        fields: Fields<'p, 'tcx>,
        ty: Ty<'tcx>,
        span: Span,
    ) -> Self {
        DeconstructedPat { ctor, fields, ty, span, reachable: Cell::new(false) }
    }

    pub(crate) fn from_pat(cx: &MatchCheckCtxt<'p, 'tcx>, pat: &Pat<'tcx>) -> Self {
        let mkpat = |pat| DeconstructedPat::from_pat(cx, pat);
        let ctor;
        let fields;
        match &pat.kind {
            PatKind::AscribeUserType { subpattern, .. }
            | PatKind::InlineConstant { subpattern, .. } => return mkpat(subpattern),
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
                        let mut wilds: SmallVec<[_; 2]> =
                            fs.iter().map(|ty| DeconstructedPat::wildcard(ty, pat.span)).collect();
                        for pat in subpatterns {
                            wilds[pat.field.index()] = mkpat(&pat.pattern);
                        }
                        fields = Fields::from_iter(cx, wilds);
                    }
                    ty::Adt(adt, args) if adt.is_box() => {
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
                        let pattern = subpatterns.into_iter().find(|pat| pat.field.index() == 0);
                        let pat = if let Some(pat) = pattern {
                            mkpat(&pat.pattern)
                        } else {
                            DeconstructedPat::wildcard(args.type_at(0), pat.span)
                        };
                        ctor = Single;
                        fields = Fields::singleton(cx, pat);
                    }
                    ty::Adt(adt, _) => {
                        ctor = match pat.kind {
                            PatKind::Leaf { .. } => Single,
                            PatKind::Variant { variant_index, .. } => Variant(variant_index),
                            _ => bug!(),
                        };
                        let variant = &adt.variant(ctor.variant_index_for_adt(*adt));
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
                            tys.map(|ty| DeconstructedPat::wildcard(ty, pat.span)).collect();
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
                match pat.ty.kind() {
                    ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) => {
                        ctor = match value.try_eval_bits(cx.tcx, cx.param_env) {
                            Some(bits) => IntRange(IntRange::from_bits(cx.tcx, pat.ty, bits)),
                            None => Opaque,
                        };
                        fields = Fields::empty();
                    }
                    ty::Float(ty::FloatTy::F32) => {
                        ctor = match value.try_eval_bits(cx.tcx, cx.param_env) {
                            Some(bits) => {
                                use rustc_apfloat::Float;
                                let value = rustc_apfloat::ieee::Single::from_bits(bits);
                                F32Range(value, value, RangeEnd::Included)
                            }
                            None => Opaque,
                        };
                        fields = Fields::empty();
                    }
                    ty::Float(ty::FloatTy::F64) => {
                        ctor = match value.try_eval_bits(cx.tcx, cx.param_env) {
                            Some(bits) => {
                                use rustc_apfloat::Float;
                                let value = rustc_apfloat::ieee::Double::from_bits(bits);
                                F64Range(value, value, RangeEnd::Included)
                            }
                            None => Opaque,
                        };
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
            PatKind::Range(box PatRange { lo, hi, end }) => {
                use rustc_apfloat::Float;
                let ty = lo.ty();
                let lo = lo.try_eval_bits(cx.tcx, cx.param_env).unwrap();
                let hi = hi.try_eval_bits(cx.tcx, cx.param_env).unwrap();
                ctor = match ty.kind() {
                    ty::Char | ty::Int(_) | ty::Uint(_) => {
                        IntRange(IntRange::from_range(cx.tcx, lo, hi, ty, *end))
                    }
                    ty::Float(ty::FloatTy::F32) => {
                        let lo = rustc_apfloat::ieee::Single::from_bits(lo);
                        let hi = rustc_apfloat::ieee::Single::from_bits(hi);
                        F32Range(lo, hi, *end)
                    }
                    ty::Float(ty::FloatTy::F64) => {
                        let lo = rustc_apfloat::ieee::Double::from_bits(lo);
                        let hi = rustc_apfloat::ieee::Double::from_bits(hi);
                        F64Range(lo, hi, *end)
                    }
                    _ => bug!("invalid type for range pattern: {}", ty),
                };
                fields = Fields::empty();
            }
            PatKind::Array { prefix, slice, suffix } | PatKind::Slice { prefix, slice, suffix } => {
                let array_len = match pat.ty.kind() {
                    ty::Array(_, length) => {
                        Some(length.eval_target_usize(cx.tcx, cx.param_env) as usize)
                    }
                    ty::Slice(_) => None,
                    _ => span_bug!(pat.span, "bad ty {:?} for slice pattern", pat.ty),
                };
                let kind = if slice.is_some() {
                    VarLen(prefix.len(), suffix.len())
                } else {
                    FixedLen(prefix.len() + suffix.len())
                };
                ctor = Slice(Slice::new(array_len, kind));
                fields =
                    Fields::from_iter(cx, prefix.iter().chain(suffix.iter()).map(|p| mkpat(&*p)));
            }
            PatKind::Or { .. } => {
                ctor = Or;
                let pats = expand_or_pat(pat);
                fields = Fields::from_iter(cx, pats.into_iter().map(mkpat));
            }
            PatKind::Error(_) => {
                ctor = Opaque;
                fields = Fields::empty();
            }
        }
        DeconstructedPat::new(ctor, fields, pat.ty, pat.span)
    }

    pub(super) fn is_or_pat(&self) -> bool {
        matches!(self.ctor, Or)
    }
    pub(super) fn flatten_or_pat(&'p self) -> SmallVec<[&'p Self; 1]> {
        if self.is_or_pat() {
            self.iter_fields().flat_map(|p| p.flatten_or_pat()).collect()
        } else {
            smallvec![self]
        }
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
        pcx: &PatCtxt<'_, 'p, 'tcx>,
        other_ctor: &Constructor<'tcx>,
    ) -> SmallVec<[&'p DeconstructedPat<'p, 'tcx>; 2]> {
        match (&self.ctor, other_ctor) {
            (Wildcard, _) => {
                // We return a wildcard for each field of `other_ctor`.
                Fields::wildcards(pcx, other_ctor).iter_patterns().collect()
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
                        let (ty::Slice(inner_ty) | ty::Array(inner_ty, _)) = *self.ty.kind() else {
                            bug!("bad slice pattern {:?} {:?}", self.ctor, self.ty);
                        };
                        let prefix = &self.fields.fields[..prefix];
                        let suffix = &self.fields.fields[self_slice.arity() - suffix..];
                        let wildcard: &_ = pcx
                            .cx
                            .pattern_arena
                            .alloc(DeconstructedPat::wildcard(inner_ty, pcx.span));
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
                    write!(f, "box {subpattern:?}")
                }
                ty::Adt(..) | ty::Tuple(..) => {
                    let variant = match self.ty.kind() {
                        ty::Adt(adt, _) => Some(adt.variant(self.ctor.variant_index_for_adt(*adt))),
                        ty::Tuple(_) => None,
                        _ => unreachable!(),
                    };

                    if let Some(variant) = variant {
                        write!(f, "{}", variant.name)?;
                    }

                    // Without `cx`, we can't know which field corresponds to which, so we can't
                    // get the names of the fields. Instead we just display everything as a tuple
                    // struct, which should be good enough.
                    write!(f, "(")?;
                    for p in self.iter_fields() {
                        write!(f, "{}", start_or_comma())?;
                        write!(f, "{p:?}")?;
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
            F32Range(lo, hi, end) => write!(f, "{lo}{end}{hi}"),
            F64Range(lo, hi, end) => write!(f, "{lo}{end}{hi}"),
            IntRange(range) => write!(f, "{range:?}"), // Best-effort, will render e.g. `false` as `0..=0`
            Str(value) => write!(f, "{value}"),
            Opaque => write!(f, "<constant pattern>"),
            Or => {
                for pat in self.iter_fields() {
                    write!(f, "{}{:?}", start_or_continue(" | "), pat)?;
                }
                Ok(())
            }
            Wildcard | Missing { .. } | NonExhaustive | Hidden => write!(f, "_ : {:?}", self.ty),
        }
    }
}

/// Same idea as `DeconstructedPat`, except this is a fictitious pattern built up for diagnostics
/// purposes. As such they don't use interning and can be cloned.
#[derive(Debug, Clone)]
pub(crate) struct WitnessPat<'tcx> {
    ctor: Constructor<'tcx>,
    pub(crate) fields: Vec<WitnessPat<'tcx>>,
    ty: Ty<'tcx>,
}

impl<'tcx> WitnessPat<'tcx> {
    pub(super) fn new(ctor: Constructor<'tcx>, fields: Vec<Self>, ty: Ty<'tcx>) -> Self {
        Self { ctor, fields, ty }
    }
    pub(super) fn wildcard(ty: Ty<'tcx>) -> Self {
        Self::new(Wildcard, Vec::new(), ty)
    }

    /// Construct a pattern that matches everything that starts with this constructor.
    /// For example, if `ctor` is a `Constructor::Variant` for `Option::Some`, we get the pattern
    /// `Some(_)`.
    pub(super) fn wild_from_ctor(pcx: &PatCtxt<'_, '_, 'tcx>, ctor: Constructor<'tcx>) -> Self {
        // Reuse `Fields::wildcards` to get the types.
        let fields = Fields::wildcards(pcx, &ctor)
            .iter_patterns()
            .map(|deco_pat| Self::wildcard(deco_pat.ty()))
            .collect();
        Self::new(ctor, fields, pcx.ty)
    }

    pub(super) fn ctor(&self) -> &Constructor<'tcx> {
        &self.ctor
    }
    pub(super) fn ty(&self) -> Ty<'tcx> {
        self.ty
    }

    pub(crate) fn to_pat(&self, cx: &MatchCheckCtxt<'_, 'tcx>) -> Pat<'tcx> {
        let is_wildcard = |pat: &Pat<'_>| matches!(pat.kind, PatKind::Wild);
        let mut subpatterns = self.iter_fields().map(|p| Box::new(p.to_pat(cx)));
        let kind = match &self.ctor {
            Single | Variant(_) => match self.ty.kind() {
                ty::Tuple(..) => PatKind::Leaf {
                    subpatterns: subpatterns
                        .enumerate()
                        .map(|(i, pattern)| FieldPat { field: FieldIdx::new(i), pattern })
                        .collect(),
                },
                ty::Adt(adt_def, _) if adt_def.is_box() => {
                    // Without `box_patterns`, the only legal pattern of type `Box` is `_` (outside
                    // of `std`). So this branch is only reachable when the feature is enabled and
                    // the pattern is a box pattern.
                    PatKind::Deref { subpattern: subpatterns.next().unwrap() }
                }
                ty::Adt(adt_def, args) => {
                    let variant_index = self.ctor.variant_index_for_adt(*adt_def);
                    let variant = &adt_def.variant(variant_index);
                    let subpatterns = Fields::list_variant_nonhidden_fields(cx, self.ty, variant)
                        .zip(subpatterns)
                        .map(|((field, _ty), pattern)| FieldPat { field, pattern })
                        .collect();

                    if adt_def.is_enum() {
                        PatKind::Variant { adt_def: *adt_def, args, variant_index, subpatterns }
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
                        suffix: Box::new([]),
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
                        let suffix: Box<[_]> = subpatterns.collect();
                        let wild = Pat::wildcard_from_ty(self.ty);
                        PatKind::Slice {
                            prefix: prefix.into_boxed_slice(),
                            slice: Some(Box::new(wild)),
                            suffix,
                        }
                    }
                }
            }
            &Str(value) => PatKind::Constant { value },
            IntRange(range) => return range.to_pat(cx.tcx, self.ty),
            Wildcard | NonExhaustive | Hidden => PatKind::Wild,
            Missing { .. } => bug!(
                "trying to convert a `Missing` constructor into a `Pat`; this is probably a bug,
                `Missing` should have been processed in `apply_constructors`"
            ),
            F32Range(..) | F64Range(..) | Opaque | Or => {
                bug!("can't convert to pattern: {:?}", self)
            }
        };

        Pat { ty: self.ty, span: DUMMY_SP, kind }
    }

    pub(super) fn iter_fields<'a>(&'a self) -> impl Iterator<Item = &'a WitnessPat<'tcx>> {
        self.fields.iter()
    }
}
