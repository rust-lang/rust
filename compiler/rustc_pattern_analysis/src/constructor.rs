//! As explained in [`crate::usefulness`], values and patterns are made from constructors applied to
//! fields. This file defines a `Constructor` enum and various operations to manipulate them.
//!
//! There are two important bits of core logic in this file: constructor inclusion and constructor
//! splitting. Constructor inclusion, i.e. whether a constructor is included in/covered by another,
//! is straightforward and defined in [`Constructor::is_covered_by`].
//!
//! Constructor splitting is mentioned in [`crate::usefulness`] but not detailed. We describe it
//! precisely here.
//!
//!
//!
//! # Constructor grouping and splitting
//!
//! As explained in the corresponding section in [`crate::usefulness`], to make usefulness tractable
//! we need to group together constructors that have the same effect when they are used to
//! specialize the matrix.
//!
//! Example:
//! ```compile_fail,E0004
//! match (0, false) {
//!     (0 ..=100, true) => {}
//!     (50..=150, false) => {}
//!     (0 ..=200, _) => {}
//! }
//! ```
//!
//! In this example we can restrict specialization to 5 cases: `0..50`, `50..=100`, `101..=150`,
//! `151..=200` and `200..`.
//!
//! In [`crate::usefulness`], we had said that `specialize` only takes value-only constructors. We
//! now relax this restriction: we allow `specialize` to take constructors like `0..50` as long as
//! we're careful to only do that with constructors that make sense. For example, `specialize(0..50,
//! (0..=100, true))` is sensible, but `specialize(50..=200, (0..=100, true))` is not.
//!
//! Constructor splitting looks at the constructors in the first column of the matrix and constructs
//! such a sensible set of constructors. Formally, we want to find a smallest disjoint set of
//! constructors:
//! - Whose union covers the whole type, and
//! - That have no non-trivial intersection with any of the constructors in the column (i.e. they're
//!     each either disjoint with or covered by any given column constructor).
//!
//! We compute this in two steps: first [`PatCx::ctors_for_ty`] determines the
//! set of all possible constructors for the type. Then [`ConstructorSet::split`] looks at the
//! column of constructors and splits the set into groups accordingly. The precise invariants of
//! [`ConstructorSet::split`] is described in [`SplitConstructorSet`].
//!
//! Constructor splitting has two interesting special cases: integer range splitting (see
//! [`IntRange::split`]) and slice splitting (see [`Slice::split`]).
//!
//!
//!
//! # The `Missing` constructor
//!
//! We detail a special case of constructor splitting that is a bit subtle. Take the following:
//!
//! ```
//! enum Direction { North, South, East, West }
//! # let wind = (Direction::North, 0u8);
//! match wind {
//!     (Direction::North, 50..) => {}
//!     (_, _) => {}
//! }
//! ```
//!
//! Here we expect constructor splitting to output two cases: `North`, and "everything else". This
//! "everything else" is represented by [`Constructor::Missing`]. Unlike other constructors, it's a
//! bit contextual: to know the exact list of constructors it represents we have to look at the
//! column. In practice however we don't need to, because by construction it only matches rows that
//! have wildcards. This is how this constructor is special: the only constructor that covers it is
//! `Wildcard`.
//!
//! The only place where we care about which constructors `Missing` represents is in diagnostics
//! (see `crate::usefulness::WitnessMatrix::apply_constructor`).
//!
//! We choose whether to specialize with `Missing` in
//! `crate::usefulness::compute_exhaustiveness_and_usefulness`.
//!
//!
//!
//! ## Empty types, empty constructors, and the `exhaustive_patterns` feature
//!
//! An empty type is a type that has no valid value, like `!`, `enum Void {}`, or `Result<!, !>`.
//! They require careful handling.
//!
//! First, for soundness reasons related to the possible existence of invalid values, by default we
//! don't treat empty types as empty. We force them to be matched with wildcards. Except if the
//! `exhaustive_patterns` feature is turned on, in which case we do treat them as empty. And also
//! except if the type has no constructors (like `enum Void {}` but not like `Result<!, !>`), we
//! specifically allow `match void {}` to be exhaustive. There are additionally considerations of
//! place validity that are handled in `crate::usefulness`. Yes this is a bit tricky.
//!
//! The second thing is that regardless of the above, it is always allowed to use all the
//! constructors of a type. For example, all the following is ok:
//!
//! ```rust,ignore(example)
//! # #![feature(never_type)]
//! # #![feature(exhaustive_patterns)]
//! fn foo(x: Option<!>) {
//!   match x {
//!     None => {}
//!     Some(_) => {}
//!   }
//! }
//! fn bar(x: &[!]) -> u32 {
//!   match x {
//!     [] => 1,
//!     [_] => 2,
//!     [_, _] => 3,
//!   }
//! }
//! ```
//!
//! Moreover, take the following:
//!
//! ```rust
//! # #![feature(never_type)]
//! # #![feature(exhaustive_patterns)]
//! # let x = None::<!>;
//! match x {
//!   None => {}
//! }
//! ```
//!
//! On a normal type, we would identify `Some` as missing and tell the user. If `x: Option<!>`
//! however (and `exhaustive_patterns` is on), it's ok to omit `Some`. When listing the constructors
//! of a type, we must therefore track which can be omitted.
//!
//! Let's call "empty" a constructor that matches no valid value for the type, like `Some` for the
//! type `Option<!>`. What this all means is that `ConstructorSet` must know which constructors are
//! empty. The difference between empty and nonempty constructors is that empty constructors need
//! not be present for the match to be exhaustive.
//!
//! A final remark: empty constructors of arity 0 break specialization, we must avoid them. The
//! reason is that if we specialize by them, nothing remains to witness the emptiness; the rest of
//! the algorithm can't distinguish them from a nonempty constructor. The only known case where this
//! could happen is the `[..]` pattern on `[!; N]` with `N > 0` so we must take care to not emit it.
//!
//! This is all handled by [`PatCx::ctors_for_ty`] and
//! [`ConstructorSet::split`]. The invariants of [`SplitConstructorSet`] are also of interest.
//!
//!
//! ## Unions
//!
//! Unions allow us to match a value via several overlapping representations at the same time. For
//! example, the following is exhaustive because when seeing the value as a boolean we handled all
//! possible cases (other cases such as `n == 3` would trigger UB).
//!
//! ```rust
//! # fn main() {
//! union U8AsBool {
//!     n: u8,
//!     b: bool,
//! }
//! let x = U8AsBool { n: 1 };
//! unsafe {
//!     match x {
//!         U8AsBool { n: 2 } => {}
//!         U8AsBool { b: true } => {}
//!         U8AsBool { b: false } => {}
//!     }
//! }
//! # }
//! ```
//!
//! Pattern-matching has no knowledge that e.g. `false as u8 == 0`, so the values we consider in the
//! algorithm look like `U8AsBool { b: true, n: 2 }`. In other words, for the most part a union is
//! treated like a struct with the same fields. The difference lies in how we construct witnesses of
//! non-exhaustiveness.
//!
//!
//! ## Opaque patterns
//!
//! Some patterns, such as constants that are not allowed to be matched structurally, cannot be
//! inspected, which we handle with `Constructor::Opaque`. Since we know nothing of these patterns,
//! we assume they never cover each other. In order to respect the invariants of
//! [`SplitConstructorSet`], we give each `Opaque` constructor a unique id so we can recognize it.

use std::cmp::{self, Ordering, max, min};
use std::fmt;
use std::iter::once;

use rustc_apfloat::ieee::{DoubleS, HalfS, IeeeFloat, QuadS, SingleS};
use rustc_index::IndexVec;
use rustc_index::bit_set::{DenseBitSet, GrowableBitSet};
use smallvec::SmallVec;

use self::Constructor::*;
use self::MaybeInfiniteInt::*;
use self::SliceKind::*;
use crate::PatCx;

/// Whether we have seen a constructor in the column or not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Presence {
    Unseen,
    Seen,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RangeEnd {
    Included,
    Excluded,
}

impl fmt::Display for RangeEnd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            RangeEnd::Included => "..=",
            RangeEnd::Excluded => "..",
        })
    }
}

/// A possibly infinite integer. Values are encoded such that the ordering on `u128` matches the
/// natural order on the original type. For example, `-128i8` is encoded as `0` and `127i8` as
/// `255`. See `signed_bias` for details.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MaybeInfiniteInt {
    NegInfinity,
    /// Encoded value. DO NOT CONSTRUCT BY HAND; use `new_finite_{int,uint}`.
    #[non_exhaustive]
    Finite(u128),
    PosInfinity,
}

impl MaybeInfiniteInt {
    pub fn new_finite_uint(bits: u128) -> Self {
        Finite(bits)
    }
    pub fn new_finite_int(bits: u128, size: u64) -> Self {
        // Perform a shift if the underlying types are signed, which makes the interval arithmetic
        // type-independent.
        let bias = 1u128 << (size - 1);
        Finite(bits ^ bias)
    }

    pub fn as_finite_uint(self) -> Option<u128> {
        match self {
            Finite(bits) => Some(bits),
            _ => None,
        }
    }
    pub fn as_finite_int(self, size: u64) -> Option<u128> {
        // We decode the shift.
        match self {
            Finite(bits) => {
                let bias = 1u128 << (size - 1);
                Some(bits ^ bias)
            }
            _ => None,
        }
    }

    /// Note: this will not turn a finite value into an infinite one or vice-versa.
    pub fn minus_one(self) -> Option<Self> {
        match self {
            Finite(n) => n.checked_sub(1).map(Finite),
            x => Some(x),
        }
    }
    /// Note: this will turn `u128::MAX` into `PosInfinity`. This means `plus_one` and `minus_one`
    /// are not strictly inverses, but that poses no problem in our use of them.
    /// this will not turn a finite value into an infinite one or vice-versa.
    pub fn plus_one(self) -> Option<Self> {
        match self {
            Finite(n) => match n.checked_add(1) {
                Some(m) => Some(Finite(m)),
                None => Some(PosInfinity),
            },
            x => Some(x),
        }
    }
}

/// An exclusive interval, used for precise integer exhaustiveness checking. `IntRange`s always
/// store a contiguous range.
///
/// `IntRange` is never used to encode an empty range or a "range" that wraps around the (offset)
/// space: i.e., `range.lo < range.hi`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct IntRange {
    pub lo: MaybeInfiniteInt, // Must not be `PosInfinity`.
    pub hi: MaybeInfiniteInt, // Must not be `NegInfinity`.
}

impl IntRange {
    /// Best effort; will not know that e.g. `255u8..` is a singleton.
    pub fn is_singleton(&self) -> bool {
        // Since `lo` and `hi` can't be the same `Infinity` and `plus_one` never changes from finite
        // to infinite, this correctly only detects ranges that contain exactly one `Finite(x)`.
        self.lo.plus_one() == Some(self.hi)
    }

    /// Construct a singleton range.
    /// `x` must be a `Finite(_)` value.
    #[inline]
    pub fn from_singleton(x: MaybeInfiniteInt) -> IntRange {
        // `unwrap()` is ok on a finite value
        IntRange { lo: x, hi: x.plus_one().unwrap() }
    }

    /// Construct a range with these boundaries.
    /// `lo` must not be `PosInfinity`. `hi` must not be `NegInfinity`.
    #[inline]
    pub fn from_range(lo: MaybeInfiniteInt, mut hi: MaybeInfiniteInt, end: RangeEnd) -> IntRange {
        if end == RangeEnd::Included {
            hi = hi.plus_one().unwrap();
        }
        if lo >= hi {
            // This should have been caught earlier by E0030.
            panic!("malformed range pattern: {lo:?}..{hi:?}");
        }
        IntRange { lo, hi }
    }

    #[inline]
    pub fn is_subrange(&self, other: &Self) -> bool {
        other.lo <= self.lo && self.hi <= other.hi
    }

    fn intersection(&self, other: &Self) -> Option<Self> {
        if self.lo < other.hi && other.lo < self.hi {
            Some(IntRange { lo: max(self.lo, other.lo), hi: min(self.hi, other.hi) })
        } else {
            None
        }
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
    ///
    /// ## `isize`/`usize`
    ///
    /// Whereas a wildcard of type `i32` stands for the range `i32::MIN..=i32::MAX`, a `usize`
    /// wildcard stands for `0..PosInfinity` and a `isize` wildcard stands for
    /// `NegInfinity..PosInfinity`. In other words, as far as `IntRange` is concerned, there are
    /// values before `isize::MIN` and after `usize::MAX`/`isize::MAX`.
    /// This is to avoid e.g. `0..(u32::MAX as usize)` from being exhaustive on one architecture and
    /// not others. This was decided in <https://github.com/rust-lang/rfcs/pull/2591>.
    ///
    /// These infinities affect splitting subtly: it is possible to get `NegInfinity..0` and
    /// `usize::MAX+1..PosInfinity` in the output. Diagnostics must be careful to handle these
    /// fictitious ranges sensibly.
    fn split(
        &self,
        column_ranges: impl Iterator<Item = IntRange>,
    ) -> impl Iterator<Item = (Presence, IntRange)> {
        // The boundaries of ranges in `column_ranges` intersected with `self`.
        // We do parenthesis matching for input ranges. A boundary counts as +1 if it starts
        // a range and -1 if it ends it. When the count is > 0 between two boundaries, we
        // are within an input range.
        let mut boundaries: Vec<(MaybeInfiniteInt, isize)> = column_ranges
            .filter_map(|r| self.intersection(&r))
            .flat_map(|r| [(r.lo, 1), (r.hi, -1)])
            .collect();
        // We sort by boundary, and for each boundary we sort the "closing parentheses" first. The
        // order of +1/-1 for a same boundary value is actually irrelevant, because we only look at
        // the accumulated count between distinct boundary values.
        boundaries.sort_unstable();

        // Accumulate parenthesis counts.
        let mut paren_counter = 0isize;
        // Gather pairs of adjacent boundaries.
        let mut prev_bdy = self.lo;
        boundaries
            .into_iter()
            // End with the end of the range. The count is ignored.
            .chain(once((self.hi, 0)))
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
                use Presence::*;
                let presence = if paren_count > 0 { Seen } else { Unseen };
                let range = IntRange { lo: prev_bdy, hi: bdy };
                (presence, range)
            })
    }
}

/// Note: this will render signed ranges incorrectly. To render properly, convert to a pattern
/// first.
impl fmt::Debug for IntRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_singleton() {
            // Only finite ranges can be singletons.
            let Finite(lo) = self.lo else { unreachable!() };
            write!(f, "{lo}")?;
        } else {
            if let Finite(lo) = self.lo {
                write!(f, "{lo}")?;
            }
            write!(f, "{}", RangeEnd::Excluded)?;
            if let Finite(hi) = self.hi {
                write!(f, "{hi}")?;
            }
        }
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SliceKind {
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
    pub fn arity(self) -> usize {
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
pub struct Slice {
    /// `None` if the matched value is a slice, `Some(n)` if it is an array of size `n`.
    pub(crate) array_len: Option<usize>,
    /// The kind of pattern it is: fixed-length `[x, y]` or variable length `[x, .., y]`.
    pub(crate) kind: SliceKind,
}

impl Slice {
    pub fn new(array_len: Option<usize>, kind: SliceKind) -> Self {
        let kind = match (array_len, kind) {
            // If the middle `..` has length 0, we effectively have a fixed-length pattern.
            (Some(len), VarLen(prefix, suffix)) if prefix + suffix == len => FixedLen(len),
            (Some(len), VarLen(prefix, suffix)) if prefix + suffix > len => panic!(
                "Slice pattern of length {} longer than its array length {len}",
                prefix + suffix
            ),
            _ => kind,
        };
        Slice { array_len, kind }
    }

    pub fn arity(self) -> usize {
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
        let mut seen_fixed_lens = GrowableBitSet::new_empty();
        match &mut max_slice {
            VarLen(max_prefix_len, max_suffix_len) => {
                // A length larger than any fixed-length slice encountered.
                // We start at 1 in case the subtype is empty because in that case the zero-length
                // slice must be treated separately from the rest.
                let mut fixed_len_upper_bound = 1;
                // We grow `max_slice` to be larger than all slices encountered, as described above.
                // `L` is `max_slice.arity()`. For diagnostics, we keep the prefix and suffix
                // lengths separate.
                for slice in column_slices {
                    match slice.kind {
                        FixedLen(len) => {
                            fixed_len_upper_bound = cmp::max(fixed_len_upper_bound, len + 1);
                            seen_fixed_lens.insert(len);
                        }
                        VarLen(prefix, suffix) => {
                            *max_prefix_len = cmp::max(*max_prefix_len, prefix);
                            *max_suffix_len = cmp::max(*max_suffix_len, suffix);
                            min_var_len = cmp::min(min_var_len, prefix + suffix);
                        }
                    }
                }
                // If `fixed_len_upper_bound >= L`, we set `L` to `fixed_len_upper_bound`.
                if let Some(delta) =
                    fixed_len_upper_bound.checked_sub(*max_prefix_len + *max_suffix_len)
                {
                    *max_prefix_len += delta
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
            let seen = if min_var_len <= arity || seen_fixed_lens.contains(arity) {
                Presence::Seen
            } else {
                Presence::Unseen
            };
            (seen, Slice::new(self.array_len, kind))
        })
    }
}

/// A globally unique id to distinguish `Opaque` patterns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpaqueId(u32);

impl OpaqueId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static OPAQUE_ID: AtomicU32 = AtomicU32::new(0);
        OpaqueId(OPAQUE_ID.fetch_add(1, Ordering::SeqCst))
    }
}

/// A value can be decomposed into a constructor applied to some fields. This struct represents
/// the constructor. See also `Fields`.
///
/// `pat_constructor` retrieves the constructor corresponding to a pattern.
/// `specialize_constructor` returns the list of fields corresponding to a pattern, given a
/// constructor. `Constructor::apply` reconstructs the pattern from a pair of `Constructor` and
/// `Fields`.
#[derive(Debug)]
pub enum Constructor<Cx: PatCx> {
    /// Tuples and structs.
    Struct,
    /// Enum variants.
    Variant(Cx::VariantIdx),
    /// References
    Ref,
    /// Array and slice patterns.
    Slice(Slice),
    /// Union field accesses.
    UnionField,
    /// Booleans
    Bool(bool),
    /// Ranges of integer literal values (`2`, `2..=5` or `2..5`).
    IntRange(IntRange),
    /// Ranges of floating-point literal values (`2.0..=5.2`).
    F16Range(IeeeFloat<HalfS>, IeeeFloat<HalfS>, RangeEnd),
    F32Range(IeeeFloat<SingleS>, IeeeFloat<SingleS>, RangeEnd),
    F64Range(IeeeFloat<DoubleS>, IeeeFloat<DoubleS>, RangeEnd),
    F128Range(IeeeFloat<QuadS>, IeeeFloat<QuadS>, RangeEnd),
    /// String literals. Strings are not quite the same as `&[u8]` so we treat them separately.
    Str(Cx::StrLit),
    /// Deref patterns (enabled by the `deref_patterns` feature) provide a way of matching on a
    /// smart pointer ADT through its pointee. They don't directly correspond to ADT constructors,
    /// and currently are not supported alongside them. Carries the type of the pointee.
    DerefPattern(Cx::Ty),
    /// Constants that must not be matched structurally. They are treated as black boxes for the
    /// purposes of exhaustiveness: we must not inspect them, and they don't count towards making a
    /// match exhaustive.
    /// Carries an id that must be unique within a match. We need this to ensure the invariants of
    /// [`SplitConstructorSet`].
    Opaque(OpaqueId),
    /// Or-pattern.
    Or,
    /// Wildcard pattern.
    Wildcard,
    /// Never pattern. Only used in `WitnessPat`. An actual never pattern should be lowered as
    /// `Wildcard`.
    Never,
    /// Fake extra constructor for enums that aren't allowed to be matched exhaustively. Also used
    /// for those types for which we cannot list constructors explicitly, like `f64` and `str`. Only
    /// used in `WitnessPat`.
    NonExhaustive,
    /// Fake extra constructor for variants that should not be mentioned in diagnostics. We use this
    /// for variants behind an unstable gate as well as `#[doc(hidden)]` ones. Only used in
    /// `WitnessPat`.
    Hidden,
    /// Fake extra constructor for constructors that are not seen in the matrix, as explained at the
    /// top of the file. Only used for specialization.
    Missing,
    /// Fake extra constructor that indicates and empty field that is private. When we encounter one
    /// we skip the column entirely so we don't observe its emptiness. Only used for specialization.
    PrivateUninhabited,
}

impl<Cx: PatCx> Clone for Constructor<Cx> {
    fn clone(&self) -> Self {
        match self {
            Constructor::Struct => Constructor::Struct,
            Constructor::Variant(idx) => Constructor::Variant(*idx),
            Constructor::Ref => Constructor::Ref,
            Constructor::Slice(slice) => Constructor::Slice(*slice),
            Constructor::UnionField => Constructor::UnionField,
            Constructor::Bool(b) => Constructor::Bool(*b),
            Constructor::IntRange(range) => Constructor::IntRange(*range),
            Constructor::F16Range(lo, hi, end) => Constructor::F16Range(*lo, *hi, *end),
            Constructor::F32Range(lo, hi, end) => Constructor::F32Range(*lo, *hi, *end),
            Constructor::F64Range(lo, hi, end) => Constructor::F64Range(*lo, *hi, *end),
            Constructor::F128Range(lo, hi, end) => Constructor::F128Range(*lo, *hi, *end),
            Constructor::Str(value) => Constructor::Str(value.clone()),
            Constructor::DerefPattern(ty) => Constructor::DerefPattern(ty.clone()),
            Constructor::Opaque(inner) => Constructor::Opaque(inner.clone()),
            Constructor::Or => Constructor::Or,
            Constructor::Never => Constructor::Never,
            Constructor::Wildcard => Constructor::Wildcard,
            Constructor::NonExhaustive => Constructor::NonExhaustive,
            Constructor::Hidden => Constructor::Hidden,
            Constructor::Missing => Constructor::Missing,
            Constructor::PrivateUninhabited => Constructor::PrivateUninhabited,
        }
    }
}

impl<Cx: PatCx> Constructor<Cx> {
    pub(crate) fn is_non_exhaustive(&self) -> bool {
        matches!(self, NonExhaustive)
    }

    pub(crate) fn as_variant(&self) -> Option<Cx::VariantIdx> {
        match self {
            Variant(i) => Some(*i),
            _ => None,
        }
    }
    fn as_bool(&self) -> Option<bool> {
        match self {
            Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub(crate) fn as_int_range(&self) -> Option<&IntRange> {
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

    /// The number of fields for this constructor. This must be kept in sync with
    /// `Fields::wildcards`.
    pub(crate) fn arity(&self, cx: &Cx, ty: &Cx::Ty) -> usize {
        cx.ctor_arity(self, ty)
    }

    /// Returns whether `self` is covered by `other`, i.e. whether `self` is a subset of `other`.
    /// For the simple cases, this is simply checking for equality. For the "grouped" constructors,
    /// this checks for inclusion.
    // We inline because this has a single call site in `Matrix::specialize_constructor`.
    #[inline]
    pub(crate) fn is_covered_by(&self, cx: &Cx, other: &Self) -> Result<bool, Cx::Error> {
        Ok(match (self, other) {
            (Wildcard, _) => {
                return Err(cx.bug(format_args!(
                    "Constructor splitting should not have returned `Wildcard`"
                )));
            }
            // Wildcards cover anything
            (_, Wildcard) => true,
            // `PrivateUninhabited` skips everything.
            (PrivateUninhabited, _) => true,
            // Only a wildcard pattern can match these special constructors.
            (Missing { .. } | NonExhaustive | Hidden, _) => false,

            (Struct, Struct) => true,
            (Ref, Ref) => true,
            (UnionField, UnionField) => true,
            (Variant(self_id), Variant(other_id)) => self_id == other_id,
            (Bool(self_b), Bool(other_b)) => self_b == other_b,

            (IntRange(self_range), IntRange(other_range)) => self_range.is_subrange(other_range),
            (F16Range(self_from, self_to, self_end), F16Range(other_from, other_to, other_end)) => {
                self_from.ge(other_from)
                    && match self_to.partial_cmp(other_to) {
                        Some(Ordering::Less) => true,
                        Some(Ordering::Equal) => other_end == self_end,
                        _ => false,
                    }
            }
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
            (
                F128Range(self_from, self_to, self_end),
                F128Range(other_from, other_to, other_end),
            ) => {
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

            // Deref patterns only interact with other deref patterns. Prior to usefulness analysis,
            // we ensure they don't appear alongside any other non-wild non-opaque constructors.
            (DerefPattern(_), DerefPattern(_)) => true,

            // Opaque constructors don't interact with anything unless they come from the
            // syntactically identical pattern.
            (Opaque(self_id), Opaque(other_id)) => self_id == other_id,
            (Opaque(..), _) | (_, Opaque(..)) => false,

            _ => {
                return Err(cx.bug(format_args!(
                    "trying to compare incompatible constructors {self:?} and {other:?}"
                )));
            }
        })
    }

    pub(crate) fn fmt_fields(
        &self,
        f: &mut fmt::Formatter<'_>,
        ty: &Cx::Ty,
        mut fields: impl Iterator<Item = impl fmt::Debug>,
    ) -> fmt::Result {
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

        match self {
            Struct | Variant(_) | UnionField => {
                Cx::write_variant_name(f, self, ty)?;
                // Without `cx`, we can't know which field corresponds to which, so we can't
                // get the names of the fields. Instead we just display everything as a tuple
                // struct, which should be good enough.
                write!(f, "(")?;
                for p in fields {
                    write!(f, "{}{:?}", start_or_comma(), p)?;
                }
                write!(f, ")")?;
            }
            // Note: given the expansion of `&str` patterns done in `expand_pattern`, we should
            // be careful to detect strings here. However a string literal pattern will never
            // be reported as a non-exhaustiveness witness, so we can ignore this issue.
            Ref => {
                write!(f, "&{:?}", fields.next().unwrap())?;
            }
            Slice(slice) => {
                write!(f, "[")?;
                match slice.kind {
                    SliceKind::FixedLen(_) => {
                        for p in fields {
                            write!(f, "{}{:?}", start_or_comma(), p)?;
                        }
                    }
                    SliceKind::VarLen(prefix_len, _) => {
                        for p in fields.by_ref().take(prefix_len) {
                            write!(f, "{}{:?}", start_or_comma(), p)?;
                        }
                        write!(f, "{}..", start_or_comma())?;
                        for p in fields {
                            write!(f, "{}{:?}", start_or_comma(), p)?;
                        }
                    }
                }
                write!(f, "]")?;
            }
            Bool(b) => write!(f, "{b}")?,
            // Best-effort, will render signed ranges incorrectly
            IntRange(range) => write!(f, "{range:?}")?,
            F16Range(lo, hi, end) => write!(f, "{lo}{end}{hi}")?,
            F32Range(lo, hi, end) => write!(f, "{lo}{end}{hi}")?,
            F64Range(lo, hi, end) => write!(f, "{lo}{end}{hi}")?,
            F128Range(lo, hi, end) => write!(f, "{lo}{end}{hi}")?,
            Str(value) => write!(f, "{value:?}")?,
            DerefPattern(_) => write!(f, "deref!({:?})", fields.next().unwrap())?,
            Opaque(..) => write!(f, "<constant pattern>")?,
            Or => {
                for pat in fields {
                    write!(f, "{}{:?}", start_or_continue(" | "), pat)?;
                }
            }
            Never => write!(f, "!")?,
            Wildcard | Missing | NonExhaustive | Hidden | PrivateUninhabited => {
                write!(f, "_ : {:?}", ty)?
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum VariantVisibility {
    /// Variant that doesn't fit the other cases, i.e. most variants.
    Visible,
    /// Variant behind an unstable gate or with the `#[doc(hidden)]` attribute. It will not be
    /// mentioned in diagnostics unless the user mentioned it first.
    Hidden,
    /// Variant that matches no value. E.g. `Some::<Option<!>>` if the `exhaustive_patterns` feature
    /// is enabled. Like `Hidden`, it will not be mentioned in diagnostics unless the user mentioned
    /// it first.
    Empty,
}

/// Describes the set of all constructors for a type. For details, in particular about the emptiness
/// of constructors, see the top of the file.
///
/// In terms of division of responsibility, [`ConstructorSet::split`] handles all of the
/// `exhaustive_patterns` feature.
#[derive(Debug)]
pub enum ConstructorSet<Cx: PatCx> {
    /// The type is a tuple or struct. `empty` tracks whether the type is empty.
    Struct { empty: bool },
    /// This type has the following list of constructors. If `variants` is empty and
    /// `non_exhaustive` is false, don't use this; use `NoConstructors` instead.
    Variants { variants: IndexVec<Cx::VariantIdx, VariantVisibility>, non_exhaustive: bool },
    /// The type is `&T`.
    Ref,
    /// The type is a union.
    Union,
    /// Booleans.
    Bool,
    /// The type is spanned by integer values. The range or ranges give the set of allowed values.
    /// The second range is only useful for `char`.
    Integers { range_1: IntRange, range_2: Option<IntRange> },
    /// The type is matched by slices. `array_len` is the compile-time length of the array, if
    /// known. If `subtype_is_empty`, all constructors are empty except possibly the zero-length
    /// slice `[]`.
    Slice { array_len: Option<usize>, subtype_is_empty: bool },
    /// The constructors cannot be listed, and the type cannot be matched exhaustively. E.g. `str`,
    /// floats.
    Unlistable,
    /// The type has no constructors (not even empty ones). This is `!` and empty enums.
    NoConstructors,
}

/// Describes the result of analyzing the constructors in a column of a match.
///
/// `present` is morally the set of constructors present in the column, and `missing` is the set of
/// constructors that exist in the type but are not present in the column.
///
/// More formally, if we discard wildcards from the column, this respects the following constraints:
/// 1. the union of `present`, `missing` and `missing_empty` covers all the constructors of the type
/// 2. each constructor in `present` is covered by something in the column
/// 3. no constructor in `missing` or `missing_empty` is covered by anything in the column
/// 4. each constructor in the column is equal to the union of one or more constructors in `present`
/// 5. `missing` does not contain empty constructors (see discussion about emptiness at the top of
///    the file);
/// 6. `missing_empty` contains only empty constructors
/// 7. constructors in `present`, `missing` and `missing_empty` are split for the column; in other
///    words, they are either fully included in or fully disjoint from each constructor in the
///    column. In yet other words, there are no non-trivial intersections like between `0..10` and
///    `5..15`.
///
/// We must be particularly careful with weird constructors like `Opaque`: they're not formally part
/// of the `ConstructorSet` for the type, yet if we forgot to include them in `present` we would be
/// ignoring any row with `Opaque`s in the algorithm. Hence the importance of point 4.
#[derive(Debug)]
pub struct SplitConstructorSet<Cx: PatCx> {
    pub present: SmallVec<[Constructor<Cx>; 1]>,
    pub missing: Vec<Constructor<Cx>>,
    pub missing_empty: Vec<Constructor<Cx>>,
}

impl<Cx: PatCx> ConstructorSet<Cx> {
    /// This analyzes a column of constructors to 1/ determine which constructors of the type (if
    /// any) are missing; 2/ split constructors to handle non-trivial intersections e.g. on ranges
    /// or slices. This can get subtle; see [`SplitConstructorSet`] for details of this operation
    /// and its invariants.
    pub fn split<'a>(
        &self,
        ctors: impl Iterator<Item = &'a Constructor<Cx>> + Clone,
    ) -> SplitConstructorSet<Cx>
    where
        Cx: 'a,
    {
        let mut present: SmallVec<[_; 1]> = SmallVec::new();
        // Empty constructors found missing.
        let mut missing_empty = Vec::new();
        // Nonempty constructors found missing.
        let mut missing = Vec::new();
        // Constructors in `ctors`, except wildcards and opaques.
        let mut seen = Vec::new();
        // If we see a deref pattern, it must be the only non-wildcard non-opaque constructor; we
        // ensure this prior to analysis.
        let mut deref_pat_present = false;
        for ctor in ctors.cloned() {
            match ctor {
                DerefPattern(..) => {
                    if !deref_pat_present {
                        deref_pat_present = true;
                        present.push(ctor);
                    }
                }
                Opaque(..) => present.push(ctor),
                Wildcard => {} // discard wildcards
                _ => seen.push(ctor),
            }
        }

        match self {
            _ if deref_pat_present => {
                // Deref patterns are the only constructor; nothing is missing.
            }
            ConstructorSet::Struct { empty } => {
                if !seen.is_empty() {
                    present.push(Struct);
                } else if *empty {
                    missing_empty.push(Struct);
                } else {
                    missing.push(Struct);
                }
            }
            ConstructorSet::Ref => {
                if !seen.is_empty() {
                    present.push(Ref);
                } else {
                    missing.push(Ref);
                }
            }
            ConstructorSet::Union => {
                if !seen.is_empty() {
                    present.push(UnionField);
                } else {
                    missing.push(UnionField);
                }
            }
            ConstructorSet::Variants { variants, non_exhaustive } => {
                let mut seen_set = DenseBitSet::new_empty(variants.len());
                for idx in seen.iter().filter_map(|c| c.as_variant()) {
                    seen_set.insert(idx);
                }
                let mut skipped_a_hidden_variant = false;

                for (idx, visibility) in variants.iter_enumerated() {
                    let ctor = Variant(idx);
                    if seen_set.contains(idx) {
                        present.push(ctor);
                    } else {
                        // We only put visible variants directly into `missing`.
                        match visibility {
                            VariantVisibility::Visible => missing.push(ctor),
                            VariantVisibility::Hidden => skipped_a_hidden_variant = true,
                            VariantVisibility::Empty => missing_empty.push(ctor),
                        }
                    }
                }

                if skipped_a_hidden_variant {
                    missing.push(Hidden);
                }
                if *non_exhaustive {
                    missing.push(NonExhaustive);
                }
            }
            ConstructorSet::Bool => {
                let mut seen_false = false;
                let mut seen_true = false;
                for b in seen.iter().filter_map(|ctor| ctor.as_bool()) {
                    if b {
                        seen_true = true;
                    } else {
                        seen_false = true;
                    }
                }
                if seen_false {
                    present.push(Bool(false));
                } else {
                    missing.push(Bool(false));
                }
                if seen_true {
                    present.push(Bool(true));
                } else {
                    missing.push(Bool(true));
                }
            }
            ConstructorSet::Integers { range_1, range_2 } => {
                let seen_ranges: Vec<_> =
                    seen.iter().filter_map(|ctor| ctor.as_int_range()).copied().collect();
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
            }
            ConstructorSet::Slice { array_len, subtype_is_empty } => {
                let seen_slices = seen.iter().filter_map(|c| c.as_slice());
                let base_slice = Slice::new(*array_len, VarLen(0, 0));
                for (seen, splitted_slice) in base_slice.split(seen_slices) {
                    let ctor = Slice(splitted_slice);
                    match seen {
                        Presence::Seen => present.push(ctor),
                        Presence::Unseen => {
                            if *subtype_is_empty && splitted_slice.arity() != 0 {
                                // We have subpatterns of an empty type, so the constructor is
                                // empty.
                                missing_empty.push(ctor);
                            } else {
                                missing.push(ctor);
                            }
                        }
                    }
                }
            }
            ConstructorSet::Unlistable => {
                // Since we can't list constructors, we take the ones in the column. This might list
                // some constructors several times but there's not much we can do.
                present.extend(seen);
                missing.push(NonExhaustive);
            }
            ConstructorSet::NoConstructors => {
                // In a `MaybeInvalid` place even an empty pattern may be reachable. We therefore
                // add a dummy empty constructor here, which will be ignored if the place is
                // `ValidOnly`.
                missing_empty.push(Never);
            }
        }

        SplitConstructorSet { present, missing, missing_empty }
    }

    /// Whether this set only contains empty constructors.
    pub(crate) fn all_empty(&self) -> bool {
        match self {
            ConstructorSet::Bool
            | ConstructorSet::Integers { .. }
            | ConstructorSet::Ref
            | ConstructorSet::Union
            | ConstructorSet::Unlistable => false,
            ConstructorSet::NoConstructors => true,
            ConstructorSet::Struct { empty } => *empty,
            ConstructorSet::Variants { variants, non_exhaustive } => {
                !*non_exhaustive
                    && variants
                        .iter()
                        .all(|visibility| matches!(visibility, VariantVisibility::Empty))
            }
            ConstructorSet::Slice { array_len, subtype_is_empty } => {
                *subtype_is_empty && matches!(array_len, Some(1..))
            }
        }
    }
}
