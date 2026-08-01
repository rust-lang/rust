use std::fmt;
use std::ops::RangeFull;

use crate::Size;
#[cfg(feature = "nightly")]
use crate::StableHash;

/// Inclusive wrap-around range of valid values, that is, if
/// start > end, it represents `start..=MAX`, followed by `0..=end`.
///
/// That is, for an i8 primitive, a range of `254..=2` means following
/// sequence:
///
///    254 (-2), 255 (-1), 0, 1, 2
///
/// This is intended specifically to mirror LLVM’s `!range` metadata semantics.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "nightly", derive(StableHash))]
pub struct WrappingRange {
    pub start: u128,
    pub end: u128,
}

impl WrappingRange {
    pub(crate) fn debug_as(&self, size: Size, is_signed: bool) -> impl fmt::Debug {
        let range = *self;
        fmt::from_fn(move |f| {
            if range == WrappingRange::full(size) {
                // This is intentionally not using `is_full_for` so that we ensure
                // different values always debug-print differently.
                // We don't need the full details when it's the canonical full range,
                // but if one is looking at the debug output it might be that seeing
                // `u8 is (..=0) | (1..)` instead of `u8 is ..` is the information
                // you needed because the problem is that despite being *a* full
                // range it's not *the* canonical one you expected it was.
                f.write_str("..")
            } else if is_signed {
                let start = size.sign_extend(range.start);
                let end = size.sign_extend(range.end);
                if start > end {
                    write!(f, "(..={}) | ({}..)", end, start)
                } else {
                    write!(f, "{}..={}", start, end)
                }
            } else {
                write!(f, "{:?}", range)
            }
        })
    }

    pub fn full(size: Size) -> Self {
        Self { start: 0, end: size.unsigned_int_max() }
    }

    /// Returns `true` if `v` is contained in the range.
    #[inline(always)]
    pub fn contains(&self, v: u128) -> bool {
        if self.start <= self.end {
            self.start <= v && v <= self.end
        } else {
            self.start <= v || v <= self.end
        }
    }

    /// Returns `true` if all the values in `other` are contained in this range,
    /// when the values are considered as having width `size`.
    #[inline(always)]
    pub fn contains_range(&self, other: Self, size: Size) -> bool {
        if self.is_full_for(size) {
            true
        } else {
            let trunc = |x| size.truncate(x);

            let delta = self.start;
            let max = trunc(self.end.wrapping_sub(delta));

            let other_start = trunc(other.start.wrapping_sub(delta));
            let other_end = trunc(other.end.wrapping_sub(delta));

            // Having shifted both input ranges by `delta`, now we only need to check
            // whether `0..=max` contains `other_start..=other_end`, which can only
            // happen if the other doesn't wrap since `self` isn't everything.
            (other_start <= other_end) && (other_end <= max)
        }
    }

    /// Returns `self` with replaced `start`
    #[inline(always)]
    pub(crate) fn with_start(mut self, start: u128) -> Self {
        self.start = start;
        self
    }

    /// Returns `self` with replaced `end`
    #[inline(always)]
    pub(crate) fn with_end(mut self, end: u128) -> Self {
        self.end = end;
        self
    }

    /// The wrapping distance from `self.start` to `self.end`.
    fn width(&self, size: Size) -> u128 {
        size.truncate(u128::wrapping_sub(self.end, self.start))
    }

    /// Returns `true` if `size` completely fills the range.
    ///
    /// Note that this is *not* the same as `self == WrappingRange::full(size)`.
    /// Niche calculations can produce full ranges which are not the canonical one;
    /// for example `Option<NonZero<u16>>` gets `valid_range: (..=0) | (1..)`.
    #[inline]
    pub fn is_full_for(&self, size: Size) -> bool {
        let max_value = size.unsigned_int_max();
        debug_assert!(self.start <= max_value && self.end <= max_value);
        self.start == (self.end.wrapping_add(1) & max_value)
    }

    /// Checks whether this range is considered non-wrapping when the values are
    /// interpreted as *unsigned* numbers of width `size`.
    ///
    /// Returns `Ok(true)` if there's no wrap-around, `Ok(false)` if there is,
    /// and `Err(..)` if the range is full so it depends how you think about it.
    #[inline]
    pub fn no_unsigned_wraparound(&self, size: Size) -> Result<bool, RangeFull> {
        if self.is_full_for(size) { Err(..) } else { Ok(self.start <= self.end) }
    }

    /// Checks whether this range is considered non-wrapping when the values are
    /// interpreted as *signed* numbers of width `size`.
    ///
    /// This is heavily dependent on the `size`, as `100..=200` does wrap when
    /// interpreted as `i8`, but doesn't when interpreted as `i16`.
    ///
    /// Returns `Ok(true)` if there's no wrap-around, `Ok(false)` if there is,
    /// and `Err(..)` if the range is full so it depends how you think about it.
    #[inline]
    pub fn no_signed_wraparound(&self, size: Size) -> Result<bool, RangeFull> {
        if self.is_full_for(size) {
            Err(..)
        } else {
            let start: i128 = size.sign_extend(self.start);
            let end: i128 = size.sign_extend(self.end);
            Ok(start <= end)
        }
    }

    /// Returns a `WrappingRange` that contains all of the values from the iterator,
    /// when they're treated as values `size` wide.
    ///
    /// # Examples
    ///
    ///
    /// ```
    /// use rustc_abi::{Size, WrappingRange};
    ///
    /// let range = WrappingRange::smallest_range_containing([2, 6, 12, 4], Size::from_bytes(2));
    /// assert_eq!(range.unwrap(), WrappingRange { start: 2, end: 12 });
    ///
    /// let range = WrappingRange::smallest_range_containing(0..=127, Size::from_bytes(1));
    /// assert_eq!(range.unwrap(), WrappingRange { start: 0, end: 127 });
    /// let range = WrappingRange::smallest_range_containing([129, 128, 127], Size::from_bytes(1));
    /// assert_eq!(range.unwrap(), WrappingRange { start: 127, end: 129 });
    ///
    /// // The size matters because it changes where the wrapping can happen:
    /// let range = WrappingRange::smallest_range_containing([1, 254], Size::from_bytes(1));
    /// assert_eq!(range.unwrap(), WrappingRange { start: 254, end: 1 });
    /// let range = WrappingRange::smallest_range_containing([1, 254], Size::from_bytes(4));
    /// assert_eq!(range.unwrap(), WrappingRange { start: 1, end: 254 });
    ///
    /// // Both `100..=228` and `..=228 | 100..` are the same size, but we pick the one without zero.
    /// let range = WrappingRange::smallest_range_containing([100, 228], Size::from_bytes(1));
    /// assert_eq!(range.unwrap(), WrappingRange { start: 100, end: 228 });
    /// // These 4 values are evenly spaced so all 4 candidate ranges have length 193:
    /// // `(..=32) | (96..)`, `(..=96) | (160..)`, `(..=160) | (224..)`, and `32..=224`.
    /// // We pick the last one as the only one that doesn't contain zero.
    /// let range = WrappingRange::smallest_range_containing([0xA0, 0xE0, 0x20, 0x60], Size::from_bytes(1));
    /// assert_eq!(range.unwrap(), WrappingRange { start: 0x20, end: 0xE0 });
    /// ```
    pub fn smallest_range_containing(
        values: impl IntoIterator<Item = u128>,
        size: Size,
    ) -> Option<Self> {
        let mut values: Vec<_> = values.into_iter().collect();
        let umax = size.unsigned_int_max();
        for value in &values {
            debug_assert!(*value <= umax, "Value {value:?} is too big for {size:?}");
        }
        values.sort_unstable();

        // Having sorted all the values, every element is a possible start point for the
        // range of values, up to the previous element (wrapping around the end of the vec).
        // Look at all those candidates and pick the one that's as narrow as possible.
        let pairs = std::iter::zip(values.iter().copied(), values.iter().copied().cycle().skip(1));
        let ranges = pairs.map(|(end, start)| WrappingRange { start, end });
        let smallest_range = ranges.min_by_key(|r| (r.width(size), r.start));
        smallest_range
    }
}

impl fmt::Debug for WrappingRange {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start > self.end {
            write!(fmt, "(..={}) | ({}..)", self.end, self.start)?;
        } else {
            write!(fmt, "{}..={}", self.start, self.end)?;
        }
        Ok(())
    }
}
