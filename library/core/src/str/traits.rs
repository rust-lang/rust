//! Trait implementations for `str`.

use crate::cmp::Ordering;
use crate::ops;
use crate::ptr;
use crate::slice::SliceIndex;

use super::ParseBoolError;

/// Implements ordering of strings.
///
/// Strings are ordered  [lexicographically](Ord#lexicographical-comparison) by their byte values. This orders Unicode code
/// points based on their positions in the code charts. This is not necessarily the same as
/// "alphabetical" order, which varies by language and locale. Sorting strings according to
/// culturally-accepted standards requires locale-specific data that is outside the scope of
/// the `str` type.
#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for str {
    #[inline]
    fn cmp(&self, other: &str) -> Ordering {
        self.as_bytes().cmp(other.as_bytes())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq for str {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_bytes() == other.as_bytes()
    }
    #[inline]
    fn ne(&self, other: &str) -> bool {
        !(*self).eq(other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Eq for str {}

/// Implements comparison operations on strings.
///
/// Strings are compared [lexicographically](Ord#lexicographical-comparison) by their byte values. This compares Unicode code
/// points based on their positions in the code charts. This is not necessarily the same as
/// "alphabetical" order, which varies by language and locale. Comparing strings according to
/// culturally-accepted standards requires locale-specific data that is outside the scope of
/// the `str` type.
#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd for str {
    #[inline]
    fn partial_cmp(&self, other: &str) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ops::Index<I> for str
where
    I: SliceIndex<str>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ops::IndexMut<I> for str
where
    I: SliceIndex<str>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

#[inline(never)]
#[cold]
#[track_caller]
fn str_index_overflow_fail() -> ! {
    panic!("attempted to index str up to maximum usize");
}

/// Implements substring slicing with syntax `&self[..]` or `&mut self[..]`.
///
/// Returns a slice of the whole string, i.e., returns `&self` or `&mut
/// self`. Equivalent to `&self[0 .. len]` or `&mut self[0 .. len]`. Unlike
/// other indexing operations, this can never panic.
///
/// This operation is *O*(1).
///
/// Prior to 1.20.0, these indexing operations were still supported by
/// direct implementation of `Index` and `IndexMut`.
///
/// Equivalent to `&self[0 .. len]` or `&mut self[0 .. len]`.
#[stable(feature = "str_checked_slicing", since = "1.20.0")]
unsafe impl SliceIndex<str> for ops::RangeFull {
    type Output = str;
    #[inline]
    fn get(self, slice: &str) -> Option<&Self::Output> {
        Some(slice)
    }
    #[inline]
    fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
        Some(slice)
    }
    #[inline]
    unsafe fn get_unchecked(self, slice: *const str) -> *const Self::Output {
        slice
    }
    #[inline]
    unsafe fn get_unchecked_mut(self, slice: *mut str) -> *mut Self::Output {
        slice
    }
    #[inline]
    fn index(self, slice: &str) -> &Self::Output {
        slice
    }
    #[inline]
    fn index_mut(self, slice: &mut str) -> &mut Self::Output {
        slice
    }
}

/// Implements substring slicing with syntax `&self[begin .. end]` or `&mut
/// self[begin .. end]`.
///
/// Returns a slice of the given string from the byte range
/// [`begin`, `end`).
///
/// This operation is *O*(1).
///
/// Prior to 1.20.0, these indexing operations were still supported by
/// direct implementation of `Index` and `IndexMut`.
///
/// # Panics
///
/// Panics if `begin` or `end` does not point to the starting byte offset of
/// a character (as defined by `is_char_boundary`), if `begin > end`, or if
/// `end > len`.
///
/// # Examples
///
/// ```
/// let s = "Löwe 老虎 Léopard";
/// assert_eq!(&s[0 .. 1], "L");
///
/// assert_eq!(&s[1 .. 9], "öwe 老");
///
/// // these will panic:
/// // byte 2 lies within `ö`:
/// // &s[2 ..3];
///
/// // byte 8 lies within `老`
/// // &s[1 .. 8];
///
/// // byte 100 is outside the string
/// // &s[3 .. 100];
/// ```
#[stable(feature = "str_checked_slicing", since = "1.20.0")]
unsafe impl SliceIndex<str> for ops::Range<usize> {
    type Output = str;
    #[inline]
    fn get(self, slice: &str) -> Option<&Self::Output> {
        if self.start <= self.end
            && slice.is_char_boundary(self.start)
            && slice.is_char_boundary(self.end)
        {
            // SAFETY: just checked that `start` and `end` are on a char boundary,
            // and we are passing in a safe reference, so the return value will also be one.
            // We also checked char boundaries, so this is valid UTF-8.
            Some(unsafe { &*self.get_unchecked(slice) })
        } else {
            None
        }
    }
    #[inline]
    fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
        if self.start <= self.end
            && slice.is_char_boundary(self.start)
            && slice.is_char_boundary(self.end)
        {
            // SAFETY: just checked that `start` and `end` are on a char boundary.
            // We know the pointer is unique because we got it from `slice`.
            Some(unsafe { &mut *self.get_unchecked_mut(slice) })
        } else {
            None
        }
    }
    #[inline]
    unsafe fn get_unchecked(self, slice: *const str) -> *const Self::Output {
        let slice = slice as *const [u8];
        // SAFETY: the caller guarantees that `self` is in bounds of `slice`
        // which satisfies all the conditions for `add`.
        let ptr = unsafe { slice.as_ptr().add(self.start) };
        let len = self.end - self.start;
        ptr::slice_from_raw_parts(ptr, len) as *const str
    }
    #[inline]
    unsafe fn get_unchecked_mut(self, slice: *mut str) -> *mut Self::Output {
        let slice = slice as *mut [u8];
        // SAFETY: see comments for `get_unchecked`.
        let ptr = unsafe { slice.as_mut_ptr().add(self.start) };
        let len = self.end - self.start;
        ptr::slice_from_raw_parts_mut(ptr, len) as *mut str
    }
    #[inline]
    fn index(self, slice: &str) -> &Self::Output {
        let (start, end) = (self.start, self.end);
        match self.get(slice) {
            Some(s) => s,
            None => super::slice_error_fail(slice, start, end),
        }
    }
    #[inline]
    fn index_mut(self, slice: &mut str) -> &mut Self::Output {
        // is_char_boundary checks that the index is in [0, .len()]
        // cannot reuse `get` as above, because of NLL trouble
        if self.start <= self.end
            && slice.is_char_boundary(self.start)
            && slice.is_char_boundary(self.end)
        {
            // SAFETY: just checked that `start` and `end` are on a char boundary,
            // and we are passing in a safe reference, so the return value will also be one.
            unsafe { &mut *self.get_unchecked_mut(slice) }
        } else {
            super::slice_error_fail(slice, self.start, self.end)
        }
    }
}

/// Implements substring slicing with syntax `&self[.. end]` or `&mut
/// self[.. end]`.
///
/// Returns a slice of the given string from the byte range \[0, `end`).
/// Equivalent to `&self[0 .. end]` or `&mut self[0 .. end]`.
///
/// This operation is *O*(1).
///
/// Prior to 1.20.0, these indexing operations were still supported by
/// direct implementation of `Index` and `IndexMut`.
///
/// # Panics
///
/// Panics if `end` does not point to the starting byte offset of a
/// character (as defined by `is_char_boundary`), or if `end > len`.
#[stable(feature = "str_checked_slicing", since = "1.20.0")]
unsafe impl SliceIndex<str> for ops::RangeTo<usize> {
    type Output = str;
    #[inline]
    fn get(self, slice: &str) -> Option<&Self::Output> {
        if slice.is_char_boundary(self.end) {
            // SAFETY: just checked that `end` is on a char boundary,
            // and we are passing in a safe reference, so the return value will also be one.
            Some(unsafe { &*self.get_unchecked(slice) })
        } else {
            None
        }
    }
    #[inline]
    fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
        if slice.is_char_boundary(self.end) {
            // SAFETY: just checked that `end` is on a char boundary,
            // and we are passing in a safe reference, so the return value will also be one.
            Some(unsafe { &mut *self.get_unchecked_mut(slice) })
        } else {
            None
        }
    }
    #[inline]
    unsafe fn get_unchecked(self, slice: *const str) -> *const Self::Output {
        let slice = slice as *const [u8];
        let ptr = slice.as_ptr();
        ptr::slice_from_raw_parts(ptr, self.end) as *const str
    }
    #[inline]
    unsafe fn get_unchecked_mut(self, slice: *mut str) -> *mut Self::Output {
        let slice = slice as *mut [u8];
        let ptr = slice.as_mut_ptr();
        ptr::slice_from_raw_parts_mut(ptr, self.end) as *mut str
    }
    #[inline]
    fn index(self, slice: &str) -> &Self::Output {
        let end = self.end;
        match self.get(slice) {
            Some(s) => s,
            None => super::slice_error_fail(slice, 0, end),
        }
    }
    #[inline]
    fn index_mut(self, slice: &mut str) -> &mut Self::Output {
        if slice.is_char_boundary(self.end) {
            // SAFETY: just checked that `end` is on a char boundary,
            // and we are passing in a safe reference, so the return value will also be one.
            unsafe { &mut *self.get_unchecked_mut(slice) }
        } else {
            super::slice_error_fail(slice, 0, self.end)
        }
    }
}

/// Implements substring slicing with syntax `&self[begin ..]` or `&mut
/// self[begin ..]`.
///
/// Returns a slice of the given string from the byte range \[`begin`, `len`).
/// Equivalent to `&self[begin .. len]` or `&mut self[begin .. len]`.
///
/// This operation is *O*(1).
///
/// Prior to 1.20.0, these indexing operations were still supported by
/// direct implementation of `Index` and `IndexMut`.
///
/// # Panics
///
/// Panics if `begin` does not point to the starting byte offset of
/// a character (as defined by `is_char_boundary`), or if `begin > len`.
#[stable(feature = "str_checked_slicing", since = "1.20.0")]
unsafe impl SliceIndex<str> for ops::RangeFrom<usize> {
    type Output = str;
    #[inline]
    fn get(self, slice: &str) -> Option<&Self::Output> {
        if slice.is_char_boundary(self.start) {
            // SAFETY: just checked that `start` is on a char boundary,
            // and we are passing in a safe reference, so the return value will also be one.
            Some(unsafe { &*self.get_unchecked(slice) })
        } else {
            None
        }
    }
    #[inline]
    fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
        if slice.is_char_boundary(self.start) {
            // SAFETY: just checked that `start` is on a char boundary,
            // and we are passing in a safe reference, so the return value will also be one.
            Some(unsafe { &mut *self.get_unchecked_mut(slice) })
        } else {
            None
        }
    }
    #[inline]
    unsafe fn get_unchecked(self, slice: *const str) -> *const Self::Output {
        let slice = slice as *const [u8];
        // SAFETY: the caller guarantees that `self` is in bounds of `slice`
        // which satisfies all the conditions for `add`.
        let ptr = unsafe { slice.as_ptr().add(self.start) };
        let len = slice.len() - self.start;
        ptr::slice_from_raw_parts(ptr, len) as *const str
    }
    #[inline]
    unsafe fn get_unchecked_mut(self, slice: *mut str) -> *mut Self::Output {
        let slice = slice as *mut [u8];
        // SAFETY: identical to `get_unchecked`.
        let ptr = unsafe { slice.as_mut_ptr().add(self.start) };
        let len = slice.len() - self.start;
        ptr::slice_from_raw_parts_mut(ptr, len) as *mut str
    }
    #[inline]
    fn index(self, slice: &str) -> &Self::Output {
        let (start, end) = (self.start, slice.len());
        match self.get(slice) {
            Some(s) => s,
            None => super::slice_error_fail(slice, start, end),
        }
    }
    #[inline]
    fn index_mut(self, slice: &mut str) -> &mut Self::Output {
        if slice.is_char_boundary(self.start) {
            // SAFETY: just checked that `start` is on a char boundary,
            // and we are passing in a safe reference, so the return value will also be one.
            unsafe { &mut *self.get_unchecked_mut(slice) }
        } else {
            super::slice_error_fail(slice, self.start, slice.len())
        }
    }
}

/// Implements substring slicing with syntax `&self[begin ..= end]` or `&mut
/// self[begin ..= end]`.
///
/// Returns a slice of the given string from the byte range
/// [`begin`, `end`]. Equivalent to `&self [begin .. end + 1]` or `&mut
/// self[begin .. end + 1]`, except if `end` has the maximum value for
/// `usize`.
///
/// This operation is *O*(1).
///
/// # Panics
///
/// Panics if `begin` does not point to the starting byte offset of
/// a character (as defined by `is_char_boundary`), if `end` does not point
/// to the ending byte offset of a character (`end + 1` is either a starting
/// byte offset or equal to `len`), if `begin > end`, or if `end >= len`.
#[stable(feature = "inclusive_range", since = "1.26.0")]
unsafe impl SliceIndex<str> for ops::RangeInclusive<usize> {
    type Output = str;
    #[inline]
    fn get(self, slice: &str) -> Option<&Self::Output> {
        if *self.end() == usize::MAX { None } else { self.into_slice_range().get(slice) }
    }
    #[inline]
    fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
        if *self.end() == usize::MAX { None } else { self.into_slice_range().get_mut(slice) }
    }
    #[inline]
    unsafe fn get_unchecked(self, slice: *const str) -> *const Self::Output {
        // SAFETY: the caller must uphold the safety contract for `get_unchecked`.
        unsafe { self.into_slice_range().get_unchecked(slice) }
    }
    #[inline]
    unsafe fn get_unchecked_mut(self, slice: *mut str) -> *mut Self::Output {
        // SAFETY: the caller must uphold the safety contract for `get_unchecked_mut`.
        unsafe { self.into_slice_range().get_unchecked_mut(slice) }
    }
    #[inline]
    fn index(self, slice: &str) -> &Self::Output {
        if *self.end() == usize::MAX {
            str_index_overflow_fail();
        }
        self.into_slice_range().index(slice)
    }
    #[inline]
    fn index_mut(self, slice: &mut str) -> &mut Self::Output {
        if *self.end() == usize::MAX {
            str_index_overflow_fail();
        }
        self.into_slice_range().index_mut(slice)
    }
}

/// Implements substring slicing with syntax `&self[..= end]` or `&mut
/// self[..= end]`.
///
/// Returns a slice of the given string from the byte range \[0, `end`\].
/// Equivalent to `&self [0 .. end + 1]`, except if `end` has the maximum
/// value for `usize`.
///
/// This operation is *O*(1).
///
/// # Panics
///
/// Panics if `end` does not point to the ending byte offset of a character
/// (`end + 1` is either a starting byte offset as defined by
/// `is_char_boundary`, or equal to `len`), or if `end >= len`.
#[stable(feature = "inclusive_range", since = "1.26.0")]
unsafe impl SliceIndex<str> for ops::RangeToInclusive<usize> {
    type Output = str;
    #[inline]
    fn get(self, slice: &str) -> Option<&Self::Output> {
        if self.end == usize::MAX { None } else { (..self.end + 1).get(slice) }
    }
    #[inline]
    fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
        if self.end == usize::MAX { None } else { (..self.end + 1).get_mut(slice) }
    }
    #[inline]
    unsafe fn get_unchecked(self, slice: *const str) -> *const Self::Output {
        // SAFETY: the caller must uphold the safety contract for `get_unchecked`.
        unsafe { (..self.end + 1).get_unchecked(slice) }
    }
    #[inline]
    unsafe fn get_unchecked_mut(self, slice: *mut str) -> *mut Self::Output {
        // SAFETY: the caller must uphold the safety contract for `get_unchecked_mut`.
        unsafe { (..self.end + 1).get_unchecked_mut(slice) }
    }
    #[inline]
    fn index(self, slice: &str) -> &Self::Output {
        if self.end == usize::MAX {
            str_index_overflow_fail();
        }
        (..self.end + 1).index(slice)
    }
    #[inline]
    fn index_mut(self, slice: &mut str) -> &mut Self::Output {
        if self.end == usize::MAX {
            str_index_overflow_fail();
        }
        (..self.end + 1).index_mut(slice)
    }
}

/// Parse a value from a string
///
/// `FromStr`'s [`from_str`] method is often used implicitly, through
/// [`str`]'s [`parse`] method. See [`parse`]'s documentation for examples.
///
/// [`from_str`]: FromStr::from_str
/// [`parse`]: str::parse
///
/// `FromStr` does not have a lifetime parameter, and so you can only parse types
/// that do not contain a lifetime parameter themselves. In other words, you can
/// parse an `i32` with `FromStr`, but not a `&i32`. You can parse a struct that
/// contains an `i32`, but not one that contains an `&i32`.
///
/// # Examples
///
/// Basic implementation of `FromStr` on an example `Point` type:
///
/// ```
/// use std::str::FromStr;
/// use std::num::ParseIntError;
///
/// #[derive(Debug, PartialEq)]
/// struct Point {
///     x: i32,
///     y: i32
/// }
///
/// impl FromStr for Point {
///     type Err = ParseIntError;
///
///     fn from_str(s: &str) -> Result<Self, Self::Err> {
///         let coords: Vec<&str> = s.trim_matches(|p| p == '(' || p == ')' )
///                                  .split(',')
///                                  .collect();
///
///         let x_fromstr = coords[0].parse::<i32>()?;
///         let y_fromstr = coords[1].parse::<i32>()?;
///
///         Ok(Point { x: x_fromstr, y: y_fromstr })
///     }
/// }
///
/// let p = Point::from_str("(1,2)");
/// assert_eq!(p.unwrap(), Point{ x: 1, y: 2} )
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait FromStr: Sized {
    /// The associated error which can be returned from parsing.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Err;

    /// Parses a string `s` to return a value of this type.
    ///
    /// If parsing succeeds, return the value inside [`Ok`], otherwise
    /// when the string is ill-formatted return an error specific to the
    /// inside [`Err`]. The error type is specific to the implementation of the trait.
    ///
    /// # Examples
    ///
    /// Basic usage with [`i32`], a type that implements `FromStr`:
    ///
    /// ```
    /// use std::str::FromStr;
    ///
    /// let s = "5";
    /// let x = i32::from_str(s).unwrap();
    ///
    /// assert_eq!(5, x);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn from_str(s: &str) -> Result<Self, Self::Err>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl FromStr for bool {
    type Err = ParseBoolError;

    /// Parse a `bool` from a string.
    ///
    /// Yields a `Result<bool, ParseBoolError>`, because `s` may or may not
    /// actually be parseable.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::str::FromStr;
    ///
    /// assert_eq!(FromStr::from_str("true"), Ok(true));
    /// assert_eq!(FromStr::from_str("false"), Ok(false));
    /// assert!(<bool as FromStr>::from_str("not even a boolean").is_err());
    /// ```
    ///
    /// Note, in many cases, the `.parse()` method on `str` is more proper.
    ///
    /// ```
    /// assert_eq!("true".parse(), Ok(true));
    /// assert_eq!("false".parse(), Ok(false));
    /// assert!("not even a boolean".parse::<bool>().is_err());
    /// ```
    #[inline]
    fn from_str(s: &str) -> Result<bool, ParseBoolError> {
        match s {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(ParseBoolError),
        }
    }
}
