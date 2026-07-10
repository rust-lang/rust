use core::{fmt, ptr, slice};

use super::{String, Vec};

/// An iterator which uses a closure to determine if a character should be removed.
///
/// This struct is created by [`String::extract_if`].
/// See its documentation for more.
///
/// # Example
///
/// ```
/// #![feature(string_extract_if)]
/// let mut s = "Hello! Привет!　你好！".to_string();
/// let iter: std::string::ExtractIf<'_, _> = s.extract_if(|c| c.len_utf8() > 2);
/// ```
#[unstable(feature = "string_extract_if", issue = "154318")]
#[must_use = "iterators are lazy and do nothing unless consumed; \
    use `retain` or `extract_if().for_each(drop)` to remove and discard characters"]
pub struct ExtractIf<'a, F> {
    /// The underlying vector of the original [`String`]. We set its length to zero in [`ExtractIf::new`]
    /// (to prevent invalid UTF-8 from being exposed if this struct is leaked before the iteration is complete)
    /// and then gradually increase the `len` as this vector's prefix keeps filling with valid UTF-8.
    /// The `len` is finally adjusted in [`ExtractIf::drop`].
    valid_prefix: &'a mut Vec<u8>,

    /// During the iteration, the underlying vector's consists of:
    /// - A valid UTF-8 prefix (`valid_prefix.len()` bytes)
    ///   of characters that we iterated over and didn't extract.
    /// - A middle portion of `bytes_removed` initialized bytes that might not be valid UTF-8.
    /// - A valid UTF-8 suffix (`old_len - bytes_removed - valid_prefix.len()` bytes)
    ///   of characters that we have not iterated over yet.
    /// - Potentially some spare capacity (`valid_prefix.capacity() - old_len`). We never touch it.
    ///
    /// The above (together with the fact that `valid_prefix.len() + bytes_removed <= old_len`,
    /// that `old_len` is never changed, and that we never reallocate the vector)
    /// is essentially this structure's invariant.
    bytes_removed: usize,

    /// The (byte) length of the original [`String`] prior to draining.
    /// Sadly, we need to hold on to a copy of it,
    /// because we temporarily lower the [`String`]'s length during iteration
    /// to prevent invalid UTF-8 from showing up.
    /// This field is never changed.
    old_len: usize,

    /// The filter test predicate.
    pred: F,
}

impl<'a, F> ExtractIf<'a, F> {
    pub(super) fn new(string: &'a mut String, pred: F) -> Self {
        let valid_prefix = &mut string.vec;
        let old_len = valid_prefix.len();
        unsafe { valid_prefix.set_len(0) };
        ExtractIf { valid_prefix, bytes_removed: 0, old_len, pred }
    }
}

#[unstable(feature = "string_extract_if", issue = "154318")]
impl<F> Iterator for ExtractIf<'_, F>
where
    F: FnMut(char) -> bool,
{
    type Item = char;

    fn next(&mut self) -> Option<char> {
        loop {
            // `valid_prefix` actually has `old_len` initialized bytes of memory in it,
            // but sadly we can't access them using `get_unchecked`
            // because it prohibits accesses outside of `0..valid_prefix.len()` range.
            // So we have to do some pointer aritmetics here instead.

            // Have to hold on to a copy of this to not materialize any `&self.valid_prefx`
            // while still using the pointer (and its derivatives) we got from `.as_mut_ptr()`.
            let valid_prefix_len = self.valid_prefix.len();

            let tail = {
                let bytes = self.valid_prefix.as_mut_ptr();
                // SAFETY: all of these bytes were initialized by the original string.
                let bytes = unsafe { slice::from_raw_parts_mut(bytes, self.old_len) };
                // SAFETY: by our invariant, `valid_prefix.len() <= old_len`.
                unsafe { bytes.get_unchecked_mut(valid_prefix_len..) }
            };

            let c = unsafe {
                // SAFETY: by our invariant, `bytes_removed <= old_len - valid_prefix.len()`.
                let valid_tail = tail.get_unchecked(self.bytes_removed..);
                // SAFETY: we have not touched these bytes before, so they remain valid UTF-8.
                str::from_utf8_unchecked(valid_tail)
            }
            .chars() // FIXME(str_first_last_char): replace this with `first_char`
            .next()?;

            let char_len = c.len_utf8();
            if (self.pred)(c) {
                self.bytes_removed += char_len;
                return Some(c);
            } else {
                let tail = tail.as_mut_ptr();
                unsafe {
                    // SAFETY: we just had a `&mut [u8]` covering both `src` and `dst`, so they are valid.
                    //         `src` had a prefix of UTF-8-encoded `char` `c`, whose `.len_utf8()` is `char_len`,
                    //         and `dst` precedes `src` in the slice mentioned above, so all of this is still in bounds.
                    ptr::copy(tail.add(self.bytes_removed), tail, char_len);

                    // SAFETY: we just appended `char_len` more bytes of valid UTF-8
                    // to the existing `valid_prefix_len` bytes of valid UTF-8.
                    self.valid_prefix.set_len(valid_prefix_len + char_len);
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.old_len - self.valid_prefix.len() - self.bytes_removed))
    }
}

#[unstable(feature = "string_extract_if", issue = "154318")]
impl<F> Drop for ExtractIf<'_, F> {
    fn drop(&mut self) {
        let hole_size = self.bytes_removed;
        let valid_prefix_len = self.valid_prefix.len();
        let valid_tail_len = self.old_len - valid_prefix_len - hole_size;
        if valid_tail_len > 0 {
            unsafe {
                let tail = self.valid_prefix.as_mut_ptr().add(valid_prefix_len);
                ptr::copy(tail.add(hole_size), tail, valid_tail_len)
            }
            // SAFETY: we had a prefix of `valid_prefix_len` bytes of valid UTF-8,
            // and a suffix of `valid_tail_len` bytes of valid UTF-8.
            // We just `prt::copy`'ed the suffix to come right after the prefix, so it's safe to bump the length.
            unsafe { self.valid_prefix.set_len(valid_prefix_len + valid_tail_len) }
        }
    }
}

#[unstable(feature = "string_extract_if", issue = "154318")]
impl<F> fmt::Debug for ExtractIf<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let valid_prefix_len = self.valid_prefix.len();
        let bytes = self.valid_prefix.as_ptr();
        // SAFETY: all of this memory was initialized by the original string.
        let bytes = unsafe { slice::from_raw_parts(bytes, self.old_len) };
        // SAFETY: by invariant, `valid_prefix_len <= old_len`.
        let (valid_prefix, tail) = unsafe { bytes.split_at_unchecked(valid_prefix_len) };
        // SAFETY: by our invariant, `bytes_removed <= old_len - valid_prefix.len()`.
        let (_hole, valid_suffix) = unsafe { tail.split_at_unchecked(self.bytes_removed) };
        // SAFETY: by our invariant, this prefix is valid UTF-8.
        let valid_prefix = unsafe { str::from_utf8_unchecked(valid_prefix) };
        // SAFETY: by our invariant, this suffix is valid UTF-8.
        let valid_suffix = unsafe { str::from_utf8_unchecked(valid_suffix) };
        f.debug_struct("ExtractIf")
            .field("retained", &valid_prefix)
            .field("remainder", &valid_suffix)
            .finish_non_exhaustive()
    }
}
