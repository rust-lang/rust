// Original implementation taken from rust-memchr.
// Copyright 2015 Andrew Gallant, bluss and Nicolas Koch

/// A safe interface to `memchr`.
///
/// Returns the index corresponding to the first occurrence of `needle` in
/// `haystack`, or `None` if one is not found.
///
/// memchr reduces to super-optimized machine code at around an order of
/// magnitude faster than `haystack.iter().position(|&b| b == needle)`.
/// (See benchmarks.)
///
/// # Examples
///
/// This shows how to find the first position of a byte in a byte string.
///
/// ```ignore (cannot-doctest-private-modules)
/// use memchr::memchr;
///
/// let haystack = b"the quick brown fox";
/// assert_eq!(memchr(b'k', haystack), Some(8));
/// ```
#[inline]
pub fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    crate::sys::memchr::memchr(needle, haystack)
}

/// A safe interface to `memrchr`.
///
/// Returns the index corresponding to the last occurrence of `needle` in
/// `haystack`, or `None` if one is not found.
///
/// # Examples
///
/// This shows how to find the last position of a byte in a byte string.
///
/// ```ignore (cannot-doctest-private-modules)
/// use memchr::memrchr;
///
/// let haystack = b"the quick brown fox";
/// assert_eq!(memrchr(b'o', haystack), Some(17));
/// ```
#[inline]
pub fn memrchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    crate::sys::memchr::memrchr(needle, haystack)
}

#[cfg(test)]
mod tests {
    // test the implementations for the current platform
    use super::{memchr, memrchr};

    #[test]
    fn matches_one() {
        assert_eq!(Some(0), memchr(b'a', b"a"));
    }

    #[test]
    fn matches_begin() {
        assert_eq!(Some(0), memchr(b'a', b"aaaa"));
    }

    #[test]
    fn matches_end() {
        assert_eq!(Some(4), memchr(b'z', b"aaaaz"));
    }

    #[test]
    fn matches_nul() {
        assert_eq!(Some(4), memchr(b'\x00', b"aaaa\x00"));
    }

    #[test]
    fn matches_past_nul() {
        assert_eq!(Some(5), memchr(b'z', b"aaaa\x00z"));
    }

    #[test]
    fn no_match_empty() {
        assert_eq!(None, memchr(b'a', b""));
    }

    #[test]
    fn no_match() {
        assert_eq!(None, memchr(b'a', b"xyz"));
    }

    #[test]
    fn matches_one_reversed() {
        assert_eq!(Some(0), memrchr(b'a', b"a"));
    }

    #[test]
    fn matches_begin_reversed() {
        assert_eq!(Some(3), memrchr(b'a', b"aaaa"));
    }

    #[test]
    fn matches_end_reversed() {
        assert_eq!(Some(0), memrchr(b'z', b"zaaaa"));
    }

    #[test]
    fn matches_nul_reversed() {
        assert_eq!(Some(4), memrchr(b'\x00', b"aaaa\x00"));
    }

    #[test]
    fn matches_past_nul_reversed() {
        assert_eq!(Some(0), memrchr(b'z', b"z\x00aaaa"));
    }

    #[test]
    fn no_match_empty_reversed() {
        assert_eq!(None, memrchr(b'a', b""));
    }

    #[test]
    fn no_match_reversed() {
        assert_eq!(None, memrchr(b'a', b"xyz"));
    }

    #[test]
    fn each_alignment() {
        let mut data = [1u8; 64];
        let needle = 2;
        let pos = 40;
        data[pos] = needle;
        for start in 0..16 {
            assert_eq!(Some(pos - start), memchr(needle, &data[start..]));
        }
    }
}
