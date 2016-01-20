// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Original implementation taken from rust-memchr
// Copyright 2015 Andrew Gallant, bluss and Nicolas Koch

#[cfg(not(target_os = "linux"))]
use slice::bytes;

/// A safe interface to `memchr`.
///
/// Returns the index corresponding to the first occurrence of `needle` in
/// `haystack`, or `None` if one is not found.
///
/// memchr reduces to super-optimized machine code at around an order of
/// magnitude faster than `haystack.iter().position(|&b| b == needle)`.
/// (See benchmarks.)
///
/// # Example
///
/// This shows how to find the first position of a byte in a byte string.
///
/// ```rust,ignore
/// use memchr::memchr;
///
/// let haystack = b"the quick brown fox";
/// assert_eq!(memchr(b'k', haystack), Some(8));
/// ```
pub fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    // libc memchr
    #[cfg(not(target_os = "windows"))]
    fn memchr_specific(needle: u8, haystack: &[u8]) -> Option<usize> {
        use libc;

        let p = unsafe {
            libc::memchr(
                haystack.as_ptr() as *const libc::c_void,
                needle as libc::c_int,
                haystack.len() as libc::size_t)
        };
        if p.is_null() {
            None
        } else {
            Some(p as usize - (haystack.as_ptr() as usize))
        }
    }

    // use fallback on windows, since it's faster
    #[cfg(target_os = "windows")]
    fn memchr_specific(needle: u8, haystack: &[u8]) -> Option<usize> {
        bytes::find_byte(needle, haystack)
    }

    memchr_specific(needle, haystack)
}

/// A safe interface to `memrchr`.
///
/// Returns the index corresponding to the last occurrence of `needle` in
/// `haystack`, or `None` if one is not found.
///
/// # Example
///
/// This shows how to find the last position of a byte in a byte string.
///
/// ```rust,ignore
/// use memchr::memrchr;
///
/// let haystack = b"the quick brown fox";
/// assert_eq!(memrchr(b'o', haystack), Some(17));
/// ```
pub fn memrchr(needle: u8, haystack: &[u8]) -> Option<usize> {

    #[cfg(target_os = "linux")]
    fn memrchr_specific(needle: u8, haystack: &[u8]) -> Option<usize> {
        use libc;

        // GNU's memrchr() will - unlike memchr() - error if haystack is empty.
        if haystack.is_empty() {return None}
        let p = unsafe {
            libc::memrchr(
                haystack.as_ptr() as *const libc::c_void,
                needle as libc::c_int,
                haystack.len() as libc::size_t)
        };
        if p.is_null() {
            None
        } else {
            Some(p as usize - (haystack.as_ptr() as usize))
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn memrchr_specific(needle: u8, haystack: &[u8]) -> Option<usize> {
        bytes::rfind_bytes(needle, haystack)
    }

    memrchr_specific(needle, haystack)
}

#[cfg(test)]
mod tests {
    // test the implementations for the current plattform
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
}
