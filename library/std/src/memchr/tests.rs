// Original implementation taken from rust-memchr.
// Copyright 2015 Andrew Gallant, bluss and Nicolas Koch

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
