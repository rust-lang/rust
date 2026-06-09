#![feature(slice_internals)]

use core::slice::memchr::{memchr, memrchr};

// test fallback implementations on all targets
fn matches_one() {
    assert_eq!(Some(0), memchr(b'a', b"a"));
}

fn matches_begin() {
    assert_eq!(Some(0), memchr(b'a', b"aaaa"));
}

fn matches_end() {
    assert_eq!(Some(4), memchr(b'z', b"aaaaz"));
}

fn matches_nul() {
    assert_eq!(Some(4), memchr(b'\x00', b"aaaa\x00"));
}

fn matches_past_nul() {
    assert_eq!(Some(5), memchr(b'z', b"aaaa\x00z"));
}

fn no_match_empty() {
    assert_eq!(None, memchr(b'a', b""));
}

fn no_match() {
    assert_eq!(None, memchr(b'a', b"xyz"));
}

fn matches_one_reversed() {
    assert_eq!(Some(0), memrchr(b'a', b"a"));
}

fn matches_begin_reversed() {
    assert_eq!(Some(3), memrchr(b'a', b"aaaa"));
}

fn matches_end_reversed() {
    assert_eq!(Some(0), memrchr(b'z', b"zaaaa"));
}

fn matches_nul_reversed() {
    assert_eq!(Some(4), memrchr(b'\x00', b"aaaa\x00"));
}

fn matches_past_nul_reversed() {
    assert_eq!(Some(0), memrchr(b'z', b"z\x00aaaa"));
}

fn no_match_empty_reversed() {
    assert_eq!(None, memrchr(b'a', b""));
}

fn no_match_reversed() {
    assert_eq!(None, memrchr(b'a', b"xyz"));
}

fn each_alignment_reversed() {
    let mut data = [1u8; 64];
    let needle = 2;
    let pos = 40;
    data[pos] = needle;
    for start in 0..16 {
        assert_eq!(Some(pos - start), memrchr(needle, &data[start..]));
    }
}

fn main() {
    matches_one();
    matches_begin();
    matches_end();
    matches_nul();
    matches_past_nul();
    no_match_empty();
    no_match();

    matches_one_reversed();
    matches_begin_reversed();
    matches_end_reversed();
    matches_nul_reversed();
    matches_past_nul_reversed();
    no_match_empty_reversed();
    no_match_reversed();
    each_alignment_reversed();
}
