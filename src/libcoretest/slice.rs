// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::slice::{Found, NotFound};

mod bytes {
    use std::slice::bytes::{position};
    use test::{mod, Bencher};

    #[test]
    fn position_fail() {
        let needle = b'x';
        let mut haystack = Vec::with_capacity(129);
        for _ in range(0u, 128) {
            assert_eq!(None, position(haystack.as_slice(), needle));
            haystack.push(b'a');
        }
    }

    #[test]
    fn position_success() {
        let needle = b'@';

        // Small cases that should not use the `uint` algorithm at all
        assert_eq!(Some(0), position(b"@", needle));
        assert_eq!(Some(1), position(b"a@", needle));
        assert_eq!(Some(2), position(b"aa@", needle));
        assert_eq!(Some(3), position(b"aaa@", needle));
        assert_eq!(Some(4), position(b"aaaa@", needle));
        assert_eq!(Some(5), position(b"aaaaa@", needle));
        assert_eq!(Some(6), position(b"aaaaaa@", needle));

        // This should use the `uint` algorithm assuming that byte literals are
        // aligned.
        assert_eq!(Some(0), position(b"@aaaaaaa", needle));
        assert_eq!(Some(1), position(b"a@aaaaaa", needle));
        assert_eq!(Some(2), position(b"aa@aaaaa", needle));
        assert_eq!(Some(3), position(b"aaa@aaaa", needle));
        assert_eq!(Some(4), position(b"aaaa@aaa", needle));
        assert_eq!(Some(5), position(b"aaaaa@aa", needle));
        assert_eq!(Some(6), position(b"aaaaaa@a", needle));
        assert_eq!(Some(7), position(b"aaaaaaa@", needle));

        // Assuming that byte literals are aligned, this will use the `uint` algorithm
        // but abort before is reaches the needle.
        assert_eq!(Some(8),  position(b"aaaaaaaa@", needle));
        assert_eq!(Some(9),  position(b"aaaaaaaaa@", needle));
        assert_eq!(Some(10), position(b"aaaaaaaaaa@", needle));
        assert_eq!(Some(11), position(b"aaaaaaaaaaa@", needle));
        assert_eq!(Some(12), position(b"aaaaaaaaaaaa@", needle));
        assert_eq!(Some(13), position(b"aaaaaaaaaaaaa@", needle));
        assert_eq!(Some(14), position(b"aaaaaaaaaaaaaa@", needle));
        assert_eq!(Some(15), position(b"aaaaaaaaaaaaaaa@", needle));

        // Random stuff
        let haystacks = [
            b"aouesnthuosneatuh@oaesnuthoeintdeoautnsaoehunithoeasnutohe",
            b"aouesnthuosneatuhoaesnuthoeintdeoautns@aoehunithoeasnutohe",
            b"aouesnthuosneatuhoaesnuthoeintdeoautnsaoehunithoeasnutohe@",
            b"aouesn@huosneatuhoaesnuthoeintdeoautnsaoehunithoeasnutohex",
            b"@aouesnhuosneatuhoaesnuthoeintdeoautnsaoehunithoeasnutohex",
            b"aaouesnhuosneatuhoaesnuthoeintde@autnsaoehunithoeasnutohex",
        ];
        for &haystack in haystacks.iter() {
            assert_eq!(haystack.iter().position(|&c| c == needle),
                       position(haystack, needle));
        }
    }

    #[bench]
    fn position_bench(b: &mut Bencher) {
        let x = Vec::from_elem(1024 * 64, 0);
        b.iter(|| test::black_box(position(x.as_slice(), 1)));
    }
}

#[test]
fn binary_search_not_found() {
    let b = [1i, 2, 4, 6, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&6)) == Found(3));
    let b = [1i, 2, 4, 6, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&5)) == NotFound(3));
    let b = [1i, 2, 4, 6, 7, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&6)) == Found(3));
    let b = [1i, 2, 4, 6, 7, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&5)) == NotFound(3));
    let b = [1i, 2, 4, 6, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&8)) == Found(4));
    let b = [1i, 2, 4, 6, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&7)) == NotFound(4));
    let b = [1i, 2, 4, 6, 7, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&8)) == Found(5));
    let b = [1i, 2, 4, 5, 6, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&7)) == NotFound(5));
    let b = [1i, 2, 4, 5, 6, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&0)) == NotFound(0));
    let b = [1i, 2, 4, 5, 6, 8];
    assert!(b.binary_search(|v| v.cmp(&9)) == NotFound(6));
}
