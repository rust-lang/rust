// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::str;
use core::uint;
use core::vec;

struct Quad {
    a: u32,
    b: u32,
    c: u32,
    d: u32
}

pub fn md4(msg: &[u8]) -> Quad {
    // subtle: if orig_len is merely uint, then the code below
    // which performs shifts by 32 bits or more has undefined
    // results.
    let orig_len: u64 = (msg.len() * 8u) as u64;

    // pad message
    let mut msg = vec::append(vec::to_owned(msg), [0x80u8]);
    let mut bitlen = orig_len + 8u64;
    while (bitlen + 64u64) % 512u64 > 0u64 {
        msg.push(0u8);
        bitlen += 8u64;
    }

    // append length
    let mut i = 0u64;
    while i < 8u64 {
        msg.push((orig_len >> (i * 8u64)) as u8);
        i += 1u64;
    }

    let mut a = 0x67452301u32;
    let mut b = 0xefcdab89u32;
    let mut c = 0x98badcfeu32;
    let mut d = 0x10325476u32;

    fn rot(r: int, x: u32) -> u32 {
        let r = r as u32;
        (x << r) | (x >> (32u32 - r))
    }

    let mut i = 0u;
    let e = msg.len();
    let mut x = vec::from_elem(16u, 0u32);
    while i < e {
        let aa = a, bb = b, cc = c, dd = d;

        let mut j = 0u, base = i;
        while j < 16u {
            x[j] = (msg[base] as u32) + (msg[base + 1u] as u32 << 8u32) +
                (msg[base + 2u] as u32 << 16u32) +
                (msg[base + 3u] as u32 << 24u32);
            j += 1u; base += 4u;
        }

        let mut j = 0u;
        while j < 16u {
            a = rot(3, a + ((b & c) | (!b & d)) + x[j]);
            j += 1u;
            d = rot(7, d + ((a & b) | (!a & c)) + x[j]);
            j += 1u;
            c = rot(11, c + ((d & a) | (!d & b)) + x[j]);
            j += 1u;
            b = rot(19, b + ((c & d) | (!c & a)) + x[j]);
            j += 1u;
        }

        let mut j = 0u;
        let q = 0x5a827999u32;
        while j < 4u {
            a = rot(3, a + ((b & c) | ((b & d) | (c & d))) + x[j] + q);
            d = rot(5, d + ((a & b) | ((a & c) | (b & c))) + x[j + 4u] + q);
            c = rot(9, c + ((d & a) | ((d & b) | (a & b))) + x[j + 8u] + q);
            b = rot(13, b + ((c & d) | ((c & a) | (d & a))) + x[j + 12u] + q);
            j += 1u;
        }

        let mut j = 0u;
        let q = 0x6ed9eba1u32;
        while j < 8u {
            let jj = if j > 2u { j - 3u } else { j };
            a = rot(3, a + (b ^ c ^ d) + x[jj] + q);
            d = rot(9, d + (a ^ b ^ c) + x[jj + 8u] + q);
            c = rot(11, c + (d ^ a ^ b) + x[jj + 4u] + q);
            b = rot(15, b + (c ^ d ^ a) + x[jj + 12u] + q);
            j += 2u;
        }

        a += aa; b += bb; c += cc; d += dd;
        i += 64u;
    }
    return Quad {a: a, b: b, c: c, d: d};
}

pub fn md4_str(msg: &[u8]) -> ~str {
    let Quad {a, b, c, d} = md4(msg);
    fn app(a: u32, b: u32, c: u32, d: u32, f: &fn(u32)) {
        f(a); f(b); f(c); f(d);
    }
    let mut result = ~"";
    do app(a, b, c, d) |u| {
        let mut i = 0u32;
        while i < 4u32 {
            let byte = (u >> (i * 8u32)) as u8;
            if byte <= 16u8 { result += ~"0"; }
            result += uint::to_str_radix(byte as uint, 16u);
            i += 1u32;
        }
    }
    result
}

pub fn md4_text(msg: &str) -> ~str { md4_str(str::to_bytes(msg)) }

#[test]
fn test_md4() {
    assert_eq!(md4_text(~""), ~"31d6cfe0d16ae931b73c59d7e0c089c0");
    assert_eq!(md4_text(~"a"), ~"bde52cb31de33e46245e05fbdbd6fb24");
    assert_eq!(md4_text(~"abc"), ~"a448017aaf21d8525fc10ae87aa6729d");
    assert!(md4_text(~"message digest") ==
        ~"d9130a8164549fe818874806e1c7014b");
    assert!(md4_text(~"abcdefghijklmnopqrstuvwxyz") ==
        ~"d79e1c308aa5bbcdeea8ed63df412da9");
    assert!(md4_text(
        ~"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\
        0123456789") == ~"043f8582f241db351ce627e153e7f0e4");
    assert!(md4_text(~"1234567890123456789012345678901234567890123456789\
                     0123456789012345678901234567890") ==
        ~"e33b4ddc9c38f2199c3e7b164fcc0536");
}
