// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15883


static FOO: u8 = b'\xF0';
static BAR: &'static [u8] = b"a\xF0\t";
static BAZ: &'static [u8] = br"a\n";

pub fn main() {
    assert_eq!(b'a', 97u8);
    assert_eq!(b'\n', 10u8);
    assert_eq!(b'\r', 13u8);
    assert_eq!(b'\t', 9u8);
    assert_eq!(b'\\', 92u8);
    assert_eq!(b'\'', 39u8);
    assert_eq!(b'\"', 34u8);
    assert_eq!(b'\0', 0u8);
    assert_eq!(b'\xF0', 240u8);
    assert_eq!(FOO, 240u8);

    match 42 {
        b'*' => {},
        _ => fail!()
    }

    match 100 {
        b'a' .. b'z' => {},
        _ => fail!()
    }

    assert_eq!(b"a\n\r\t\\\'\"\0\xF0",
               &[97u8, 10u8, 13u8, 9u8, 92u8, 39u8, 34u8, 0u8, 240u8]);
    assert_eq!(b"a\
                 b", &[97u8, 98u8]);
    assert_eq!(BAR, &[97u8, 240u8, 9u8]);

    match &[97u8, 10u8] {
        b"a\n" => {},
        _ => fail!(),
    }

    let buf = vec!(97u8, 98, 99, 100);
    assert_eq!(match buf.slice(0, 3) {
         b"def" => 1u,
         b"abc" => 2u,
         _ => 3u
    }, 2);

    assert_eq!(BAZ, &[97u8, 92u8, 110u8]);
    assert_eq!(br"a\n", &[97u8, 92u8, 110u8]);
    assert_eq!(br"a\n", b"a\\n");
    assert_eq!(br###"a"##b"###, &[97u8, 34u8, 35u8, 35u8, 98u8]);
    assert_eq!(br###"a"##b"###, b"a\"##b");
}
