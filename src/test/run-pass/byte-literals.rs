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


static FOO: u8 = b'\xF0';
static BAR: &'static [u8] = b"a\xF0\t";
static BAR_FIXED: &'static [u8; 3] = b"a\xF0\t";
static BAZ: &'static [u8] = br"a\n";

pub fn main() {
    let bar: &'static [u8] = b"a\xF0\t";
    let bar_fixed: &'static [u8; 3] = b"a\xF0\t";

    assert_eq!(b'a', 97u8);
    assert_eq!(b'\n', 10u8);
    assert_eq!(b'\r', 13u8);
    assert_eq!(b'\t', 9u8);
    assert_eq!(b'\e', 0x1B);
    assert_eq!(b'\\', 92u8);
    assert_eq!(b'\'', 39u8);
    assert_eq!(b'\"', 34u8);
    assert_eq!(b'\0', 0u8);
    assert_eq!(b'\xF0', 240u8);
    assert_eq!(FOO, 240u8);

    match 42 {
        b'*' => {},
        _ => panic!()
    }

    match 100 {
        b'a' ... b'z' => {},
        _ => panic!()
    }

    let expected: &[_] = &[97u8, 10u8, 13u8, 9u8, 92u8, 39u8, 34u8, 0u8, 240u8, 0x1B];
    assert_eq!(b"a\n\r\t\\\'\"\0\xF0\e", expected);
    let expected: &[_] = &[97u8, 98u8];
    assert_eq!(b"a\
                 b", expected);
    let expected: &[_] = &[97u8, 240u8, 9u8];
    assert_eq!(BAR, expected);
    assert_eq!(BAR_FIXED, expected);
    assert_eq!(bar, expected);
    assert_eq!(bar_fixed, expected);

    let val = &[97u8, 10u8];
    match val {
        b"a\n" => {},
        _ => panic!(),
    }

    let buf = vec!(97u8, 98, 99, 100);
    assert_eq!(match &buf[0..3] {
         b"def" => 1,
         b"abc" => 2,
         _ => 3
    }, 2);

    let expected: &[_] = &[97u8, 92u8, 110u8];
    assert_eq!(BAZ, expected);
    let expected: &[_] = &[97u8, 92u8, 110u8];
    assert_eq!(br"a\n", expected);
    assert_eq!(br"a\n", b"a\\n");
    let expected: &[_] = &[97u8, 34u8, 35u8, 35u8, 98u8];
    assert_eq!(br###"a"##b"###, expected);
    assert_eq!(br###"a"##b"###, b"a\"##b");
}
