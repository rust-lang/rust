// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast::Visibility;

pub trait FindUncommented {
    fn find_uncommented(&self, pat: &str) -> Option<usize>;
}

impl FindUncommented for str {
    fn find_uncommented(&self, pat: &str) -> Option<usize> {
        let mut needle_iter = pat.chars();
        let mut possible_comment = false;

        for (i, b) in self.char_indices() {
            match needle_iter.next() {
                Some(c) => {
                    if b != c {
                        needle_iter = pat.chars();
                    }
                },
                None => return Some(i - pat.len())
            }

            if possible_comment {
                if b == '/' {
                    return self[(i+1)..].find('\n')
                                        .and_then(|end| {
                                            self[(end + i + 2)..].find_uncommented(pat)
                                                                 .map(|idx| idx + end + i + 2)
                                        });
                } else if b == '*' {
                    return self[(i+1)..].find("*/")
                                        .and_then(|end| {
                                            self[(end + i + 3)..].find_uncommented(pat)
                                                                 .map(|idx| idx + end + i + 3)
                                        });
                } else {
                    possible_comment = false;
                }
            } else {
                possible_comment = b == '/';
            }
        }

        // Handle case where the pattern is a suffix of the search string
        match needle_iter.next() {
            Some(_) => None,
            None => Some(self.len() - pat.len())
        }
    }
}

#[test]
fn test_find_uncommented() {
    fn check(haystack: &str, needle: &str, expected: Option<usize>) {
        assert_eq!(expected, haystack.find_uncommented(needle));
    }

    check("/*//*/test", "test", Some(6));
    check("//test\ntest", "test", Some(7));
    check("/* comment only */", "whatever", None);
    check("/* comment */ some text /* more commentary */ result", "result", Some(46));
    check("sup // sup", "p", Some(2));
    check("sup", "x", None);
    check("π? /**/ π is nice!", "π is nice", Some(9));
    check("/*sup yo? \n sup*/ sup", "p", Some(20));
    check("hel/*lohello*/lo", "hello", None);
    check("acb", "ab", None);
}

#[inline]
pub fn prev_char(s: &str, mut i: usize) -> usize {
    if i == 0 { return 0; }

    i -= 1;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[inline]
pub fn next_char(s: &str, mut i: usize) -> usize {
    if i >= s.len() { return s.len(); }

    while !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

#[inline]
pub fn make_indent(width: usize) -> String {
    let mut indent = String::with_capacity(width);
    for _ in 0..width {
        indent.push(' ')
    }
    indent
}

#[inline]
pub fn format_visibility(vis: Visibility) -> &'static str {
    match vis {
        Visibility::Public => "pub ",
        Visibility::Inherited => ""
    }
}

#[inline]
#[cfg(target_pointer_width="64")]
// Based on the trick layed out at
// http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
pub fn round_up_to_power_of_two(mut x: usize) -> usize {
    x = x.wrapping_sub(1);
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    x.wrapping_add(1)
}

#[inline]
#[cfg(target_pointer_width="32")]
pub fn round_up_to_power_of_two(mut x: usize) -> usize {
    x = x.wrapping_sub(1);
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x.wrapping_add(1)
}

// Macro for deriving implementations of Decodable for enums
#[macro_export]
macro_rules! impl_enum_decodable {
    ( $e:ident, $( $x:ident ),* ) => {
        impl ::rustc_serialize::Decodable for $e {
            fn decode<D: ::rustc_serialize::Decoder>(d: &mut D) -> Result<Self, D::Error> {
                let s = try!(d.read_str());
                match &*s {
                    $(
                        stringify!($x) => Ok($e::$x),
                    )*
                    _ => Err(d.error("Bad variant")),
                }
            }
        }
    };
}

#[test]
fn power_rounding() {
    assert_eq!(0, round_up_to_power_of_two(0));
    assert_eq!(1, round_up_to_power_of_two(1));
    assert_eq!(64, round_up_to_power_of_two(33));
    assert_eq!(256, round_up_to_power_of_two(256));
}
