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
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    x + 1
}

#[cfg(target_pointer_width="32")]
pub fn round_up_to_power_of_two(mut x: usize) -> usize {
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x + 1
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
    assert_eq!(1, round_up_to_power_of_two(1));
    assert_eq!(64, round_up_to_power_of_two(33));
    assert_eq!(256, round_up_to_power_of_two(256));
}
