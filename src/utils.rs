// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast::{self, Visibility, Attribute, MetaItem, MetaItem_};
use syntax::codemap::{CodeMap, Span, BytePos};

use comment::FindUncommented;

use SKIP_ANNOTATION;

// Computes the length of a string's last line, minus offset.
#[inline]
pub fn extra_offset(text: &str, offset: usize) -> usize {
    match text.rfind('\n') {
        // 1 for newline character
        Some(idx) => text.len() - idx - 1 - offset,
        None => text.len(),
    }
}

#[inline]
pub fn span_after(original: Span, needle: &str, codemap: &CodeMap) -> BytePos {
    let snippet = codemap.span_to_snippet(original).unwrap();

    original.lo + BytePos(snippet.find_uncommented(needle).unwrap() as u32 + 1)
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
        Visibility::Inherited => "",
    }
}

#[inline]
pub fn format_mutability(mutability: ast::Mutability) -> &'static str {
    match mutability {
        ast::Mutability::MutMutable => "mut ",
        ast::Mutability::MutImmutable => "",
    }
}

// The width of the first line in s.
#[inline]
pub fn first_line_width(s: &str) -> usize {
    match s.find('\n') {
        Some(n) => n,
        None => s.len(),
    }
}

// The width of the last line in s.
#[inline]
pub fn last_line_width(s: &str) -> usize {
    match s.rfind('\n') {
        Some(n) => s.len() - n - 1,
        None => s.len(),
    }
}

#[inline]
fn is_skip(meta_item: &MetaItem) -> bool {
    match meta_item.node {
        MetaItem_::MetaWord(ref s) => *s == SKIP_ANNOTATION,
        _ => false,
    }
}

#[inline]
pub fn contains_skip(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| is_skip(&a.node.value))
}

// Find the end of a TyParam
#[inline]
pub fn end_typaram(typaram: &ast::TyParam) -> BytePos {
    typaram.bounds.last().map(|bound| match *bound {
        ast::RegionTyParamBound(ref lt) => lt.span,
        ast::TraitTyParamBound(ref prt, _) => prt.span,
    }).unwrap_or(typaram.span).hi
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

        impl ::std::str::FromStr for $e {
            type Err = &'static str;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match &*s {
                    $(
                        stringify!($x) => Ok($e::$x),
                    )*
                    _ => Err("Bad variant"),
                }
            }
        }
    };
}

// Same as try!, but for Option
#[macro_export]
macro_rules! try_opt {
    ($expr:expr) => (match $expr {
        Some(val) => val,
        None => { return None; }
    })
}

#[test]
fn power_rounding() {
    assert_eq!(0, round_up_to_power_of_two(0));
    assert_eq!(1, round_up_to_power_of_two(1));
    assert_eq!(64, round_up_to_power_of_two(33));
    assert_eq!(256, round_up_to_power_of_two(256));
}
