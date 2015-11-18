// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::Ordering;

use syntax::ast::{self, Visibility, Attribute, MetaItem, MetaItem_};
use syntax::codemap::{CodeMap, Span, BytePos};

use Indent;
use comment::FindUncommented;
use rewrite::{Rewrite, RewriteContext};

use SKIP_ANNOTATION;

// Computes the length of a string's last line, minus offset.
#[inline]
pub fn extra_offset(text: &str, offset: Indent) -> usize {
    match text.rfind('\n') {
        // 1 for newline character
        Some(idx) => text.len() - idx - 1 - offset.width(),
        None => text.len(),
    }
}

#[inline]
pub fn span_after(original: Span, needle: &str, codemap: &CodeMap) -> BytePos {
    let snippet = codemap.span_to_snippet(original).unwrap();
    let offset = snippet.find_uncommented(needle).unwrap() + needle.len();

    original.lo + BytePos(offset as u32)
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
        MetaItem_::MetaList(ref s, ref l) => {
            *s == "cfg_attr" && l.len() == 2 && is_skip(&l[1])
        }
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
    typaram.bounds
           .last()
           .map(|bound| {
               match *bound {
                   ast::RegionTyParamBound(ref lt) => lt.span,
                   ast::TraitTyParamBound(ref prt, _) => prt.span,
               }
           })
           .unwrap_or(typaram.span)
           .hi
}

#[inline]
pub fn semicolon_for_expr(expr: &ast::Expr) -> bool {
    match expr.node {
        ast::Expr_::ExprRet(..) |
        ast::Expr_::ExprAgain(..) |
        ast::Expr_::ExprBreak(..) => true,
        _ => false,
    }
}

#[inline]
pub fn semicolon_for_stmt(stmt: &ast::Stmt) -> bool {
    match stmt.node {
        ast::Stmt_::StmtSemi(ref expr, _) => {
            match expr.node {
                ast::Expr_::ExprWhile(..) |
                ast::Expr_::ExprWhileLet(..) |
                ast::Expr_::ExprLoop(..) |
                ast::Expr_::ExprForLoop(..) => false,
                _ => true,
            }
        }
        ast::Stmt_::StmtExpr(..) => false,
        _ => true,
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

        impl ::config::ConfigType for $e {
            fn get_variant_names() -> String {
                let mut variants = Vec::new();
                $(
                    variants.push(stringify!($x));
                )*
                format!("[{}]", variants.join("|"))
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

// Wraps string-like values in an Option. Returns Some when the string adheres
// to the Rewrite constraints defined for the Rewrite trait and else otherwise.
pub fn wrap_str<S: AsRef<str>>(s: S, max_width: usize, width: usize, offset: Indent) -> Option<S> {
    {
        let snippet = s.as_ref();

        if !snippet.contains('\n') && snippet.len() > width {
            return None;
        } else {
            let mut lines = snippet.lines();

            // The caller of this function has already placed `offset`
            // characters on the first line.
            let first_line_max_len = try_opt!(max_width.checked_sub(offset.width()));
            if lines.next().unwrap().len() > first_line_max_len {
                return None;
            }

            // The other lines must fit within the maximum width.
            if lines.find(|line| line.len() > max_width).is_some() {
                return None;
            }

            // `width` is the maximum length of the last line, excluding
            // indentation.
            // A special check for the last line, since the caller may
            // place trailing characters on this line.
            if snippet.lines().rev().next().unwrap().len() > offset.width() + width {
                return None;
            }
        }
    }

    Some(s)
}

impl Rewrite for String {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        wrap_str(self, context.config.max_width, width, offset).map(ToOwned::to_owned)
    }
}

// Binary search in integer range. Returns the first Ok value returned by the
// callback.
// The callback takes an integer and returns either an Ok, or an Err indicating
// whether the `guess' was too high (Ordering::Less), or too low.
// This function is guaranteed to try to the hi value first.
pub fn binary_search<C, T>(mut lo: usize, mut hi: usize, callback: C) -> Option<T>
    where C: Fn(usize) -> Result<T, Ordering>
{
    let mut middle = hi;

    while lo <= hi {
        match callback(middle) {
            Ok(val) => return Some(val),
            Err(Ordering::Less) => {
                hi = middle - 1;
            }
            Err(..) => {
                lo = middle + 1;
            }
        }
        middle = (hi + lo) / 2;
    }

    None
}

#[test]
fn bin_search_test() {
    let closure = |i| {
        match i {
            4 => Ok(()),
            j if j > 4 => Err(Ordering::Less),
            j if j < 4 => Err(Ordering::Greater),
            _ => unreachable!(),
        }
    };

    assert_eq!(Some(()), binary_search(1, 10, &closure));
    assert_eq!(None, binary_search(1, 3, &closure));
    assert_eq!(Some(()), binary_search(0, 44, &closure));
    assert_eq!(Some(()), binary_search(4, 125, &closure));
    assert_eq!(None, binary_search(6, 100, &closure));
}

#[test]
fn power_rounding() {
    assert_eq!(0, round_up_to_power_of_two(0));
    assert_eq!(1, round_up_to_power_of_two(1));
    assert_eq!(64, round_up_to_power_of_two(33));
    assert_eq!(256, round_up_to_power_of_two(256));
}
