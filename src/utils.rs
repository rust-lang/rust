// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::borrow::Cow;
use std::cmp::Ordering;

use itertools::Itertools;

use syntax::ast::{self, Visibility, Attribute, MetaItem, MetaItemKind, Path};
use syntax::codemap::BytePos;
use syntax::abi;

use Indent;
use rewrite::{Rewrite, RewriteContext};

use SKIP_ANNOTATION;

// Computes the length of a string's last line, minus offset.
pub fn extra_offset(text: &str, offset: Indent) -> usize {
    match text.rfind('\n') {
        // 1 for newline character
        Some(idx) => text.len().checked_sub(idx + 1 + offset.width()).unwrap_or(0),
        None => text.len(),
    }
}

// Uses Cow to avoid allocating in the common cases.
pub fn format_visibility(vis: &Visibility) -> Cow<'static, str> {
    match *vis {
        Visibility::Public => Cow::from("pub "),
        Visibility::Inherited => Cow::from(""),
        Visibility::Crate(_) => Cow::from("pub(crate) "),
        Visibility::Restricted { ref path, .. } => {
            let Path { global, ref segments, .. } = **path;
            let prefix = if global { "::" } else { "" };
            let mut segments_iter = segments.iter().map(|seg| seg.identifier.name.as_str());

            Cow::from(format!("pub({}{}) ", prefix, segments_iter.join("::")))
        }
    }
}

#[inline]
pub fn format_unsafety(unsafety: ast::Unsafety) -> &'static str {
    match unsafety {
        ast::Unsafety::Unsafe => "unsafe ",
        ast::Unsafety::Normal => "",
    }
}

#[inline]
pub fn format_mutability(mutability: ast::Mutability) -> &'static str {
    match mutability {
        ast::Mutability::Mutable => "mut ",
        ast::Mutability::Immutable => "",
    }
}

#[inline]
pub fn format_abi(abi: abi::Abi, explicit_abi: bool) -> String {
    if abi == abi::Abi::C && !explicit_abi {
        "extern ".into()
    } else {
        format!("extern {} ", abi)
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
pub fn trimmed_last_line_width(s: &str) -> usize {
    match s.rfind('\n') {
        Some(n) => s[(n + 1)..].trim().len(),
        None => s.trim().len(),
    }
}

#[inline]
fn is_skip(meta_item: &MetaItem) -> bool {
    match meta_item.node {
        MetaItemKind::Word(ref s) => *s == SKIP_ANNOTATION,
        MetaItemKind::List(ref s, ref l) => *s == "cfg_attr" && l.len() == 2 && is_skip(&l[1]),
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
        .map_or(typaram.span, |bound| {
            match *bound {
                ast::RegionTyParamBound(ref lt) => lt.span,
                ast::TraitTyParamBound(ref prt, _) => prt.span,
            }
        })
        .hi
}

#[inline]
pub fn semicolon_for_expr(expr: &ast::Expr) -> bool {
    match expr.node {
        ast::ExprKind::Ret(..) |
        ast::ExprKind::Again(..) |
        ast::ExprKind::Break(..) => true,
        _ => false,
    }
}

#[inline]
pub fn semicolon_for_stmt(stmt: &ast::Stmt) -> bool {
    match stmt.node {
        ast::StmtKind::Semi(ref expr, _) => {
            match expr.node {
                ast::ExprKind::While(..) |
                ast::ExprKind::WhileLet(..) |
                ast::ExprKind::Loop(..) |
                ast::ExprKind::ForLoop(..) => false,
                _ => true,
            }
        }
        ast::StmtKind::Expr(..) => false,
        _ => true,
    }
}

#[inline]
pub fn trim_newlines(input: &str) -> &str {
    match input.find(|c| c != '\n' && c != '\r') {
        Some(start) => {
            let end = input.rfind(|c| c != '\n' && c != '\r').unwrap_or(0) + 1;
            &input[start..end]
        }
        None => "",
    }
}

// Macro for deriving implementations of Decodable for enums
#[macro_export]
macro_rules! impl_enum_decodable {
    ( $e:ident, $( $x:ident ),* ) => {
        impl ::rustc_serialize::Decodable for $e {
            fn decode<D: ::rustc_serialize::Decoder>(d: &mut D) -> Result<Self, D::Error> {
                use std::ascii::AsciiExt;
                let s = try!(d.read_str());
                $(
                    if stringify!($x).eq_ignore_ascii_case(&s) {
                      return Ok($e::$x);
                    }
                )*
                Err(d.error("Bad variant"))
            }
        }

        impl ::std::str::FromStr for $e {
            type Err = &'static str;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                use std::ascii::AsciiExt;
                $(
                    if stringify!($x).eq_ignore_ascii_case(s) {
                        return Ok($e::$x);
                    }
                )*
                Err("Bad variant")
            }
        }

        impl ::config::ConfigType for $e {
            fn doc_hint() -> String {
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

macro_rules! msg {
    ($($arg:tt)*) => (
        match writeln!(&mut ::std::io::stderr(), $($arg)* ) {
            Ok(_) => {},
            Err(x) => panic!("Unable to write to stderr: {}", x),
        }
    )
}

// For format_missing and last_pos, need to use the source callsite (if applicable).
// Required as generated code spans aren't guaranteed to follow on from the last span.
macro_rules! source {
    ($this:ident, $sp: expr) => {
        $this.codemap.source_callsite($sp)
    }
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
            if lines.any(|line| line.len() > max_width) {
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

pub fn left_most_sub_expr(e: &ast::Expr) -> &ast::Expr {
    match e.node {
        ast::ExprKind::InPlace(ref e, _) |
        ast::ExprKind::Call(ref e, _) |
        ast::ExprKind::Binary(_, ref e, _) |
        ast::ExprKind::Cast(ref e, _) |
        ast::ExprKind::Type(ref e, _) |
        ast::ExprKind::Assign(ref e, _) |
        ast::ExprKind::AssignOp(_, ref e, _) |
        ast::ExprKind::Field(ref e, _) |
        ast::ExprKind::TupField(ref e, _) |
        ast::ExprKind::Index(ref e, _) |
        ast::ExprKind::Range(Some(ref e), _, _) => left_most_sub_expr(e),
        // FIXME needs Try in Syntex
        // ast::ExprKind::Try(ref f) => left_most_sub_expr(e),
        _ => e,
    }
}
