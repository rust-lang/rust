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

use syntax::ast::{self, Visibility, Attribute, MetaItem, MetaItemKind, NestedMetaItem,
                  NestedMetaItemKind, Path};
use syntax::codemap::{BytePos, Span, NO_EXPANSION};
use syntax::abi;

use Shape;
use rewrite::{Rewrite, RewriteContext};

use SKIP_ANNOTATION;

// Computes the length of a string's last line, minus offset.
pub fn extra_offset(text: &str, shape: Shape) -> usize {
    match text.rfind('\n') {
        // 1 for newline character
        Some(idx) => {
            text.len()
                .checked_sub(idx + 1 + shape.used_width())
                .unwrap_or(0)
        }
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
            let Path { ref segments, .. } = **path;
            let mut segments_iter = segments.iter().map(|seg| seg.identifier.name.to_string());
            if path.is_global() {
                segments_iter.next().expect(
                    "Non-global path in pub(restricted)?",
                );
            }
            let is_keyword = |s: &str| s == "self" || s == "super";
            let path = segments_iter.collect::<Vec<_>>().join("::");
            let in_str = if is_keyword(&path) { "" } else { "in " };

            Cow::from(format!("pub({}{}) ", in_str, path))
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
        MetaItemKind::Word => meta_item.name == SKIP_ANNOTATION,
        MetaItemKind::List(ref l) => {
            meta_item.name == "cfg_attr" && l.len() == 2 && is_skip_nested(&l[1])
        }
        _ => false,
    }
}

#[inline]
fn is_skip_nested(meta_item: &NestedMetaItem) -> bool {
    match meta_item.node {
        NestedMetaItemKind::MetaItem(ref mi) => is_skip(mi),
        NestedMetaItemKind::Literal(_) => false,
    }
}

#[inline]
pub fn contains_skip(attrs: &[Attribute]) -> bool {
    attrs.iter().any(
        |a| a.meta().map_or(false, |a| is_skip(&a)),
    )
}

// Find the end of a TyParam
#[inline]
pub fn end_typaram(typaram: &ast::TyParam) -> BytePos {
    typaram
        .bounds
        .last()
        .map_or(typaram.span, |bound| match *bound {
            ast::RegionTyParamBound(ref lt) => lt.span,
            ast::TraitTyParamBound(ref prt, _) => prt.span,
        })
        .hi
}

#[inline]
pub fn semicolon_for_expr(expr: &ast::Expr) -> bool {
    match expr.node {
        ast::ExprKind::Ret(..) |
        ast::ExprKind::Continue(..) |
        ast::ExprKind::Break(..) => true,
        _ => false,
    }
}

#[inline]
pub fn semicolon_for_stmt(stmt: &ast::Stmt) -> bool {
    match stmt.node {
        ast::StmtKind::Semi(ref expr) => {
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
pub fn stmt_expr(stmt: &ast::Stmt) -> Option<&ast::Expr> {
    match stmt.node {
        ast::StmtKind::Expr(ref expr) => Some(expr),
        _ => None,
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

// Macro for deriving implementations of Serialize/Deserialize for enums
#[macro_export]
macro_rules! impl_enum_serialize_and_deserialize {
    ( $e:ident, $( $x:ident ),* ) => {
        impl ::serde::ser::Serialize for $e {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where S: ::serde::ser::Serializer
            {
                use serde::ser::Error;

                // We don't know whether the user of the macro has given us all options.
                #[allow(unreachable_patterns)]
                match *self {
                    $(
                        $e::$x => serializer.serialize_str(stringify!($x)),
                    )*
                    _ => {
                        Err(S::Error::custom(format!("Cannot serialize {:?}", self)))
                    }
                }
            }
        }

        impl<'de> ::serde::de::Deserialize<'de> for $e {
            fn deserialize<D>(d: D) -> Result<Self, D::Error>
                    where D: ::serde::Deserializer<'de> {
                use std::ascii::AsciiExt;
                use serde::de::{Error, Visitor};
                use std::marker::PhantomData;
                use std::fmt;
                struct StringOnly<T>(PhantomData<T>);
                impl<'de, T> Visitor<'de> for StringOnly<T>
                        where T: ::serde::Deserializer<'de> {
                    type Value = String;
                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("string")
                    }
                    fn visit_str<E>(self, value: &str) -> Result<String, E> {
                        Ok(String::from(value))
                    }
                }
                let s = d.deserialize_string(StringOnly::<D>(PhantomData))?;
                $(
                    if stringify!($x).eq_ignore_ascii_case(&s) {
                      return Ok($e::$x);
                    }
                )*
                static ALLOWED: &'static[&str] = &[$(stringify!($x),)*];
                Err(D::Error::unknown_variant(&s, ALLOWED))
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
        $sp.source_callsite()
    }
}

pub fn mk_sp(lo: BytePos, hi: BytePos) -> Span {
    Span {
        lo,
        hi,
        ctxt: NO_EXPANSION,
    }
}

// Wraps string-like values in an Option. Returns Some when the string adheres
// to the Rewrite constraints defined for the Rewrite trait and else otherwise.
pub fn wrap_str<S: AsRef<str>>(s: S, max_width: usize, shape: Shape) -> Option<S> {
    {
        let snippet = s.as_ref();

        if !snippet.is_empty() {
            if !snippet.contains('\n') && snippet.len() > shape.width {
                return None;
            } else {
                let mut lines = snippet.lines();

                // The caller of this function has already placed `shape.offset`
                // characters on the first line.
                let first_line_max_len = try_opt!(max_width.checked_sub(shape.indent.width()));
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
                if snippet.lines().rev().next().unwrap().len() >
                    shape.indent.width() + shape.width
                {
                    return None;
                }
            }
        }
    }

    Some(s)
}

impl Rewrite for String {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        wrap_str(self, context.config.max_width(), shape).map(ToOwned::to_owned)
    }
}

// Binary search in integer range. Returns the first Ok value returned by the
// callback.
// The callback takes an integer and returns either an Ok, or an Err indicating
// whether the `guess' was too high (Ordering::Less), or too low.
// This function is guaranteed to try to the hi value first.
pub fn binary_search<C, T>(mut lo: usize, mut hi: usize, callback: C) -> Option<T>
where
    C: Fn(usize) -> Result<T, Ordering>,
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

#[inline]
pub fn colon_spaces(before: bool, after: bool) -> &'static str {
    match (before, after) {
        (true, true) => " : ",
        (true, false) => " :",
        (false, true) => ": ",
        (false, false) => ":",
    }
}

#[test]
fn bin_search_test() {
    let closure = |i| match i {
        4 => Ok(()),
        j if j > 4 => Err(Ordering::Less),
        j if j < 4 => Err(Ordering::Greater),
        _ => unreachable!(),
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
        ast::ExprKind::Range(Some(ref e), _, _) |
        ast::ExprKind::Try(ref e) => left_most_sub_expr(e),
        _ => e,
    }
}
