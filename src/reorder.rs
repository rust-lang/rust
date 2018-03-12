// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Reorder items.
//!
//! `mod`, `extern crate` and `use` declarations are reorderd in alphabetical
//! order. Trait items are reordered in pre-determined order (associated types
//! and constatns comes before methods).

// TODO(#2455): Reorder trait items.

use config::{Config, lists::*};
use syntax::ast::UseTreeKind;
use syntax::{ast, attr, codemap::Span};

use attr::filter_inline_attrs;
use codemap::LineRangeUtils;
use comment::combine_strs_with_missing_comments;
use imports::{path_to_imported_ident, rewrite_import};
use items::{is_mod_decl, rewrite_extern_crate, rewrite_mod};
use lists::{itemize_list, write_list, ListFormatting};
use rewrite::{Rewrite, RewriteContext};
use shape::Shape;
use spanned::Spanned;
use utils::mk_sp;
use visitor::FmtVisitor;

use std::cmp::{Ord, Ordering, PartialOrd};

fn compare_use_trees(a: &ast::UseTree, b: &ast::UseTree) -> Ordering {
    let aa = UseTree::from_ast(a).normalize();
    let bb = UseTree::from_ast(b).normalize();
    aa.cmp(&bb)
}

/// Choose the ordering between the given two items.
fn compare_items(a: &ast::Item, b: &ast::Item) -> Ordering {
    match (&a.node, &b.node) {
        (&ast::ItemKind::Mod(..), &ast::ItemKind::Mod(..)) => {
            a.ident.name.as_str().cmp(&b.ident.name.as_str())
        }
        (&ast::ItemKind::Use(ref a_tree), &ast::ItemKind::Use(ref b_tree)) => {
            compare_use_trees(a_tree, b_tree)
        }
        (&ast::ItemKind::ExternCrate(ref a_name), &ast::ItemKind::ExternCrate(ref b_name)) => {
            // `extern crate foo as bar;`
            //               ^^^ Comparing this.
            let a_orig_name =
                a_name.map_or_else(|| a.ident.name.as_str(), |symbol| symbol.as_str());
            let b_orig_name =
                b_name.map_or_else(|| b.ident.name.as_str(), |symbol| symbol.as_str());
            let result = a_orig_name.cmp(&b_orig_name);
            if result != Ordering::Equal {
                return result;
            }

            // `extern crate foo as bar;`
            //                      ^^^ Comparing this.
            match (a_name, b_name) {
                (Some(..), None) => Ordering::Greater,
                (None, Some(..)) => Ordering::Less,
                (None, None) => Ordering::Equal,
                (Some(..), Some(..)) => a.ident.name.as_str().cmp(&b.ident.name.as_str()),
            }
        }
        _ => unreachable!(),
    }
}

/// Rewrite a list of items with reordering. Every item in `items` must have
/// the same `ast::ItemKind`.
fn rewrite_reorderable_items(
    context: &RewriteContext,
    reorderable_items: &[&ast::Item],
    shape: Shape,
    span: Span,
) -> Option<String> {
    let items = itemize_list(
        context.snippet_provider,
        reorderable_items.iter(),
        "",
        ";",
        |item| item.span().lo(),
        |item| item.span().hi(),
        |item| {
            let attrs = filter_inline_attrs(&item.attrs, item.span());
            let attrs_str = attrs.rewrite(context, shape)?;

            let missed_span = if attrs.is_empty() {
                mk_sp(item.span.lo(), item.span.lo())
            } else {
                mk_sp(attrs.last().unwrap().span.hi(), item.span.lo())
            };

            let item_str = match item.node {
                ast::ItemKind::Use(ref tree) => {
                    rewrite_import(context, &item.vis, tree, &item.attrs, shape)?
                }
                ast::ItemKind::ExternCrate(..) => rewrite_extern_crate(context, item)?,
                ast::ItemKind::Mod(..) => rewrite_mod(item),
                _ => return None,
            };

            combine_strs_with_missing_comments(
                context,
                &attrs_str,
                &item_str,
                missed_span,
                shape,
                false,
            )
        },
        span.lo(),
        span.hi(),
        false,
    );
    let mut item_pair_vec: Vec<_> = items.zip(reorderable_items.iter()).collect();
    item_pair_vec.sort_by(|a, b| compare_items(a.1, b.1));
    let item_vec: Vec<_> = item_pair_vec.into_iter().map(|pair| pair.0).collect();

    let fmt = ListFormatting {
        tactic: DefinitiveListTactic::Vertical,
        separator: "",
        trailing_separator: SeparatorTactic::Never,
        separator_place: SeparatorPlace::Back,
        shape,
        ends_with_newline: true,
        preserve_newline: false,
        config: context.config,
    };

    write_list(&item_vec, &fmt)
}

fn contains_macro_use_attr(item: &ast::Item) -> bool {
    attr::contains_name(&filter_inline_attrs(&item.attrs, item.span()), "macro_use")
}

/// A simplified version of `ast::ItemKind`.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum ReorderableItemKind {
    ExternCrate,
    Mod,
    Use,
    /// An item that cannot be reordered. Either has an unreorderable item kind
    /// or an `macro_use` attribute.
    Other,
}

impl ReorderableItemKind {
    pub fn from(item: &ast::Item) -> Self {
        match item.node {
            _ if contains_macro_use_attr(item) => ReorderableItemKind::Other,
            ast::ItemKind::ExternCrate(..) => ReorderableItemKind::ExternCrate,
            ast::ItemKind::Mod(..) if is_mod_decl(item) => ReorderableItemKind::Mod,
            ast::ItemKind::Use(..) => ReorderableItemKind::Use,
            _ => ReorderableItemKind::Other,
        }
    }

    pub fn is_same_item_kind(&self, item: &ast::Item) -> bool {
        ReorderableItemKind::from(item) == *self
    }

    pub fn is_reorderable(&self, config: &Config) -> bool {
        match *self {
            ReorderableItemKind::ExternCrate => config.reorder_extern_crates(),
            ReorderableItemKind::Mod => config.reorder_modules(),
            ReorderableItemKind::Use => config.reorder_imports(),
            ReorderableItemKind::Other => false,
        }
    }

    pub fn in_group(&self, config: &Config) -> bool {
        match *self {
            ReorderableItemKind::ExternCrate => config.reorder_extern_crates_in_group(),
            ReorderableItemKind::Mod => config.reorder_modules(),
            ReorderableItemKind::Use => config.reorder_imports_in_group(),
            ReorderableItemKind::Other => false,
        }
    }
}

impl<'b, 'a: 'b> FmtVisitor<'a> {
    /// Format items with the same item kind and reorder them. If `in_group` is
    /// `true`, then the items separated by an empty line will not be reordered
    /// together.
    fn walk_reorderable_items(
        &mut self,
        items: &[&ast::Item],
        item_kind: ReorderableItemKind,
        in_group: bool,
    ) -> usize {
        let mut last = self.codemap.lookup_line_range(items[0].span());
        let item_length = items
            .iter()
            .take_while(|ppi| {
                item_kind.is_same_item_kind(&***ppi) && (!in_group || {
                    let current = self.codemap.lookup_line_range(ppi.span());
                    let in_same_group = current.lo < last.hi + 2;
                    last = current;
                    in_same_group
                })
            })
            .count();
        let items = &items[..item_length];

        let at_least_one_in_file_lines = items
            .iter()
            .any(|item| !out_of_file_lines_range!(self, item.span));

        if at_least_one_in_file_lines && !items.is_empty() {
            let lo = items.first().unwrap().span().lo();
            let hi = items.last().unwrap().span().hi();
            let span = mk_sp(lo, hi);
            let rw = rewrite_reorderable_items(&self.get_context(), items, self.shape(), span);
            self.push_rewrite(span, rw);
        } else {
            for item in items {
                self.push_rewrite(item.span, None);
            }
        }

        item_length
    }

    /// Visit and format the given items. Items are reordered If they are
    /// consecutive and reorderable.
    pub fn visit_items_with_reordering(&mut self, mut items: &[&ast::Item]) {
        while !items.is_empty() {
            // If the next item is a `use`, `extern crate` or `mod`, then extract it and any
            // subsequent items that have the same item kind to be reordered within
            // `walk_reorderable_items`. Otherwise, just format the next item for output.
            let item_kind = ReorderableItemKind::from(items[0]);
            if item_kind.is_reorderable(self.config) {
                let visited_items_num =
                    self.walk_reorderable_items(items, item_kind, item_kind.in_group(self.config));
                let (_, rest) = items.split_at(visited_items_num);
                items = rest;
            } else {
                // Reaching here means items were not reordered. There must be at least
                // one item left in `items`, so calling `unwrap()` here is safe.
                let (item, rest) = items.split_first().unwrap();
                self.visit_item(item);
                items = rest;
            }
        }
    }
}

// Ordering of imports

// We order imports by translating to our own representation and then sorting.
// The Rust AST data structures are really bad for this. Rustfmt applies a bunch
// of normalisations to imports and since we want to sort based on the result
// of these (and to maintain idempotence) we must apply the same normalisations
// to the data structures for sorting.
//
// We sort `self` and `super` before other imports, then identifier imports,
// then glob imports, then lists of imports. We do not take aliases into account
// when ordering unless the imports are identical except for the alias (rare in
// practice).

// FIXME(#2531) - we should unify the comparison code here with the formatting
// code elsewhere since we are essentially string-ifying twice. Furthermore, by
// parsing to our own format on comparison, we repeat a lot of work when
// sorting.

// FIXME we do a lot of allocation to make our own representation.
#[derive(Debug, Clone, Eq, PartialEq)]
enum UseSegment {
    Ident(String, Option<String>),
    Slf(Option<String>),
    Super(Option<String>),
    Glob,
    List(Vec<UseTree>),
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct UseTree {
    path: Vec<UseSegment>,
}

impl UseSegment {
    // Clone a version of self with any top-level alias removed.
    fn remove_alias(&self) -> UseSegment {
        match *self {
            UseSegment::Ident(ref s, _) => UseSegment::Ident(s.clone(), None),
            UseSegment::Slf(_) => UseSegment::Slf(None),
            UseSegment::Super(_) => UseSegment::Super(None),
            _ => self.clone(),
        }
    }
}

impl UseTree {
    fn from_ast(a: &ast::UseTree) -> UseTree {
        let mut result = UseTree { path: vec![] };
        for p in &a.prefix.segments {
            result.path.push(UseSegment::Ident(
                (*p.identifier.name.as_str()).to_owned(),
                None,
            ));
        }
        match a.kind {
            UseTreeKind::Glob => {
                result.path.push(UseSegment::Glob);
            }
            UseTreeKind::Nested(ref list) => {
                result.path.push(UseSegment::List(
                    list.iter().map(|t| Self::from_ast(&t.0)).collect(),
                ));
            }
            UseTreeKind::Simple(ref rename) => {
                let mut name = (*path_to_imported_ident(&a.prefix).name.as_str()).to_owned();
                let alias = if &name == &*rename.name.as_str() {
                    None
                } else {
                    Some((&*rename.name.as_str()).to_owned())
                };

                let segment = if &name == "self" {
                    UseSegment::Slf(alias)
                } else if &name == "super" {
                    UseSegment::Super(alias)
                } else {
                    UseSegment::Ident(name, alias)
                };

                // `name` is already in result.
                result.path.pop();
                result.path.push(segment);
            }
        }
        result
    }

    // Do the adjustments that rustfmt does elsewhere to use paths.
    fn normalize(mut self) -> UseTree {
        let mut last = self.path.pop().expect("Empty use tree?");
        // Hack around borrow checker.
        let mut normalize_sole_list = false;
        let mut aliased_self = false;

        // Normalise foo::self -> foo.
        if let UseSegment::Slf(None) = last {
            return self;
        }

        // Normalise foo::self as bar -> foo as bar.
        if let UseSegment::Slf(_) = last {
            match self.path.last() {
                None => {}
                Some(UseSegment::Ident(_, None)) => {
                    aliased_self = true;
                }
                _ => unreachable!(),
            }
        }

        if aliased_self {
            match self.path.last() {
                Some(UseSegment::Ident(_, ref mut old_rename)) => {
                    assert!(old_rename.is_none());
                    if let UseSegment::Slf(Some(rename)) = last {
                        *old_rename = Some(rename);
                        return self;
                    }
                }
                _ => unreachable!(),
            }
        }

        // Normalise foo::{bar} -> foo::bar
        if let UseSegment::List(ref list) = last {
            if list.len() == 1 && list[0].path.len() == 1 {
                normalize_sole_list = true;
            }
        }

        if normalize_sole_list {
            match last {
                UseSegment::List(list) => {
                    self.path.push(list[0].path[0].clone());
                    return self.normalize();
                }
                _ => unreachable!(),
            }
        }

        // Recursively normalize elements of a list use (including sorting the list).
        if let UseSegment::List(list) = last {
            let mut list: Vec<_> = list.into_iter().map(|ut| ut.normalize()).collect();
            list.sort();
            last = UseSegment::List(list);
        }

        self.path.push(last);
        self
    }
}

impl PartialOrd for UseSegment {
    fn partial_cmp(&self, other: &UseSegment) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialOrd for UseTree {
    fn partial_cmp(&self, other: &UseTree) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for UseSegment {
    fn cmp(&self, other: &UseSegment) -> Ordering {
        use self::UseSegment::*;

        match (self, other) {
            (&Slf(ref a), &Slf(ref b)) | (&Super(ref a), &Super(ref b)) => a.cmp(b),
            (&Glob, &Glob) => Ordering::Equal,
            (&Ident(ref ia, ref aa), &Ident(ref ib, ref ab)) => {
                let ident_ord = ia.cmp(ib);
                if ident_ord != Ordering::Equal {
                    return ident_ord;
                }
                if aa.is_none() && ab.is_some() {
                    return Ordering::Less;
                }
                if aa.is_some() && ab.is_none() {
                    return Ordering::Greater;
                }
                aa.cmp(ab)
            }
            (&List(ref a), &List(ref b)) => {
                for (a, b) in a.iter().zip(b.iter()) {
                    let ord = a.cmp(b);
                    if ord != Ordering::Equal {
                        return ord;
                    }
                }

                a.len().cmp(&b.len())
            }
            (&Slf(_), _) => Ordering::Less,
            (_, &Slf(_)) => Ordering::Greater,
            (&Super(_), _) => Ordering::Less,
            (_, &Super(_)) => Ordering::Greater,
            (&Ident(..), _) => Ordering::Less,
            (_, &Ident(..)) => Ordering::Greater,
            (&Glob, _) => Ordering::Less,
            (_, &Glob) => Ordering::Greater,
        }
    }
}
impl Ord for UseTree {
    fn cmp(&self, other: &UseTree) -> Ordering {
        for (a, b) in self.path.iter().zip(other.path.iter()) {
            let ord = a.cmp(b);
            // The comparison without aliases is a hack to avoid situations like
            // comparing `a::b` to `a as c` - where the latter should be ordered
            // first since it is shorter.
            if ord != Ordering::Equal && a.remove_alias().cmp(&b.remove_alias()) != Ordering::Equal
            {
                return ord;
            }
        }

        self.path.len().cmp(&other.path.len())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // Parse the path part of an import. This parser is not robust and is only
    // suitable for use in a test harness.
    fn parse_use_tree(s: &str) -> UseTree {
        use std::iter::Peekable;
        use std::mem::swap;
        use std::str::Chars;

        struct Parser<'a> {
            input: Peekable<Chars<'a>>,
        }

        impl<'a> Parser<'a> {
            fn bump(&mut self) {
                self.input.next().unwrap();
            }
            fn eat(&mut self, c: char) {
                assert!(self.input.next().unwrap() == c);
            }
            fn push_segment(
                result: &mut Vec<UseSegment>,
                buf: &mut String,
                alias_buf: &mut Option<String>,
            ) {
                if !buf.is_empty() {
                    let mut alias = None;
                    swap(alias_buf, &mut alias);
                    if buf == "self" {
                        result.push(UseSegment::Slf(alias));
                        *buf = String::new();
                        *alias_buf = None;
                    } else if buf == "super" {
                        result.push(UseSegment::Super(alias));
                        *buf = String::new();
                        *alias_buf = None;
                    } else {
                        let mut name = String::new();
                        swap(buf, &mut name);
                        result.push(UseSegment::Ident(name, alias));
                    }
                }
            }
            fn parse_in_list(&mut self) -> UseTree {
                let mut result = vec![];
                let mut buf = String::new();
                let mut alias_buf = None;
                while let Some(&c) = self.input.peek() {
                    match c {
                        '{' => {
                            assert!(buf.is_empty());
                            self.bump();
                            result.push(UseSegment::List(self.parse_list()));
                            self.eat('}');
                        }
                        '*' => {
                            assert!(buf.is_empty());
                            self.bump();
                            result.push(UseSegment::Glob);
                        }
                        ':' => {
                            self.bump();
                            self.eat(':');
                            Self::push_segment(&mut result, &mut buf, &mut alias_buf);
                        }
                        '}' | ',' => {
                            Self::push_segment(&mut result, &mut buf, &mut alias_buf);
                            return UseTree { path: result };
                        }
                        ' ' => {
                            self.bump();
                            self.eat('a');
                            self.eat('s');
                            self.eat(' ');
                            alias_buf = Some(String::new());
                        }
                        c => {
                            self.bump();
                            if let Some(ref mut buf) = alias_buf {
                                buf.push(c);
                            } else {
                                buf.push(c);
                            }
                        }
                    }
                }
                Self::push_segment(&mut result, &mut buf, &mut alias_buf);
                UseTree { path: result }
            }

            fn parse_list(&mut self) -> Vec<UseTree> {
                let mut result = vec![];
                loop {
                    match self.input.peek().unwrap() {
                        ',' | ' ' => self.bump(),
                        '}' => {
                            return result;
                        }
                        _ => result.push(self.parse_in_list()),
                    }
                }
            }
        }

        let mut parser = Parser {
            input: s.chars().peekable(),
        };
        parser.parse_in_list()
    }

    #[test]
    fn test_use_tree_normalize() {
        assert_eq!(parse_use_tree("a::self").normalize(), parse_use_tree("a"));
        assert_eq!(
            parse_use_tree("a::self as foo").normalize(),
            parse_use_tree("a as foo")
        );
        assert_eq!(parse_use_tree("a::{self}").normalize(), parse_use_tree("a"));
        assert_eq!(parse_use_tree("a::{b}").normalize(), parse_use_tree("a::b"));
        assert_eq!(
            parse_use_tree("a::{b, c::self}").normalize(),
            parse_use_tree("a::{b, c}")
        );
        assert_eq!(
            parse_use_tree("a::{b as bar, c::self}").normalize(),
            parse_use_tree("a::{b as bar, c}")
        );
    }

    #[test]
    fn test_use_tree_ord() {
        assert!(parse_use_tree("a").normalize() < parse_use_tree("aa").normalize());
        assert!(parse_use_tree("a").normalize() < parse_use_tree("a::a").normalize());
        assert!(parse_use_tree("a").normalize() < parse_use_tree("*").normalize());
        assert!(parse_use_tree("a").normalize() < parse_use_tree("{a, b}").normalize());
        assert!(parse_use_tree("*").normalize() < parse_use_tree("{a, b}").normalize());

        assert!(
            parse_use_tree("aaaaaaaaaaaaaaa::{bb, cc, dddddddd}").normalize()
                < parse_use_tree("aaaaaaaaaaaaaaa::{bb, cc, ddddddddd}").normalize()
        );
        assert!(
            parse_use_tree("serde::de::{Deserialize}").normalize()
                < parse_use_tree("serde_json").normalize()
        );
        assert!(parse_use_tree("a::b::c").normalize() < parse_use_tree("a::b::*").normalize());
        assert!(
            parse_use_tree("foo::{Bar, Baz}").normalize()
                < parse_use_tree("{Bar, Baz}").normalize()
        );

        assert!(
            parse_use_tree("foo::{self as bar}").normalize()
                < parse_use_tree("foo::{qux as bar}").normalize()
        );
        assert!(
            parse_use_tree("foo::{qux as bar}").normalize()
                < parse_use_tree("foo::{baz, qux as bar}").normalize()
        );
        assert!(
            parse_use_tree("foo::{self as bar, baz}").normalize()
                < parse_use_tree("foo::{baz, qux as bar}").normalize()
        );

        assert!(parse_use_tree("Foo").normalize() < parse_use_tree("foo").normalize());
        assert!(parse_use_tree("foo").normalize() < parse_use_tree("foo::Bar").normalize());

        assert!(
            parse_use_tree("std::cmp::{d, c, b, a}").normalize()
                < parse_use_tree("std::cmp::{b, e, g, f}").normalize()
        );
    }
}
