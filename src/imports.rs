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

use config::lists::*;
use syntax::ast::{self, UseTreeKind};
use syntax::source_map::{self, BytePos, Span, DUMMY_SP};

use comment::combine_strs_with_missing_comments;
use config::IndentStyle;
use lists::{definitive_tactic, itemize_list, write_list, ListFormatting, ListItem, Separator};
use rewrite::{Rewrite, RewriteContext};
use shape::Shape;
use source_map::SpanUtils;
use spanned::Spanned;
use utils::{is_same_visibility, mk_sp, rewrite_ident};
use visitor::FmtVisitor;

use std::borrow::Cow;
use std::fmt;

/// Returns a name imported by a `use` declaration. e.g. returns `Ordering`
/// for `std::cmp::Ordering` and `self` for `std::cmp::self`.
pub fn path_to_imported_ident(path: &ast::Path) -> ast::Ident {
    path.segments.last().unwrap().ident
}

impl<'a> FmtVisitor<'a> {
    pub fn format_import(&mut self, item: &ast::Item, tree: &ast::UseTree) {
        let span = item.span();
        let shape = self.shape();
        let rw = UseTree::from_ast(
            &self.get_context(),
            tree,
            None,
            Some(item.vis.clone()),
            Some(item.span.lo()),
            Some(item.attrs.clone()),
        ).rewrite_top_level(&self.get_context(), shape);
        match rw {
            Some(ref s) if s.is_empty() => {
                // Format up to last newline
                let prev_span = mk_sp(self.last_pos, source!(self, span).lo());
                let trimmed_snippet = self.snippet(prev_span).trim_right();
                let span_end = self.last_pos + BytePos(trimmed_snippet.len() as u32);
                self.format_missing(span_end);
                // We have an excessive newline from the removed import.
                if self.buffer.ends_with('\n') {
                    self.buffer.pop();
                    self.line_number -= 1;
                }
                self.last_pos = source!(self, span).hi();
            }
            Some(ref s) => {
                self.format_missing_with_indent(source!(self, span).lo());
                self.push_str(s);
                self.last_pos = source!(self, span).hi();
            }
            None => {
                self.format_missing_with_indent(source!(self, span).lo());
                self.format_missing(source!(self, span).hi());
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
#[derive(Clone, Eq, PartialEq)]
pub enum UseSegment {
    Ident(String, Option<String>),
    Slf(Option<String>),
    Super(Option<String>),
    Glob,
    List(Vec<UseTree>),
}

#[derive(Clone)]
pub struct UseTree {
    pub path: Vec<UseSegment>,
    pub span: Span,
    // Comment information within nested use tree.
    pub list_item: Option<ListItem>,
    // Additional fields for top level use items.
    // Should we have another struct for top-level use items rather than reusing this?
    visibility: Option<ast::Visibility>,
    attrs: Option<Vec<ast::Attribute>>,
}

impl PartialEq for UseTree {
    fn eq(&self, other: &UseTree) -> bool {
        self.path == other.path
    }
}
impl Eq for UseTree {}

impl Spanned for UseTree {
    fn span(&self) -> Span {
        let lo = if let Some(ref attrs) = self.attrs {
            attrs.iter().next().map_or(self.span.lo(), |a| a.span.lo())
        } else {
            self.span.lo()
        };
        mk_sp(lo, self.span.hi())
    }
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

    fn from_path_segment(
        context: &RewriteContext,
        path_seg: &ast::PathSegment,
    ) -> Option<UseSegment> {
        let name = rewrite_ident(context, path_seg.ident);
        if name.is_empty() || name == "{{root}}" {
            return None;
        }
        Some(match name {
            "self" => UseSegment::Slf(None),
            "super" => UseSegment::Super(None),
            _ => UseSegment::Ident((*name).to_owned(), None),
        })
    }
}

pub fn merge_use_trees(use_trees: Vec<UseTree>) -> Vec<UseTree> {
    let mut result = Vec::with_capacity(use_trees.len());
    for use_tree in use_trees {
        if use_tree.has_comment() || use_tree.attrs.is_some() {
            result.push(use_tree);
            continue;
        }

        for flattened in use_tree.flatten() {
            merge_use_trees_inner(&mut result, flattened);
        }
    }
    result
}

fn merge_use_trees_inner(trees: &mut Vec<UseTree>, use_tree: UseTree) {
    for tree in trees.iter_mut() {
        if tree.share_prefix(&use_tree) {
            tree.merge(use_tree);
            return;
        }
    }

    trees.push(use_tree);
}

impl fmt::Debug for UseTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Debug for UseSegment {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for UseSegment {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            UseSegment::Glob => write!(f, "*"),
            UseSegment::Ident(ref s, _) => write!(f, "{}", s),
            UseSegment::Slf(..) => write!(f, "self"),
            UseSegment::Super(..) => write!(f, "super"),
            UseSegment::List(ref list) => {
                write!(f, "{{")?;
                for (i, item) in list.iter().enumerate() {
                    let is_last = i == list.len() - 1;
                    write!(f, "{}", item)?;
                    if !is_last {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "}}")
            }
        }
    }
}
impl fmt::Display for UseTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, segment) in self.path.iter().enumerate() {
            let is_last = i == self.path.len() - 1;
            write!(f, "{}", segment)?;
            if !is_last {
                write!(f, "::")?;
            }
        }
        write!(f, "")
    }
}

impl UseTree {
    // Rewrite use tree with `use ` and a trailing `;`.
    pub fn rewrite_top_level(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        let vis = self.visibility.as_ref().map_or(Cow::from(""), |vis| {
            ::utils::format_visibility(context, &vis)
        });
        let use_str = self
            .rewrite(context, shape.offset_left(vis.len())?)
            .map(|s| {
                if s.is_empty() {
                    s.to_owned()
                } else {
                    format!("{}use {};", vis, s)
                }
            })?;
        if let Some(ref attrs) = self.attrs {
            let attr_str = attrs.rewrite(context, shape)?;
            let lo = attrs.last().as_ref()?.span().hi();
            let hi = self.span.lo();
            let span = mk_sp(lo, hi);
            combine_strs_with_missing_comments(context, &attr_str, &use_str, span, shape, false)
        } else {
            Some(use_str)
        }
    }

    // FIXME: Use correct span?
    // The given span is essentially incorrect, since we are reconstructing
    // use statements. This should not be a problem, though, since we have
    // already tried to extract comment and observed that there are no comment
    // around the given use item, and the span will not be used afterward.
    fn from_path(path: Vec<UseSegment>, span: Span) -> UseTree {
        UseTree {
            path,
            span,
            list_item: None,
            visibility: None,
            attrs: None,
        }
    }

    pub fn from_ast_with_normalization(
        context: &RewriteContext,
        item: &ast::Item,
    ) -> Option<UseTree> {
        match item.node {
            ast::ItemKind::Use(ref use_tree) => Some(
                UseTree::from_ast(
                    context,
                    use_tree,
                    None,
                    Some(item.vis.clone()),
                    Some(item.span.lo()),
                    if item.attrs.is_empty() {
                        None
                    } else {
                        Some(item.attrs.clone())
                    },
                ).normalize(),
            ),
            _ => None,
        }
    }

    fn from_ast(
        context: &RewriteContext,
        a: &ast::UseTree,
        list_item: Option<ListItem>,
        visibility: Option<ast::Visibility>,
        opt_lo: Option<BytePos>,
        attrs: Option<Vec<ast::Attribute>>,
    ) -> UseTree {
        let span = if let Some(lo) = opt_lo {
            mk_sp(lo, a.span.hi())
        } else {
            a.span
        };
        let mut result = UseTree {
            path: vec![],
            span,
            list_item,
            visibility,
            attrs,
        };
        for p in &a.prefix.segments {
            if let Some(use_segment) = UseSegment::from_path_segment(context, p) {
                result.path.push(use_segment);
            }
        }
        match a.kind {
            UseTreeKind::Glob => {
                result.path.push(UseSegment::Glob);
            }
            UseTreeKind::Nested(ref list) => {
                // Extract comments between nested use items.
                // This needs to be done before sorting use items.
                let items: Vec<_> = itemize_list(
                    context.snippet_provider,
                    list.iter().map(|(tree, _)| tree),
                    "}",
                    ",",
                    |tree| tree.span.lo(),
                    |tree| tree.span.hi(),
                    |_| Some("".to_owned()), // We only need comments for now.
                    context.snippet_provider.span_after(a.span, "{"),
                    a.span.hi(),
                    false,
                ).collect();
                result.path.push(UseSegment::List(
                    list.iter()
                        .zip(items.into_iter())
                        .map(|(t, list_item)| {
                            Self::from_ast(context, &t.0, Some(list_item), None, None, None)
                        }).collect(),
                ));
            }
            UseTreeKind::Simple(ref rename, ..) => {
                let name = rewrite_ident(context, path_to_imported_ident(&a.prefix)).to_owned();
                let alias = rename.and_then(|ident| {
                    if ident.name == "_" {
                        // for impl-only-use
                        Some("_".to_owned())
                    } else if ident == path_to_imported_ident(&a.prefix) {
                        None
                    } else {
                        Some(rewrite_ident(context, ident).to_owned())
                    }
                });
                let segment = match name.as_ref() {
                    "self" => UseSegment::Slf(alias),
                    "super" => UseSegment::Super(alias),
                    _ => UseSegment::Ident(name, alias),
                };

                // `name` is already in result.
                result.path.pop();
                result.path.push(segment);
            }
        }
        result
    }

    // Do the adjustments that rustfmt does elsewhere to use paths.
    pub fn normalize(mut self) -> UseTree {
        let mut last = self.path.pop().expect("Empty use tree?");
        // Hack around borrow checker.
        let mut normalize_sole_list = false;
        let mut aliased_self = false;

        // Remove foo::{} or self without attributes.
        match last {
            _ if self.attrs.is_some() => (),
            UseSegment::List(ref list) if list.is_empty() => {
                self.path = vec![];
                return self;
            }
            UseSegment::Slf(None) if self.path.is_empty() && self.visibility.is_some() => {
                self.path = vec![];
                return self;
            }
            _ => (),
        }

        // Normalise foo::self -> foo.
        if let UseSegment::Slf(None) = last {
            if !self.path.is_empty() {
                return self;
            }
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

        let mut done = false;
        if aliased_self {
            match self.path.last_mut() {
                Some(UseSegment::Ident(_, ref mut old_rename)) => {
                    assert!(old_rename.is_none());
                    if let UseSegment::Slf(Some(rename)) = last.clone() {
                        *old_rename = Some(rename);
                        done = true;
                    }
                }
                _ => unreachable!(),
            }
        }

        if done {
            return self;
        }

        // Normalise foo::{bar} -> foo::bar
        if let UseSegment::List(ref list) = last {
            if list.len() == 1 {
                normalize_sole_list = true;
            }
        }

        if normalize_sole_list {
            match last {
                UseSegment::List(list) => {
                    for seg in &list[0].path {
                        self.path.push(seg.clone());
                    }
                    return self.normalize();
                }
                _ => unreachable!(),
            }
        }

        // Recursively normalize elements of a list use (including sorting the list).
        if let UseSegment::List(list) = last {
            let mut list = list
                .into_iter()
                .map(|ut| ut.normalize())
                .collect::<Vec<_>>();
            list.sort();
            last = UseSegment::List(list);
        }

        self.path.push(last);
        self
    }

    fn has_comment(&self) -> bool {
        self.list_item.as_ref().map_or(false, ListItem::has_comment)
    }

    fn same_visibility(&self, other: &UseTree) -> bool {
        match (&self.visibility, &other.visibility) {
            (
                Some(source_map::Spanned {
                    node: ast::VisibilityKind::Inherited,
                    ..
                }),
                None,
            )
            | (
                None,
                Some(source_map::Spanned {
                    node: ast::VisibilityKind::Inherited,
                    ..
                }),
            )
            | (None, None) => true,
            (Some(ref a), Some(ref b)) => is_same_visibility(a, b),
            _ => false,
        }
    }

    fn share_prefix(&self, other: &UseTree) -> bool {
        if self.path.is_empty()
            || other.path.is_empty()
            || self.attrs.is_some()
            || !self.same_visibility(other)
        {
            false
        } else {
            self.path[0] == other.path[0]
        }
    }

    fn flatten(self) -> Vec<UseTree> {
        if self.path.is_empty() {
            return vec![self];
        }
        match self.path.clone().last().unwrap() {
            UseSegment::List(list) => {
                let prefix = &self.path[..self.path.len() - 1];
                let mut result = vec![];
                for nested_use_tree in list {
                    for flattend in &mut nested_use_tree.clone().flatten() {
                        let mut new_path = prefix.to_vec();
                        new_path.append(&mut flattend.path);
                        result.push(UseTree {
                            path: new_path,
                            span: self.span,
                            list_item: None,
                            visibility: self.visibility.clone(),
                            attrs: None,
                        });
                    }
                }

                result
            }
            _ => vec![self],
        }
    }

    fn merge(&mut self, other: UseTree) {
        let mut new_path = vec![];
        for (a, b) in self
            .path
            .clone()
            .iter_mut()
            .zip(other.path.clone().into_iter())
        {
            if *a == b {
                new_path.push(b);
            } else {
                break;
            }
        }
        if let Some(merged) = merge_rest(&self.path, &other.path, new_path.len()) {
            new_path.push(merged);
            self.span = self.span.to(other.span);
        }
        self.path = new_path;
    }
}

fn merge_rest(a: &[UseSegment], b: &[UseSegment], len: usize) -> Option<UseSegment> {
    let a_rest = &a[len..];
    let b_rest = &b[len..];
    if a_rest.is_empty() && b_rest.is_empty() {
        return None;
    }
    if a_rest.is_empty() {
        return Some(UseSegment::List(vec![
            UseTree::from_path(vec![UseSegment::Slf(None)], DUMMY_SP),
            UseTree::from_path(b_rest.to_vec(), DUMMY_SP),
        ]));
    }
    if b_rest.is_empty() {
        return Some(UseSegment::List(vec![
            UseTree::from_path(vec![UseSegment::Slf(None)], DUMMY_SP),
            UseTree::from_path(a_rest.to_vec(), DUMMY_SP),
        ]));
    }
    if let UseSegment::List(mut list) = a_rest[0].clone() {
        merge_use_trees_inner(&mut list, UseTree::from_path(b_rest.to_vec(), DUMMY_SP));
        list.sort();
        return Some(UseSegment::List(list.clone()));
    }
    let mut list = vec![
        UseTree::from_path(a_rest.to_vec(), DUMMY_SP),
        UseTree::from_path(b_rest.to_vec(), DUMMY_SP),
    ];
    list.sort();
    Some(UseSegment::List(list))
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

        fn is_upper_snake_case(s: &str) -> bool {
            s.chars()
                .all(|c| c.is_uppercase() || c == '_' || c.is_numeric())
        }

        match (self, other) {
            (&Slf(ref a), &Slf(ref b)) | (&Super(ref a), &Super(ref b)) => a.cmp(b),
            (&Glob, &Glob) => Ordering::Equal,
            (&Ident(ref ia, ref aa), &Ident(ref ib, ref ab)) => {
                // snake_case < CamelCase < UPPER_SNAKE_CASE
                if ia.starts_with(char::is_uppercase) && ib.starts_with(char::is_lowercase) {
                    return Ordering::Greater;
                }
                if ia.starts_with(char::is_lowercase) && ib.starts_with(char::is_uppercase) {
                    return Ordering::Less;
                }
                if is_upper_snake_case(ia) && !is_upper_snake_case(ib) {
                    return Ordering::Greater;
                }
                if !is_upper_snake_case(ia) && is_upper_snake_case(ib) {
                    return Ordering::Less;
                }
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

fn rewrite_nested_use_tree(
    context: &RewriteContext,
    use_tree_list: &[UseTree],
    shape: Shape,
) -> Option<String> {
    let mut list_items = Vec::with_capacity(use_tree_list.len());
    let nested_shape = match context.config.imports_indent() {
        IndentStyle::Block => shape
            .block_indent(context.config.tab_spaces())
            .with_max_width(context.config)
            .sub_width(1)?,
        IndentStyle::Visual => shape.visual_indent(0),
    };
    for use_tree in use_tree_list {
        if let Some(mut list_item) = use_tree.list_item.clone() {
            list_item.item = use_tree.rewrite(context, nested_shape);
            list_items.push(list_item);
        } else {
            list_items.push(ListItem::from_str(use_tree.rewrite(context, nested_shape)?));
        }
    }
    let has_nested_list = use_tree_list.iter().any(|use_segment| {
        use_segment
            .path
            .last()
            .map_or(false, |last_segment| match last_segment {
                UseSegment::List(..) => true,
                _ => false,
            })
    });

    let remaining_width = if has_nested_list {
        0
    } else {
        shape.width.saturating_sub(2)
    };

    let tactic = definitive_tactic(
        &list_items,
        context.config.imports_layout(),
        Separator::Comma,
        remaining_width,
    );

    let ends_with_newline = context.config.imports_indent() == IndentStyle::Block
        && tactic != DefinitiveListTactic::Horizontal;
    let trailing_separator = if ends_with_newline {
        context.config.trailing_comma()
    } else {
        SeparatorTactic::Never
    };
    let fmt = ListFormatting::new(nested_shape, context.config)
        .tactic(tactic)
        .trailing_separator(trailing_separator)
        .ends_with_newline(ends_with_newline)
        .preserve_newline(true)
        .nested(has_nested_list);

    let list_str = write_list(&list_items, &fmt)?;

    let result = if (list_str.contains('\n') || list_str.len() > remaining_width)
        && context.config.imports_indent() == IndentStyle::Block
    {
        format!(
            "{{\n{}{}\n{}}}",
            nested_shape.indent.to_string(context.config),
            list_str,
            shape.indent.to_string(context.config)
        )
    } else {
        format!("{{{}}}", list_str)
    };

    Some(result)
}

impl Rewrite for UseSegment {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        Some(match self {
            UseSegment::Ident(ref ident, Some(ref rename)) => format!("{} as {}", ident, rename),
            UseSegment::Ident(ref ident, None) => ident.clone(),
            UseSegment::Slf(Some(ref rename)) => format!("self as {}", rename),
            UseSegment::Slf(None) => "self".to_owned(),
            UseSegment::Super(Some(ref rename)) => format!("super as {}", rename),
            UseSegment::Super(None) => "super".to_owned(),
            UseSegment::Glob => "*".to_owned(),
            UseSegment::List(ref use_tree_list) => rewrite_nested_use_tree(
                context,
                use_tree_list,
                // 1 = "{" and "}"
                shape.offset_left(1)?.sub_width(1)?,
            )?,
        })
    }
}

impl Rewrite for UseTree {
    // This does NOT format attributes and visibility or add a trailing `;`.
    fn rewrite(&self, context: &RewriteContext, mut shape: Shape) -> Option<String> {
        let mut result = String::with_capacity(256);
        let mut iter = self.path.iter().peekable();
        while let Some(ref segment) = iter.next() {
            let segment_str = segment.rewrite(context, shape)?;
            result.push_str(&segment_str);
            if iter.peek().is_some() {
                result.push_str("::");
                // 2 = "::"
                shape = shape.offset_left(2 + segment_str.len())?;
            }
        }
        Some(result)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use syntax::source_map::DUMMY_SP;

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
                            return UseTree {
                                path: result,
                                span: DUMMY_SP,
                                list_item: None,
                                visibility: None,
                                attrs: None,
                            };
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
                UseTree {
                    path: result,
                    span: DUMMY_SP,
                    list_item: None,
                    visibility: None,
                    attrs: None,
                }
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

    macro parse_use_trees($($s:expr),* $(,)*) {
        vec![
            $(parse_use_tree($s),)*
        ]
    }

    #[test]
    fn test_use_tree_merge() {
        macro test_merge([$($input:expr),* $(,)*], [$($output:expr),* $(,)*]) {
            assert_eq!(
                merge_use_trees(parse_use_trees!($($input,)*)),
                parse_use_trees!($($output,)*),
            );
        }

        test_merge!(["a::b::{c, d}", "a::b::{e, f}"], ["a::b::{c, d, e, f}"]);
        test_merge!(["a::b::c", "a::b"], ["a::b::{self, c}"]);
        test_merge!(["a::b", "a::b"], ["a::b"]);
        test_merge!(["a", "a::b", "a::b::c"], ["a::{self, b::{self, c}}"]);
        test_merge!(
            ["a::{b::{self, c}, d::e}", "a::d::f"],
            ["a::{b::{self, c}, d::{e, f}}"]
        );
        test_merge!(
            ["a::d::f", "a::{b::{self, c}, d::e}"],
            ["a::{b::{self, c}, d::{e, f}}"]
        );
        test_merge!(
            ["a::{c, d, b}", "a::{d, e, b, a, f}", "a::{f, g, c}"],
            ["a::{a, b, c, d, e, f, g}"]
        );
    }

    #[test]
    fn test_use_tree_flatten() {
        assert_eq!(
            parse_use_tree("a::b::{c, d, e, f}").flatten(),
            parse_use_trees!("a::b::c", "a::b::d", "a::b::e", "a::b::f",)
        );

        assert_eq!(
            parse_use_tree("a::b::{c::{d, e, f}, g, h::{i, j, k}}").flatten(),
            parse_use_trees![
                "a::b::c::d",
                "a::b::c::e",
                "a::b::c::f",
                "a::b::g",
                "a::b::h::i",
                "a::b::h::j",
                "a::b::h::k",
            ]
        );
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

        assert!(parse_use_tree("foo").normalize() < parse_use_tree("Foo").normalize());
        assert!(parse_use_tree("foo").normalize() < parse_use_tree("foo::Bar").normalize());

        assert!(
            parse_use_tree("std::cmp::{d, c, b, a}").normalize()
                < parse_use_tree("std::cmp::{b, e, g, f}").normalize()
        );
    }
}
