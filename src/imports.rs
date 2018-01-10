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

use syntax::ast;
use syntax::codemap::{BytePos, Span};

use codemap::SpanUtils;
use comment::combine_strs_with_missing_comments;
use config::IndentStyle;
use lists::{definitive_tactic, itemize_list, write_list, DefinitiveListTactic, ListFormatting,
            ListItem, Separator, SeparatorPlace, SeparatorTactic};
use rewrite::{Rewrite, RewriteContext};
use shape::Shape;
use spanned::Spanned;
use types::{rewrite_path, PathContext};
use utils::{format_visibility, mk_sp};
use visitor::{rewrite_extern_crate, FmtVisitor};

fn compare_path_segments(a: &ast::PathSegment, b: &ast::PathSegment) -> Ordering {
    a.identifier.name.as_str().cmp(&b.identifier.name.as_str())
}

fn compare_paths(a: &ast::Path, b: &ast::Path) -> Ordering {
    for segment in a.segments.iter().zip(b.segments.iter()) {
        let ord = compare_path_segments(segment.0, segment.1);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    a.segments.len().cmp(&b.segments.len())
}

fn compare_use_trees(a: &ast::UseTree, b: &ast::UseTree, nested: bool) -> Ordering {
    use ast::UseTreeKind::*;

    // `use_nested_groups` is not yet supported, remove the `if !nested` when support will be
    // fully added
    if !nested {
        let paths_cmp = compare_paths(&a.prefix, &b.prefix);
        if paths_cmp != Ordering::Equal {
            return paths_cmp;
        }
    }

    match (&a.kind, &b.kind) {
        (&Simple(ident_a), &Simple(ident_b)) => {
            let name_a = &*path_to_imported_ident(&a.prefix).name.as_str();
            let name_b = &*path_to_imported_ident(&b.prefix).name.as_str();
            let name_ordering = if name_a == "self" {
                if name_b == "self" {
                    Ordering::Equal
                } else {
                    Ordering::Less
                }
            } else if name_b == "self" {
                Ordering::Greater
            } else {
                name_a.cmp(name_b)
            };
            if name_ordering == Ordering::Equal {
                if ident_a.name.as_str() != name_a {
                    if ident_b.name.as_str() != name_b {
                        ident_a.name.as_str().cmp(&ident_b.name.as_str())
                    } else {
                        Ordering::Greater
                    }
                } else {
                    Ordering::Less
                }
            } else {
                name_ordering
            }
        }
        (&Glob, &Glob) => Ordering::Equal,
        (&Simple(_), _) | (&Glob, &Nested(_)) => Ordering::Less,
        (&Nested(ref a_items), &Nested(ref b_items)) => {
            let mut a = a_items
                .iter()
                .map(|&(ref tree, _)| tree.clone())
                .collect::<Vec<_>>();
            let mut b = b_items
                .iter()
                .map(|&(ref tree, _)| tree.clone())
                .collect::<Vec<_>>();
            a.sort_by(|a, b| compare_use_trees(a, b, true));
            b.sort_by(|a, b| compare_use_trees(a, b, true));
            for comparison_pair in a.iter().zip(b.iter()) {
                let ord = compare_use_trees(comparison_pair.0, comparison_pair.1, true);
                if ord != Ordering::Equal {
                    return ord;
                }
            }
            a.len().cmp(&b.len())
        }
        (&Glob, &Simple(_)) | (&Nested(_), _) => Ordering::Greater,
    }
}

fn compare_use_items(a: &ast::Item, b: &ast::Item) -> Option<Ordering> {
    match (&a.node, &b.node) {
        (&ast::ItemKind::Use(ref a_tree), &ast::ItemKind::Use(ref b_tree)) => {
            Some(compare_use_trees(a_tree, b_tree, false))
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
                return Some(result);
            }

            // `extern crate foo as bar;`
            //                      ^^^ Comparing this.
            let result = match (a_name, b_name) {
                (Some(..), None) => Ordering::Greater,
                (None, Some(..)) => Ordering::Less,
                (None, None) => Ordering::Equal,
                (Some(..), Some(..)) => a.ident.name.cmp(&b.ident.name),
            };
            Some(result)
        }
        _ => None,
    }
}

// TODO (some day) remove unused imports, expand globs, compress many single
// imports into a list import.

fn rewrite_prefix(path: &ast::Path, context: &RewriteContext, shape: Shape) -> Option<String> {
    if path.segments.len() > 1 && path_to_imported_ident(path).to_string() == "self" {
        let path = &ast::Path {
            span: path.span,
            segments: path.segments[..path.segments.len() - 1].to_owned(),
        };
        rewrite_path(context, PathContext::Import, None, path, shape)
    } else {
        rewrite_path(context, PathContext::Import, None, path, shape)
    }
}

impl Rewrite for ast::UseTree {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        match self.kind {
            ast::UseTreeKind::Nested(ref items) => {
                rewrite_nested_use_tree(shape, &self.prefix, items, self.span, context)
            }
            ast::UseTreeKind::Glob => {
                let prefix_shape = shape.sub_width(3)?;

                if !self.prefix.segments.is_empty() {
                    let path_str = rewrite_prefix(&self.prefix, context, prefix_shape)?;
                    Some(format!("{}::*", path_str))
                } else {
                    Some("*".to_owned())
                }
            }
            ast::UseTreeKind::Simple(ident) => {
                let ident_str = ident.to_string();

                // 4 = " as ".len()
                let is_same_name_bind = path_to_imported_ident(&self.prefix) == ident;
                let prefix_shape = if is_same_name_bind {
                    shape
                } else {
                    shape.sub_width(ident_str.len() + 4)?
                };
                let path_str = rewrite_prefix(&self.prefix, context, prefix_shape)
                    .unwrap_or_else(|| context.snippet(self.prefix.span).to_owned());

                if is_same_name_bind {
                    Some(path_str)
                } else {
                    Some(format!("{} as {}", path_str, ident_str))
                }
            }
        }
    }
}

fn is_unused_import(tree: &ast::UseTree, attrs: &[ast::Attribute]) -> bool {
    attrs.is_empty() && is_unused_import_inner(tree)
}

fn is_unused_import_inner(tree: &ast::UseTree) -> bool {
    match tree.kind {
        ast::UseTreeKind::Nested(ref items) => match items.len() {
            0 => true,
            1 => is_unused_import_inner(&items[0].0),
            _ => false,
        },
        _ => false,
    }
}

// Rewrite `use foo;` WITHOUT attributes.
fn rewrite_import(
    context: &RewriteContext,
    vis: &ast::Visibility,
    tree: &ast::UseTree,
    attrs: &[ast::Attribute],
    shape: Shape,
) -> Option<String> {
    let vis = format_visibility(vis);
    // 4 = `use `, 1 = `;`
    let rw = shape
        .offset_left(vis.len() + 4)
        .and_then(|shape| shape.sub_width(1))
        .and_then(|shape| {
            // If we have an empty nested group with no attributes, we erase it
            if is_unused_import(tree, attrs) {
                Some("".to_owned())
            } else {
                tree.rewrite(context, shape)
            }
        });
    match rw {
        Some(ref s) if !s.is_empty() => Some(format!("{}use {};", vis, s)),
        _ => rw,
    }
}

fn rewrite_imports(
    context: &RewriteContext,
    use_items: &[&ast::Item],
    shape: Shape,
    span: Span,
) -> Option<String> {
    let items = itemize_list(
        context.codemap,
        use_items.iter(),
        "",
        ";",
        |item| item.span().lo(),
        |item| item.span().hi(),
        |item| {
            let attrs_str = item.attrs.rewrite(context, shape)?;

            let missed_span = if item.attrs.is_empty() {
                mk_sp(item.span.lo(), item.span.lo())
            } else {
                mk_sp(item.attrs.last().unwrap().span.hi(), item.span.lo())
            };

            let item_str = match item.node {
                ast::ItemKind::Use(ref tree) => {
                    rewrite_import(context, &item.vis, tree, &item.attrs, shape)?
                }
                ast::ItemKind::ExternCrate(..) => rewrite_extern_crate(context, item)?,
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
    let mut item_pair_vec: Vec<_> = items.zip(use_items.iter()).collect();
    item_pair_vec.sort_by(|a, b| compare_use_items(a.1, b.1).unwrap());
    let item_vec: Vec<_> = item_pair_vec.into_iter().map(|pair| pair.0).collect();

    let fmt = ListFormatting {
        tactic: DefinitiveListTactic::Vertical,
        separator: "",
        trailing_separator: SeparatorTactic::Never,
        separator_place: SeparatorPlace::Back,
        shape: shape,
        ends_with_newline: true,
        preserve_newline: false,
        config: context.config,
    };

    write_list(&item_vec, &fmt)
}

impl<'a> FmtVisitor<'a> {
    pub fn format_imports(&mut self, use_items: &[&ast::Item]) {
        if use_items.is_empty() {
            return;
        }

        let lo = use_items.first().unwrap().span().lo();
        let hi = use_items.last().unwrap().span().hi();
        let span = mk_sp(lo, hi);
        let rw = rewrite_imports(&self.get_context(), use_items, self.shape(), span);
        self.push_rewrite(span, rw);
    }

    pub fn format_import(&mut self, item: &ast::Item, tree: &ast::UseTree) {
        let span = item.span;
        let shape = self.shape();
        let rw = rewrite_import(&self.get_context(), &item.vis, tree, &item.attrs, shape);
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

fn rewrite_nested_use_tree_single(
    context: &RewriteContext,
    path_str: &str,
    tree: &ast::UseTree,
    shape: Shape,
) -> Option<String> {
    match tree.kind {
        ast::UseTreeKind::Simple(rename) => {
            let ident = path_to_imported_ident(&tree.prefix);
            let mut item_str = rewrite_prefix(&tree.prefix, context, shape)?;
            if item_str == "self" {
                item_str = "".to_owned();
            }

            let path_item_str = if path_str.is_empty() {
                if item_str.is_empty() {
                    "self".to_owned()
                } else {
                    item_str
                }
            } else if item_str.is_empty() {
                path_str.to_owned()
            } else {
                format!("{}::{}", path_str, item_str)
            };

            Some(if ident == rename {
                path_item_str
            } else {
                format!("{} as {}", path_item_str, rename)
            })
        }
        ast::UseTreeKind::Glob | ast::UseTreeKind::Nested(..) => {
            // 2 = "::"
            let nested_shape = shape.offset_left(path_str.len() + 2)?;
            tree.rewrite(context, nested_shape)
                .map(|item| format!("{}::{}", path_str, item))
        }
    }
}

#[derive(Eq, PartialEq)]
enum ImportItem<'a> {
    // `self` or `self as a`
    SelfImport(&'a str),
    // name_one, name_two, ...
    SnakeCase(&'a str),
    // NameOne, NameTwo, ...
    CamelCase(&'a str),
    // NAME_ONE, NAME_TWO, ...
    AllCaps(&'a str),
    // Failed to format the import item
    Invalid,
}

impl<'a> ImportItem<'a> {
    fn from_str(s: &str) -> ImportItem {
        if s == "self" || s.starts_with("self as") {
            ImportItem::SelfImport(s)
        } else if s.chars().all(|c| c.is_lowercase() || c == '_' || c == ' ') {
            ImportItem::SnakeCase(s)
        } else if s.chars().all(|c| c.is_uppercase() || c == '_' || c == ' ') {
            ImportItem::AllCaps(s)
        } else {
            ImportItem::CamelCase(s)
        }
    }

    fn from_opt_str(s: Option<&String>) -> ImportItem {
        s.map_or(ImportItem::Invalid, |s| ImportItem::from_str(s))
    }

    fn to_str(&self) -> Option<&str> {
        match *self {
            ImportItem::SelfImport(s)
            | ImportItem::SnakeCase(s)
            | ImportItem::CamelCase(s)
            | ImportItem::AllCaps(s) => Some(s),
            ImportItem::Invalid => None,
        }
    }

    fn to_u32(&self) -> u32 {
        match *self {
            ImportItem::SelfImport(..) => 0,
            ImportItem::SnakeCase(..) => 1,
            ImportItem::CamelCase(..) => 2,
            ImportItem::AllCaps(..) => 3,
            ImportItem::Invalid => 4,
        }
    }
}

impl<'a> PartialOrd for ImportItem<'a> {
    fn partial_cmp(&self, other: &ImportItem<'a>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for ImportItem<'a> {
    fn cmp(&self, other: &ImportItem<'a>) -> Ordering {
        let res = self.to_u32().cmp(&other.to_u32());
        if res != Ordering::Equal {
            return res;
        }
        self.to_str().map_or(Ordering::Greater, |self_str| {
            other
                .to_str()
                .map_or(Ordering::Less, |other_str| self_str.cmp(other_str))
        })
    }
}

// Pretty prints a multi-item import.
// If the path list is empty, it leaves the braces empty.
fn rewrite_nested_use_tree(
    shape: Shape,
    path: &ast::Path,
    trees: &[(ast::UseTree, ast::NodeId)],
    span: Span,
    context: &RewriteContext,
) -> Option<String> {
    // Returns a different option to distinguish `::foo` and `foo`
    let path_str = rewrite_path(context, PathContext::Import, None, path, shape)?;

    match trees.len() {
        0 => {
            let shape = shape.offset_left(path_str.len() + 3)?;
            return rewrite_path(context, PathContext::Import, None, path, shape)
                .map(|path_str| format!("{}::{{}}", path_str));
        }
        1 => {
            return rewrite_nested_use_tree_single(context, &path_str, &trees[0].0, shape);
        }
        _ => (),
    }

    let path_str = if path_str.is_empty() {
        path_str
    } else {
        format!("{}::", path_str)
    };

    // 2 = "{}"
    let remaining_width = shape.width.checked_sub(path_str.len() + 2).unwrap_or(0);
    let nested_indent = match context.config.imports_indent() {
        IndentStyle::Block => shape.indent.block_indent(context.config),
        // 1 = `{`
        IndentStyle::Visual => shape.visual_indent(path_str.len() + 1).indent,
    };

    let nested_shape = match context.config.imports_indent() {
        IndentStyle::Block => Shape::indented(nested_indent, context.config).sub_width(1)?,
        IndentStyle::Visual => Shape::legacy(remaining_width, nested_indent),
    };

    let mut items = {
        // Dummy value, see explanation below.
        let mut items = vec![ListItem::from_str("")];
        let iter = itemize_list(
            context.codemap,
            trees.iter().map(|tree| &tree.0),
            "}",
            ",",
            |tree| tree.span.lo(),
            |tree| tree.span.hi(),
            |tree| tree.rewrite(context, nested_shape),
            context.codemap.span_after(span, "{"),
            span.hi(),
            false,
        );
        items.extend(iter);
        items
    };

    // We prefixed the item list with a dummy value so that we can
    // potentially move "self" to the front of the vector without touching
    // the rest of the items.
    let has_self = move_self_to_front(&mut items);
    let first_index = if has_self { 0 } else { 1 };

    if context.config.reorder_imported_names() {
        items[1..].sort_by(|a, b| {
            let a = ImportItem::from_opt_str(a.item.as_ref());
            let b = ImportItem::from_opt_str(b.item.as_ref());
            a.cmp(&b)
        });
    }

    let tactic = definitive_tactic(
        &items[first_index..],
        context.config.imports_layout(),
        Separator::Comma,
        remaining_width,
    );

    let ends_with_newline = context.config.imports_indent() == IndentStyle::Block
        && tactic != DefinitiveListTactic::Horizontal;

    let fmt = ListFormatting {
        tactic: tactic,
        separator: ",",
        trailing_separator: if ends_with_newline {
            context.config.trailing_comma()
        } else {
            SeparatorTactic::Never
        },
        separator_place: SeparatorPlace::Back,
        shape: nested_shape,
        ends_with_newline: ends_with_newline,
        preserve_newline: true,
        config: context.config,
    };
    let list_str = write_list(&items[first_index..], &fmt)?;

    let result = if list_str.contains('\n') && context.config.imports_indent() == IndentStyle::Block
    {
        format!(
            "{}{{\n{}{}\n{}}}",
            path_str,
            nested_shape.indent.to_string(context.config),
            list_str,
            shape.indent.to_string(context.config)
        )
    } else {
        format!("{}{{{}}}", path_str, list_str)
    };
    Some(result)
}

// Returns true when self item was found.
fn move_self_to_front(items: &mut Vec<ListItem>) -> bool {
    match items
        .iter()
        .position(|item| item.item.as_ref().map(|x| &x[..]) == Some("self"))
    {
        Some(pos) => {
            items[0] = items.remove(pos);
            true
        }
        None => false,
    }
}

fn path_to_imported_ident(path: &ast::Path) -> ast::Ident {
    path.segments.last().unwrap().identifier
}
