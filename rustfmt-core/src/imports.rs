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
use syntax::ast;
use syntax::codemap::{BytePos, Span};

use codemap::SpanUtils;
use config::IndentStyle;
use lists::{definitive_tactic, itemize_list, write_list, ListFormatting, ListItem, Separator};
use rewrite::{Rewrite, RewriteContext};
use shape::Shape;
use types::{rewrite_path, PathContext};
use utils::{format_visibility, mk_sp};
use visitor::FmtVisitor;

/// Returns a name imported by a `use` declaration. e.g. returns `Ordering`
/// for `std::cmp::Ordering` and `self` for `std::cmp::self`.
pub fn path_to_imported_ident(path: &ast::Path) -> ast::Ident {
    path.segments.last().unwrap().identifier
}

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
pub fn rewrite_import(
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

impl<'a> FmtVisitor<'a> {
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
        tactic,
        separator: ",",
        trailing_separator: if ends_with_newline {
            context.config.trailing_comma()
        } else {
            SeparatorTactic::Never
        },
        separator_place: SeparatorPlace::Back,
        shape: nested_shape,
        ends_with_newline,
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
