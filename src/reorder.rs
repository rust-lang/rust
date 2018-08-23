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
//! and constants comes before methods).

// FIXME(#2455): Reorder trait items.

use config::Config;
use syntax::{ast, attr, source_map::Span};

use attr::filter_inline_attrs;
use comment::combine_strs_with_missing_comments;
use imports::{merge_use_trees, UseTree};
use items::{is_mod_decl, rewrite_extern_crate, rewrite_mod};
use lists::{itemize_list, write_list, ListFormatting, ListItem};
use rewrite::{Rewrite, RewriteContext};
use shape::Shape;
use source_map::LineRangeUtils;
use spanned::Spanned;
use utils::mk_sp;
use visitor::FmtVisitor;

use std::cmp::{Ord, Ordering};

/// Choose the ordering between the given two items.
fn compare_items(a: &ast::Item, b: &ast::Item) -> Ordering {
    match (&a.node, &b.node) {
        (&ast::ItemKind::Mod(..), &ast::ItemKind::Mod(..)) => {
            a.ident.name.as_str().cmp(&b.ident.name.as_str())
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

fn wrap_reorderable_items(
    context: &RewriteContext,
    list_items: &[ListItem],
    shape: Shape,
) -> Option<String> {
    let fmt = ListFormatting::new(shape, context.config).separator("");
    write_list(list_items, &fmt)
}

fn rewrite_reorderable_item(
    context: &RewriteContext,
    item: &ast::Item,
    shape: Shape,
) -> Option<String> {
    let attrs = filter_inline_attrs(&item.attrs, item.span());
    let attrs_str = attrs.rewrite(context, shape)?;

    let missed_span = if attrs.is_empty() {
        mk_sp(item.span.lo(), item.span.lo())
    } else {
        mk_sp(attrs.last().unwrap().span.hi(), item.span.lo())
    };

    let item_str = match item.node {
        ast::ItemKind::ExternCrate(..) => rewrite_extern_crate(context, item)?,
        ast::ItemKind::Mod(..) => rewrite_mod(context, item),
        _ => return None,
    };

    combine_strs_with_missing_comments(context, &attrs_str, &item_str, missed_span, shape, false)
}

/// Rewrite a list of items with reordering. Every item in `items` must have
/// the same `ast::ItemKind`.
fn rewrite_reorderable_items(
    context: &RewriteContext,
    reorderable_items: &[&ast::Item],
    shape: Shape,
    span: Span,
) -> Option<String> {
    match reorderable_items[0].node {
        // FIXME: Remove duplicated code.
        ast::ItemKind::Use(..) => {
            let mut normalized_items: Vec<_> = reorderable_items
                .iter()
                .filter_map(|item| UseTree::from_ast_with_normalization(context, item))
                .collect();
            let cloned = normalized_items.clone();
            // Add comments before merging.
            let list_items = itemize_list(
                context.snippet_provider,
                cloned.iter(),
                "",
                ";",
                |item| item.span().lo(),
                |item| item.span().hi(),
                |_item| Some("".to_owned()),
                span.lo(),
                span.hi(),
                false,
            );
            for (item, list_item) in normalized_items.iter_mut().zip(list_items) {
                item.list_item = Some(list_item.clone());
            }
            if context.config.merge_imports() {
                normalized_items = merge_use_trees(normalized_items);
            }
            normalized_items.sort();

            // 4 = "use ", 1 = ";"
            let nested_shape = shape.offset_left(4)?.sub_width(1)?;
            let item_vec: Vec<_> = normalized_items
                .into_iter()
                .map(|use_tree| ListItem {
                    item: use_tree.rewrite_top_level(context, nested_shape),
                    ..use_tree.list_item.unwrap_or_else(ListItem::empty)
                }).collect();

            wrap_reorderable_items(context, &item_vec, nested_shape)
        }
        _ => {
            let list_items = itemize_list(
                context.snippet_provider,
                reorderable_items.iter(),
                "",
                ";",
                |item| item.span().lo(),
                |item| item.span().hi(),
                |item| rewrite_reorderable_item(context, item, shape),
                span.lo(),
                span.hi(),
                false,
            );

            let mut item_pair_vec: Vec<_> = list_items.zip(reorderable_items.iter()).collect();
            item_pair_vec.sort_by(|a, b| compare_items(a.1, b.1));
            let item_vec: Vec<_> = item_pair_vec.into_iter().map(|pair| pair.0).collect();

            wrap_reorderable_items(context, &item_vec, shape)
        }
    }
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
    fn from(item: &ast::Item) -> Self {
        match item.node {
            _ if contains_macro_use_attr(item) => ReorderableItemKind::Other,
            ast::ItemKind::ExternCrate(..) => ReorderableItemKind::ExternCrate,
            ast::ItemKind::Mod(..) if is_mod_decl(item) => ReorderableItemKind::Mod,
            ast::ItemKind::Use(..) => ReorderableItemKind::Use,
            _ => ReorderableItemKind::Other,
        }
    }

    fn is_same_item_kind(&self, item: &ast::Item) -> bool {
        ReorderableItemKind::from(item) == *self
    }

    fn is_reorderable(&self, config: &Config) -> bool {
        match *self {
            ReorderableItemKind::ExternCrate => config.reorder_imports(),
            ReorderableItemKind::Mod => config.reorder_modules(),
            ReorderableItemKind::Use => config.reorder_imports(),
            ReorderableItemKind::Other => false,
        }
    }

    fn in_group(&self) -> bool {
        match *self {
            ReorderableItemKind::ExternCrate
            | ReorderableItemKind::Mod
            | ReorderableItemKind::Use => true,
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
        let mut last = self.source_map.lookup_line_range(items[0].span());
        let item_length = items
            .iter()
            .take_while(|ppi| {
                item_kind.is_same_item_kind(&***ppi)
                    && (!in_group || {
                        let current = self.source_map.lookup_line_range(ppi.span());
                        let in_same_group = current.lo < last.hi + 2;
                        last = current;
                        in_same_group
                    })
            }).count();
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
                    self.walk_reorderable_items(items, item_kind, item_kind.in_group());
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
