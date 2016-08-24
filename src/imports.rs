// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use Indent;
use utils;
use syntax::codemap::{self, BytePos, Span};
use codemap::SpanUtils;
use lists::{write_list, itemize_list, ListItem, ListFormatting, SeparatorTactic, definitive_tactic};
use types::rewrite_path;
use rewrite::{Rewrite, RewriteContext};
use visitor::FmtVisitor;
use std::cmp::{self, Ordering};

use syntax::{ast, ptr};

fn path_of(a: &ast::ViewPath_) -> &ast::Path {
    match a {
        &ast::ViewPath_::ViewPathSimple(_, ref p) => p,
        &ast::ViewPath_::ViewPathGlob(ref p) => p,
        &ast::ViewPath_::ViewPathList(ref p, _) => p,
    }
}

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

fn compare_path_list_items(a: &ast::PathListItem, b: &ast::PathListItem) -> Ordering {
    let name_ordering = match a.node.name() {
        Some(a_name) => {
            match b.node.name() {
                Some(b_name) => a_name.name.as_str().cmp(&b_name.name.as_str()),
                None => Ordering::Greater,
            }
        }
        None => {
            match b.node.name() {
                Some(_) => Ordering::Less,
                None => Ordering::Equal,
            }
        }
    };
    if name_ordering == Ordering::Equal {
        match a.node.rename() {
            Some(a_rename) => {
                match b.node.rename() {
                    Some(b_rename) => a_rename.name.as_str().cmp(&b_rename.name.as_str()),
                    None => Ordering::Greater,
                }
            }
            None => {
                match b.node.name() {
                    Some(_) => Ordering::Less,
                    None => Ordering::Equal,
                }
            }
        }
    } else {
        name_ordering
    }
}

fn compare_path_list_item_lists(a_items: &Vec<ast::PathListItem>,
                                b_items: &Vec<ast::PathListItem>)
                                -> Ordering {
    let mut a = a_items.clone();
    let mut b = b_items.clone();
    a.sort_by(|a, b| compare_path_list_items(a, b));
    b.sort_by(|a, b| compare_path_list_items(a, b));
    for comparison_pair in a.iter().zip(b.iter()) {
        let ord = compare_path_list_items(comparison_pair.0, comparison_pair.1);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    a.len().cmp(&b.len())
}

fn compare_view_path_types(a: &ast::ViewPath_, b: &ast::ViewPath_) -> Ordering {
    use syntax::ast::ViewPath_::*;
    match (a, b) {
        (&ViewPathSimple(..), &ViewPathSimple(..)) => Ordering::Equal,
        (&ViewPathSimple(..), _) => Ordering::Less,
        (&ViewPathGlob(_), &ViewPathSimple(..)) => Ordering::Greater,
        (&ViewPathGlob(_), &ViewPathGlob(_)) => Ordering::Equal,
        (&ViewPathGlob(_), &ViewPathList(..)) => Ordering::Less,
        (&ViewPathList(_, ref a_items), &ViewPathList(_, ref b_items)) => {
            compare_path_list_item_lists(a_items, b_items)
        }
        (&ViewPathList(..), _) => Ordering::Greater,
    }
}

fn compare_view_paths(a: &ast::ViewPath_, b: &ast::ViewPath_) -> Ordering {
    match compare_paths(path_of(a), path_of(b)) {
        Ordering::Equal => compare_view_path_types(a, b),
        cmp => cmp,
    }
}

fn compare_use_items(a: &ast::Item, b: &ast::Item) -> Option<Ordering> {
    match (&a.node, &b.node) {
        (&ast::ItemKind::Use(ref a_vp), &ast::ItemKind::Use(ref b_vp)) => {
            Some(compare_view_paths(&a_vp.node, &b_vp.node))
        }
        _ => None,
    }
}

// TODO (some day) remove unused imports, expand globs, compress many single
// imports into a list import.

impl Rewrite for ast::ViewPath {
    // Returns an empty string when the ViewPath is empty (like foo::bar::{})
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        match self.node {
            ast::ViewPath_::ViewPathList(_, ref path_list) if path_list.is_empty() => {
                Some(String::new())
            }
            ast::ViewPath_::ViewPathList(ref path, ref path_list) => {
                rewrite_use_list(width, offset, path, path_list, self.span, context)
            }
            ast::ViewPath_::ViewPathGlob(_) => {
                // FIXME convert to list?
                None
            }
            ast::ViewPath_::ViewPathSimple(ident, ref path) => {
                let ident_str = ident.to_string();
                // 4 = " as ".len()
                let budget = try_opt!(width.checked_sub(ident_str.len() + 4));
                let path_str = try_opt!(rewrite_path(context, false, None, path, budget, offset));

                Some(if path.segments.last().unwrap().identifier == ident {
                    path_str
                } else {
                    format!("{} as {}", path_str, ident_str)
                })
            }
        }
    }
}

impl<'a> FmtVisitor<'a> {
    pub fn format_imports(&mut self, use_items: &[ptr::P<ast::Item>]) {
        // Find the location immediately before the first use item in the run. This must not lie
        // before the current `self.last_pos`
        let pos_before_first_use_item = use_items.first()
            .map(|p_i| cmp::max(self.last_pos, p_i.span.lo))
            .unwrap_or(self.last_pos);
        // Construct a list of pairs, each containing a `use` item and the start of span before
        // that `use` item.
        let mut last_pos_of_prev_use_item = pos_before_first_use_item;
        let mut ordered_use_items = use_items.iter()
            .map(|p_i| {
                let new_item = (&*p_i, last_pos_of_prev_use_item);
                last_pos_of_prev_use_item = p_i.span.hi;
                new_item
            })
            .collect::<Vec<_>>();
        let pos_after_last_use_item = last_pos_of_prev_use_item;
        // Order the imports by view-path & other import path properties
        ordered_use_items.sort_by(|a, b| compare_use_items(a.0, b.0).unwrap());
        // First, output the span before the first import
        let prev_span_str = self.snippet(codemap::mk_sp(self.last_pos, pos_before_first_use_item));
        // Look for purely trailing space at the start of the prefix snippet before a linefeed, or
        // a prefix that's entirely horizontal whitespace.
        let prefix_span_start = match prev_span_str.find('\n') {
            Some(offset) if prev_span_str[..offset].trim().is_empty() => {
                self.last_pos + BytePos(offset as u32)
            }
            None if prev_span_str.trim().is_empty() => pos_before_first_use_item,
            _ => self.last_pos,
        };
        // Look for indent (the line part preceding the use is all whitespace) and excise that
        // from the prefix
        let span_end = match prev_span_str.rfind('\n') {
            Some(offset) if prev_span_str[offset..].trim().is_empty() => {
                self.last_pos + BytePos(offset as u32)
            }
            _ => pos_before_first_use_item,
        };

        self.last_pos = prefix_span_start;
        self.format_missing(span_end);
        for ordered in ordered_use_items {
            // Fake out the formatter by setting `self.last_pos` to the appropriate location before
            // each item before visiting it.
            self.last_pos = ordered.1;
            self.visit_item(&ordered.0);
        }
        self.last_pos = pos_after_last_use_item;
    }

    pub fn format_import(&mut self, vis: &ast::Visibility, vp: &ast::ViewPath, span: Span) {
        let vis = utils::format_visibility(vis);
        let mut offset = self.block_indent;
        offset.alignment += vis.len() + "use ".len();
        // 1 = ";"
        match vp.rewrite(&self.get_context(),
                         self.config.max_width - offset.width() - 1,
                         offset) {
            Some(ref s) if s.is_empty() => {
                // Format up to last newline
                let prev_span = codemap::mk_sp(self.last_pos, source!(self, span).lo);
                let span_end = match self.snippet(prev_span).rfind('\n') {
                    Some(offset) => self.last_pos + BytePos(offset as u32),
                    None => source!(self, span).lo,
                };
                self.format_missing(span_end);
                self.last_pos = source!(self, span).hi;
            }
            Some(ref s) => {
                let s = format!("{}use {};", vis, s);
                self.format_missing_with_indent(source!(self, span).lo);
                self.buffer.push_str(&s);
                self.last_pos = source!(self, span).hi;
            }
            None => {
                self.format_missing_with_indent(source!(self, span).lo);
                self.format_missing(source!(self, span).hi);
            }
        }
    }
}

fn rewrite_single_use_list(path_str: Option<String>, vpi: &ast::PathListItem) -> String {
    let path_item_str = if let ast::PathListItemKind::Ident { name, .. } = vpi.node {
        // A name.
        match path_str {
            Some(path_str) => format!("{}::{}", path_str, name),
            None => name.to_string(),
        }
    } else {
        // `self`.
        match path_str {
            Some(path_str) => path_str,
            // This catches the import: use {self}, which is a compiler error, so we just
            // leave it alone.
            None => "{self}".to_owned(),
        }
    };

    append_alias(path_item_str, vpi)
}

fn rewrite_path_item(vpi: &&ast::PathListItem) -> Option<String> {
    let path_item_str = match vpi.node {
        ast::PathListItemKind::Ident { name, .. } => name.to_string(),
        ast::PathListItemKind::Mod { .. } => "self".to_owned(),
    };

    Some(append_alias(path_item_str, vpi))
}

fn append_alias(path_item_str: String, vpi: &ast::PathListItem) -> String {
    match vpi.node {
        ast::PathListItemKind::Ident { rename: Some(rename), .. } |
        ast::PathListItemKind::Mod { rename: Some(rename), .. } => {
            format!("{} as {}", path_item_str, rename)
        }
        _ => path_item_str,
    }
}

// Pretty prints a multi-item import.
// Assumes that path_list.len() > 0.
pub fn rewrite_use_list(width: usize,
                        offset: Indent,
                        path: &ast::Path,
                        path_list: &[ast::PathListItem],
                        span: Span,
                        context: &RewriteContext)
                        -> Option<String> {
    // Returns a different option to distinguish `::foo` and `foo`
    let opt_path_str = if !path.to_string().is_empty() {
        Some(path.to_string())
    } else if path.global {
        // path is absolute, we return an empty String to avoid a double `::`
        Some(String::new())
    } else {
        None
    };

    match path_list.len() {
        0 => unreachable!(),
        1 => return Some(rewrite_single_use_list(opt_path_str, &path_list[0])),
        _ => (),
    }

    // 2 = ::
    let path_separation_w = if opt_path_str.is_some() { 2 } else { 0 };
    // 1 = {
    let supp_indent = path.to_string().len() + path_separation_w + 1;
    // 1 = }
    let remaining_width = width.checked_sub(supp_indent + 1).unwrap_or(0);

    let mut items = {
        // Dummy value, see explanation below.
        let mut items = vec![ListItem::from_str("")];
        let iter = itemize_list(context.codemap,
                                path_list.iter(),
                                "}",
                                |vpi| vpi.span.lo,
                                |vpi| vpi.span.hi,
                                rewrite_path_item,
                                context.codemap.span_after(span, "{"),
                                span.hi);
        items.extend(iter);
        items
    };

    // We prefixed the item list with a dummy value so that we can
    // potentially move "self" to the front of the vector without touching
    // the rest of the items.
    let has_self = move_self_to_front(&mut items);
    let first_index = if has_self { 0 } else { 1 };

    if context.config.reorder_imported_names {
        items[1..].sort_by(|a, b| a.item.cmp(&b.item));
    }

    let tactic = definitive_tactic(&items[first_index..],
                                   ::lists::ListTactic::Mixed,
                                   remaining_width);
    let fmt = ListFormatting {
        tactic: tactic,
        separator: ",",
        trailing_separator: SeparatorTactic::Never,
        indent: offset + supp_indent,
        // FIXME This is too conservative, and will not use all width
        // available
        // (loose 1 column (";"))
        width: remaining_width,
        ends_with_newline: false,
        config: context.config,
    };
    let list_str = try_opt!(write_list(&items[first_index..], &fmt));

    Some(match opt_path_str {
        Some(opt_path_str) => format!("{}::{{{}}}", opt_path_str, list_str),
        None => format!("{{{}}}", list_str),
    })
}

// Returns true when self item was found.
fn move_self_to_front(items: &mut Vec<ListItem>) -> bool {
    match items.iter().position(|item| item.item.as_ref().map(|x| &x[..]) == Some("self")) {
        Some(pos) => {
            items[0] = items.remove(pos);
            true
        }
        None => false,
    }
}
