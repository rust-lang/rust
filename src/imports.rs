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
use lists::{write_list, itemize_list, ListItem, ListFormatting, SeparatorTactic, definitive_tactic};
use types::rewrite_path;
use utils::span_after;
use rewrite::{Rewrite, RewriteContext};

use syntax::ast;
use syntax::codemap::Span;

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

fn rewrite_single_use_list(path_str: String, vpi: &ast::PathListItem) -> String {
    let path_item_str = if let ast::PathListItem_::PathListIdent{ name, .. } = vpi.node {
        // A name.
        if path_str.is_empty() {
            name.to_string()
        } else {
            format!("{}::{}", path_str, name)
        }
    } else {
        // `self`.
        if !path_str.is_empty() {
            path_str
        } else {
            // This catches the import: use {self}, which is a compiler error, so we just
            // leave it alone.
            "{self}".to_owned()
        }
    };

    append_alias(path_item_str, vpi)
}

fn rewrite_path_item(vpi: &&ast::PathListItem) -> Option<String> {
    let path_item_str = match vpi.node {
        ast::PathListItem_::PathListIdent{ name, .. } => {
            name.to_string()
        }
        ast::PathListItem_::PathListMod{ .. } => {
            "self".to_owned()
        }
    };

    Some(append_alias(path_item_str, vpi))
}

fn append_alias(path_item_str: String, vpi: &ast::PathListItem) -> String {
    match vpi.node {
        ast::PathListItem_::PathListIdent{ rename: Some(rename), .. } |
        ast::PathListItem_::PathListMod{ rename: Some(rename), .. } => {
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
    // 1 = {}
    let budget = try_opt!(width.checked_sub(1));
    let path_str = try_opt!(rewrite_path(context, false, None, path, budget, offset));

    match path_list.len() {
        0 => unreachable!(),
        1 => return Some(rewrite_single_use_list(path_str, &path_list[0])),
        _ => (),
    }

    // 2 = ::
    let path_separation_w = if !path_str.is_empty() {
        2
    } else {
        0
    };
    // 1 = {
    let supp_indent = path_str.len() + path_separation_w + 1;
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
                                span_after(span, "{", context.codemap),
                                span.hi);
        items.extend(iter);
        items
    };

    // We prefixed the item list with a dummy value so that we can
    // potentially move "self" to the front of the vector without touching
    // the rest of the items.
    let has_self = move_self_to_front(&mut items);
    let first_index = if has_self {
        0
    } else {
        1
    };

    if context.config.reorder_imports {
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

    Some(if path_str.is_empty() {
        format!("{{{}}}", list_str)
    } else {
        format!("{}::{{{}}}", path_str, list_str)
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
