// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lists::{write_list, itemize_list, ListItem, ListFormatting, SeparatorTactic, ListTactic};
use utils::span_after;
use rewrite::{Rewrite, RewriteContext};
use config::Config;

use syntax::ast;
use syntax::print::pprust;
use syntax::codemap::{CodeMap, Span};

// TODO (some day) remove unused imports, expand globs, compress many single imports into a list import

impl Rewrite for ast::ViewPath {
    // Returns an empty string when the ViewPath is empty (like foo::bar::{})
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: usize) -> Option<String> {
        match self.node {
            ast::ViewPath_::ViewPathList(ref path, ref path_list) => {
                Some(rewrite_use_list(width,
                                      offset,
                                      path,
                                      path_list,
                                      self.span,
                                      context.codemap,
                                      context.config).unwrap_or("".to_owned()))
            }
            ast::ViewPath_::ViewPathGlob(_) => {
                // FIXME convert to list?
                None
            }
            ast::ViewPath_::ViewPathSimple(ident, ref path) => {
                let path_str = pprust::path_to_string(path);

                Some(if path.segments.last().unwrap().identifier == ident {
                         path_str
                     } else {
                         format!("{} as {}", path_str, ident)
                     })
            }
        }
    }
}

fn rewrite_single_use_list(path_str: String, vpi: ast::PathListItem) -> String {
    if let ast::PathListItem_::PathListIdent{ name, .. } = vpi.node {
        if path_str.len() == 0 {
            name.to_string()
        } else {
            format!("{}::{}", path_str, name)
        }
    } else {
        if path_str.len() != 0 {
            path_str
        } else {
            // This catches the import: use {self}, which is a compiler error, so we just
            // leave it alone.
            "{self}".to_owned()
        }
    }
}

// Basically just pretty prints a multi-item import.
// Returns None when the import can be removed.
pub fn rewrite_use_list(width: usize,
                        offset: usize,
                        path: &ast::Path,
                        path_list: &[ast::PathListItem],
                        span: Span,
                        codemap: &CodeMap,
                        config: &Config)
                        -> Option<String> {
    let path_str = pprust::path_to_string(path);

    match path_list.len() {
        0 => return None,
        1 => return Some(rewrite_single_use_list(path_str, path_list[0])),
        _ => ()
    }

    // 2 = ::
    let path_separation_w = if path_str.len() > 0 {
        2
    } else {
        0
    };
    // 1 = {
    let supp_indent = path_str.len() + path_separation_w + 1;
    // 1 = }
    let remaining_width = width.checked_sub(supp_indent + 1).unwrap_or(0);

    let fmt = ListFormatting {
        tactic: ListTactic::Mixed,
        separator: ",",
        trailing_separator: SeparatorTactic::Never,
        indent: offset + supp_indent,
        h_width: remaining_width,
        // FIXME This is too conservative, and will not use all width
        // available
        // (loose 1 column (";"))
        v_width: remaining_width,
        ends_with_newline: true,
    };

    let mut items = itemize_list(codemap,
                                 vec![ListItem::from_str("")], /* Dummy value, explanation
                                                                * below */
                                 path_list.iter(),
                                 ",",
                                 "}",
                                 |vpi| vpi.span.lo,
                                 |vpi| vpi.span.hi,
                                 |vpi| match vpi.node {
                                     ast::PathListItem_::PathListIdent{ name, .. } => {
                                         name.to_string()
                                     }
                                     ast::PathListItem_::PathListMod{ .. } => {
                                         "self".to_owned()
                                     }
                                 },
                                 span_after(span, "{", codemap),
                                 span.hi);

    // We prefixed the item list with a dummy value so that we can
    // potentially move "self" to the front of the vector without touching
    // the rest of the items.
    // FIXME: Make more efficient by using a linked list? That would
    // require changes to the signatures of itemize_list and write_list.
    let has_self = move_self_to_front(&mut items);
    let first_index = if has_self {
        0
    } else {
        1
    };

    if config.reorder_imports {
        items[1..].sort_by(|a, b| a.item.cmp(&b.item));
    }

    let list = write_list(&items[first_index..], &fmt);

    Some(if path_str.len() == 0 {
            format!("{{{}}}", list)
        } else {
            format!("{}::{{{}}}", path_str, list)
        })
}

// Returns true when self item was found.
fn move_self_to_front(items: &mut Vec<ListItem>) -> bool {
    match items.iter().position(|item| item.item == "self") {
        Some(pos) => {
            items[0] = items.remove(pos);
            true
        },
        None => false
    }
}
