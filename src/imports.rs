// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use visitor::FmtVisitor;
use lists::{write_list, itemize_list, ListItem, ListFormatting, SeparatorTactic, ListTactic};
use utils::{span_after, format_visibility};

use syntax::ast;
use syntax::parse::token;
use syntax::print::pprust;
use syntax::codemap::Span;

// TODO (some day) remove unused imports, expand globs, compress many single imports into a list import

fn rewrite_single_use_list(path_str: String, vpi: ast::PathListItem, vis: &str) -> String {
    if let ast::PathListItem_::PathListIdent{ name, .. } = vpi.node {
        let name_str = token::get_ident(name).to_string();
        if path_str.len() == 0 {
            format!("{}use {};", vis, name_str)
        } else {
            format!("{}use {}::{};", vis, path_str, name_str)
        }
    } else {
        if path_str.len() != 0 {
            format!("{}use {};", vis, path_str)
        } else {
            // This catches the import: use {self}, which is a compiler error, so we just
            // leave it alone.
            format!("{}use {{self}};", vis)
        }
    }
}

impl<'a> FmtVisitor<'a> {
    // Basically just pretty prints a multi-item import.
    // Returns None when the import can be removed.
    pub fn rewrite_use_list(&self,
                            block_indent: usize,
                            one_line_budget: usize, // excluding indentation
                            multi_line_budget: usize,
                            path: &ast::Path,
                            path_list: &[ast::PathListItem],
                            visibility: ast::Visibility,
                            span: Span)
                            -> Option<String> {
        let path_str = pprust::path_to_string(path);
        let vis = format_visibility(visibility);

        match path_list.len() {
            0 => return None,
            1 => return Some(rewrite_single_use_list(path_str, path_list[0], vis)),
            _ => ()
        }

        // 2 = ::
        let path_separation_w = if path_str.len() > 0 { 2 } else { 0 };
        // 5 = "use " + {
        let indent = path_str.len() + 5 + path_separation_w + vis.len();

        // 2 = } + ;
        let used_width = indent + 2;

        // Break as early as possible when we've blown our budget.
        let remaining_line_budget = one_line_budget.checked_sub(used_width).unwrap_or(0);
        let remaining_multi_budget = multi_line_budget.checked_sub(used_width).unwrap_or(0);

        let fmt = ListFormatting {
            tactic: ListTactic::Mixed,
            separator: ",",
            trailing_separator: SeparatorTactic::Never,
            indent: block_indent + indent,
            h_width: remaining_line_budget,
            v_width: remaining_multi_budget,
            ends_with_newline: true,
        };

        let mut items = itemize_list(self.codemap,
                                     vec![ListItem::from_str("")], /* Dummy value, explanation
                                                                    * below */
                                     path_list.iter(),
                                     ",",
                                     "}",
                                     |vpi| vpi.span.lo,
                                     |vpi| vpi.span.hi,
                                     |vpi| match vpi.node {
                                         ast::PathListItem_::PathListIdent{ name, .. } => {
                                             token::get_ident(name).to_string()
                                         }
                                         ast::PathListItem_::PathListMod{ .. } => {
                                             "self".to_owned()
                                         }
                                     },
                                     span_after(span, "{", self.codemap),
                                     span.hi);

        // We prefixed the item list with a dummy value so that we can
        // potentially move "self" to the front of the vector without touching
        // the rest of the items.
        // FIXME: Make more efficient by using a linked list? That would
        // require changes to the signatures of itemize_list and write_list.
        let has_self = move_self_to_front(&mut items);
        let first_index = if has_self { 0 } else { 1 };

        if self.config.reorder_imports {
            items[1..].sort_by(|a, b| a.item.cmp(&b.item));
        }

        let list = write_list(&items[first_index..], &fmt);

        Some(if path_str.len() == 0 {
            format!("{}use {{{}}};", vis, list)
        } else {
            format!("{}use {}::{{{}}};", vis, path_str, list)
        })
    }
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
