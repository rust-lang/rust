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
use lists::{write_list, ListFormatting, SeparatorTactic, ListTactic};

use syntax::ast;
use syntax::parse::token;
use syntax::print::pprust;

// TODO change import lists with one item to a single import
//      remove empty lists (if they're even possible)
// TODO (some day) remove unused imports, expand globs, compress many single imports into a list import

impl<'a> FmtVisitor<'a> {
    // Basically just pretty prints a multi-item import.
    pub fn rewrite_use_list(&mut self,
                            block_indent: usize,
                            one_line_budget: usize, // excluding indentation
                            multi_line_budget: usize,
                            path: &ast::Path,
                            path_list: &[ast::PathListItem],
                            visibility: ast::Visibility) -> String {
        let path_str = pprust::path_to_string(&path);

        let vis = match visibility {
            ast::Public => "pub ",
            _ => ""
        };

        // 2 = ::
        let path_separation_w = if path_str.len() > 0 { 2 } else { 0 };
        // 5 = "use " + {
        let indent = path_str.len() + 5 + path_separation_w + vis.len();
        // 2 = } + ;
        let used_width = indent + 2;

        // Break as early as possible when we've blown our budget.
        let remaining_line_budget = if used_width > one_line_budget {
            0
        } else {
            one_line_budget - used_width
        };
        let remaining_multi_budget = if used_width > multi_line_budget {
            0
        } else {
            multi_line_budget - used_width
        };

        let fmt = ListFormatting {
            tactic: ListTactic::Mixed,
            separator: ",",
            trailing_separator: SeparatorTactic::Never,
            indent: block_indent + indent,
            h_width: remaining_line_budget,
            v_width: remaining_multi_budget,
        };

        // TODO handle any comments inbetween items.
        // If `self` is in the list, put it first.
        let head = if path_list.iter().any(|vpi|
            if let ast::PathListItem_::PathListMod{ .. } = vpi.node {
                true
            } else {
                false
            }
        ) {
            Some(("self".to_owned(), String::new()))
        } else {
            None
        };

        let items: Vec<_> = head.into_iter().chain(path_list.iter().filter_map(|vpi| {
            match vpi.node {
                ast::PathListItem_::PathListIdent{ name, .. } => {
                    Some((token::get_ident(name).to_string(), String::new()))
                }
                // Skip `self`, because we added it above.
                ast::PathListItem_::PathListMod{ .. } => None,
            }
        })).collect();
        if path_str.len() == 0 {
            format!("{}use {{{}}};", vis, write_list(&items, &fmt))
        } else {
            format!("{}use {}::{{{}}};", vis, path_str, write_list(&items, &fmt))
        }
    }
}
