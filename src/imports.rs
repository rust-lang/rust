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
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::print::pprust;

use {IDEAL_WIDTH, MAX_WIDTH};

// TODO remove empty lists (if they're even possible)
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
    pub fn rewrite_use_list(&mut self,
                            path: &ast::Path,
                            path_list: &[ast::PathListItem],
                            visibility: ast::Visibility,
                            vp_span: Span) -> String {
        let path_str = pprust::path_to_string(&path);

        let vis = match visibility {
            ast::Public => "pub ",
            _ => ""
        };

        if path_list.len() == 1 {
            return rewrite_single_use_list(path_str, path_list[0], vis);
        }

        // FIXME check indentation
        let l_loc = self.codemap.lookup_char_pos(vp_span.lo);

        // 1 = {
        let mut indent = l_loc.col.0 + path_str.len() + 1;
        if path_str.len() > 0 {
            // 2 = ::
            indent += 2;
        }
        // 2 = } + ;
        let used_width = indent + 2 + vis.len();
        let budget = if used_width >= IDEAL_WIDTH {
            if used_width < MAX_WIDTH {
                MAX_WIDTH - used_width
            } else {
                // Give up
                return String::new();
            }
        } else {
            IDEAL_WIDTH - used_width
        };
        let fmt = ListFormatting {
            tactic: ListTactic::Mixed,
            separator: ",",
            trailing_separator: SeparatorTactic::Never,
            indent: indent,
            h_width: budget,
            v_width: budget,
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
