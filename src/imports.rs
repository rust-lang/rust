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

use IDEAL_WIDTH;

// TODO change import lists with one item to a single import
//      remove empty lists (if they're even possible)
// TODO (some day) remove unused imports, expand globs, compress many single imports into a list import

impl<'a> FmtVisitor<'a> {
    // Basically just pretty prints a multi-item import.
    pub fn rewrite_use_list(&mut self,
                        path: &ast::Path,
                        path_list: &[ast::PathListItem],
                        vp_span: Span) -> String {
        // FIXME remove unused imports

        // FIXME check indentation
        let l_loc = self.codemap.lookup_char_pos(vp_span.lo);

        let path_str = pprust::path_to_string(&path);

        // 3 = :: + {
        let indent = l_loc.col.0 + path_str.len() + 3;
        let fmt = ListFormatting {
            tactic: ListTactic::Mixed,
            separator: ",",
            trailing_separator: SeparatorTactic::Never,
            indent: indent,
            // 2 = } + ;
            h_width: IDEAL_WIDTH - (indent + path_str.len() + 2),
            v_width: IDEAL_WIDTH - (indent + path_str.len() + 2),
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
            Some(("self".to_string(), String::new()))
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

        format!("use {}::{{{}}};", path_str, write_list(&items, &fmt))
    }
}
