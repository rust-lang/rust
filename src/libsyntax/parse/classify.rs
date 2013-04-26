// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
  Predicates on exprs and stmts that the pretty-printer and parser use
 */

use ast;
use codemap;

pub fn expr_requires_semi_to_be_stmt(e: @ast::expr) -> bool {
    match e.node {
      ast::expr_if(*)
      | ast::expr_match(*)
      | ast::expr_block(_)
      | ast::expr_while(*)
      | ast::expr_loop(*)
      | ast::expr_call(_, _, ast::DoSugar)
      | ast::expr_call(_, _, ast::ForSugar)
      | ast::expr_method_call(_, _, _, _, ast::DoSugar)
      | ast::expr_method_call(_, _, _, _, ast::ForSugar) => false,
      _ => true
    }
}

pub fn expr_is_simple_block(e: @ast::expr) -> bool {
    match e.node {
        ast::expr_block(
            codemap::spanned {
                node: ast::blk_ { rules: ast::default_blk, _ }, _ }
        ) => true,
      _ => false
    }
}

pub fn stmt_ends_with_semi(stmt: &ast::stmt) -> bool {
    return match stmt.node {
        ast::stmt_decl(d, _) => {
            match d.node {
                ast::decl_local(_) => true,
                ast::decl_item(_) => false
            }
        }
        ast::stmt_expr(e, _) => { expr_requires_semi_to_be_stmt(e) }
        ast::stmt_semi(*) => { false }
        ast::stmt_mac(*) => { false }
    }
}
