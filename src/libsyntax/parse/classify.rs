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

// does this expression require a semicolon to be treated
// as a statement? The negation of this: 'can this expression
// be used as a statement without a semicolon' -- is used
// as an early-bail-out in the parser so that, for instance,
// 'if true {...} else {...}
//  |x| 5 '
// isn't parsed as (if true {...} else {...} | x) | 5
pub fn expr_requires_semi_to_be_stmt(e: @ast::Expr) -> bool {
    match e.node {
      ast::ExprIf(..)
      | ast::ExprMatch(..)
      | ast::ExprBlock(_)
      | ast::ExprWhile(..)
      | ast::ExprLoop(..)
      | ast::ExprForLoop(..)
      | ast::ExprCall(_, _, ast::ForSugar)
      | ast::ExprMethodCall(_, _, _, _, ast::ForSugar) => false,
      _ => true
    }
}

pub fn expr_is_simple_block(e: @ast::Expr) -> bool {
    match e.node {
        ast::ExprBlock(block) => block.rules == ast::DefaultBlock,
      _ => false
    }
}

// this statement requires a semicolon after it.
// note that in one case (stmt_semi), we've already
// seen the semicolon, and thus don't need another.
pub fn stmt_ends_with_semi(stmt: &ast::Stmt) -> bool {
    return match stmt.node {
        ast::StmtDecl(d, _) => {
            match d.node {
                ast::DeclLocal(_) => true,
                ast::DeclItem(_) => false
            }
        }
        ast::StmtExpr(e, _) => { expr_requires_semi_to_be_stmt(e) }
        ast::StmtSemi(..) => { false }
        ast::StmtMac(..) => { false }
    }
}
