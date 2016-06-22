// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use Resolver;
use rustc::session::Session;
use syntax::ast;
use syntax::fold::Folder;
use syntax::ptr::P;
use syntax::util::move_map::MoveMap;

impl<'a> Resolver<'a> {
    pub fn assign_node_ids(&mut self, krate: ast::Crate) -> ast::Crate {
        NodeIdAssigner {
            sess: self.session,
        }.fold_crate(krate)
    }
}

struct NodeIdAssigner<'a> {
    sess: &'a Session,
}

impl<'a> Folder for NodeIdAssigner<'a> {
    fn new_id(&mut self, old_id: ast::NodeId) -> ast::NodeId {
        assert_eq!(old_id, ast::DUMMY_NODE_ID);
        self.sess.next_node_id()
    }

    fn fold_block(&mut self, block: P<ast::Block>) -> P<ast::Block> {
        block.map(|mut block| {
            block.id = self.new_id(block.id);

            let stmt = block.stmts.pop();
            block.stmts = block.stmts.move_flat_map(|s| self.fold_stmt(s).into_iter());
            if let Some(ast::Stmt { node: ast::StmtKind::Expr(expr), span, .. }) = stmt {
                // Avoid wasting a node id on a trailing expression statement,
                // which shares a HIR node with the expression itself.
                let expr = self.fold_expr(expr);
                block.stmts.push(ast::Stmt {
                    id: expr.id,
                    node: ast::StmtKind::Expr(expr),
                    span: span,
                });
            } else if let Some(stmt) = stmt {
                block.stmts.extend(self.fold_stmt(stmt));
            }

            block
        })
    }
}

