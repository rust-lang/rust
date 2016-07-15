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
use syntax::ext::mtwt;
use syntax::fold::{self, Folder};
use syntax::ptr::P;
use syntax::util::move_map::MoveMap;
use syntax::util::small_vector::SmallVector;

use std::collections::HashMap;
use std::mem;

impl<'a> Resolver<'a> {
    pub fn assign_node_ids(&mut self, krate: ast::Crate) -> ast::Crate {
        NodeIdAssigner {
            sess: self.session,
            macros_at_scope: &mut self.macros_at_scope,
        }.fold_crate(krate)
    }
}

struct NodeIdAssigner<'a> {
    sess: &'a Session,
    macros_at_scope: &'a mut HashMap<ast::NodeId, Vec<ast::Mrk>>,
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
            let mut macros = Vec::new();
            block.stmts = block.stmts.move_flat_map(|stmt| {
                if let ast::StmtKind::Item(ref item) = stmt.node {
                    if let ast::ItemKind::Mac(..) = item.node {
                        macros.push(mtwt::outer_mark(item.ident.ctxt));
                        return None;
                    }
                }

                let stmt = self.fold_stmt(stmt).pop().unwrap();
                if !macros.is_empty() {
                    self.macros_at_scope.insert(stmt.id, mem::replace(&mut macros, Vec::new()));
                }
                Some(stmt)
            });

            stmt.and_then(|mut stmt| {
                // Avoid wasting a node id on a trailing expression statement,
                // which shares a HIR node with the expression itself.
                if let ast::StmtKind::Expr(expr) = stmt.node {
                    let expr = self.fold_expr(expr);
                    stmt.id = expr.id;
                    stmt.node = ast::StmtKind::Expr(expr);
                    Some(stmt)
                } else {
                    self.fold_stmt(stmt).pop()
                }
            }).map(|stmt| {
                if !macros.is_empty() {
                    self.macros_at_scope.insert(stmt.id, mem::replace(&mut macros, Vec::new()));
                }
                block.stmts.push(stmt);
            });

            block
        })
    }

    fn fold_item(&mut self, item: P<ast::Item>) -> SmallVector<P<ast::Item>> {
        match item.node {
            ast::ItemKind::Mac(..) => SmallVector::zero(),
            _ => fold::noop_fold_item(item, self),
        }
    }
}
