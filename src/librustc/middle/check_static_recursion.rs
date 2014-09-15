// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This compiler pass detects static items that refer to themselves
// recursively.

use driver::session::Session;
use middle::resolve;
use middle::def::DefStatic;

use syntax::ast::{Crate, Expr, ExprPath, Item, ItemStatic, NodeId};
use syntax::{ast_util, ast_map};
use syntax::visit::Visitor;
use syntax::visit;

struct CheckCrateVisitor<'a, 'ast: 'a> {
    sess: &'a Session,
    def_map: &'a resolve::DefMap,
    ast_map: &'a ast_map::Map<'ast>
}

impl<'v, 'a, 'ast> Visitor<'v> for CheckCrateVisitor<'a, 'ast> {
    fn visit_item(&mut self, i: &Item) {
        check_item(self, i);
    }
}

pub fn check_crate<'ast>(sess: &Session,
                         krate: &Crate,
                         def_map: &resolve::DefMap,
                         ast_map: &ast_map::Map<'ast>) {
    let mut visitor = CheckCrateVisitor {
        sess: sess,
        def_map: def_map,
        ast_map: ast_map
    };
    visit::walk_crate(&mut visitor, krate);
    sess.abort_if_errors();
}

fn check_item(v: &mut CheckCrateVisitor, it: &Item) {
    match it.node {
        ItemStatic(_, _, ref ex) => {
            check_item_recursion(v.sess, v.ast_map, v.def_map, it);
            visit::walk_expr(v, &**ex)
        },
        _ => visit::walk_item(v, it)
    }
}

struct CheckItemRecursionVisitor<'a, 'ast: 'a> {
    root_it: &'a Item,
    sess: &'a Session,
    ast_map: &'a ast_map::Map<'ast>,
    def_map: &'a resolve::DefMap,
    idstack: Vec<NodeId>
}

// Make sure a const item doesn't recursively refer to itself
// FIXME: Should use the dependency graph when it's available (#1356)
pub fn check_item_recursion<'a>(sess: &'a Session,
                                ast_map: &'a ast_map::Map,
                                def_map: &'a resolve::DefMap,
                                it: &'a Item) {

    let mut visitor = CheckItemRecursionVisitor {
        root_it: it,
        sess: sess,
        ast_map: ast_map,
        def_map: def_map,
        idstack: Vec::new()
    };
    visitor.visit_item(it);
}

impl<'a, 'ast, 'v> Visitor<'v> for CheckItemRecursionVisitor<'a, 'ast> {
    fn visit_item(&mut self, it: &Item) {
        if self.idstack.iter().any(|x| x == &(it.id)) {
            self.sess.span_err(self.root_it.span, "recursive constant");
            return;
        }
        self.idstack.push(it.id);
        visit::walk_item(self, it);
        self.idstack.pop();
    }

    fn visit_expr(&mut self, e: &Expr) {
        match e.node {
            ExprPath(..) => {
                match self.def_map.borrow().find(&e.id) {
                    Some(&DefStatic(def_id, _)) if
                            ast_util::is_local(def_id) => {
                        self.visit_item(&*self.ast_map.expect_item(def_id.node));
                    }
                    _ => ()
                }
            },
            _ => ()
        }
        visit::walk_expr(self, e);
    }
}
