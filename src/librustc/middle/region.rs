// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

This file actually contains two passes related to regions.  The first
pass builds up the `scope_map`, which describes the parent links in
the region hierarchy.  The second pass infers which types must be
region parameterized.

Most of the documentation on regions can be found in
`middle/typeck/infer/region_inference.rs`

*/


use driver::session::Session;
use middle::ty::{FreeRegion};
use middle::ty;

use std::hashmap::{HashMap, HashSet};
use syntax::codemap::Span;
use syntax::{ast, visit};
use syntax::visit::{Visitor,fn_kind};
use syntax::ast::{P,Block,item,fn_decl,NodeId,Arm,Pat,Stmt,Expr,Local};

/**
The region maps encode information about region relationships.

- `scope_map` maps from:
  - an expression to the expression or block encoding the maximum
    (static) lifetime of a value produced by that expression.  This is
    generally the innermost call, statement, match, or block.
  - a variable or binding id to the block in which that variable is declared.
- `free_region_map` maps from:
  - a free region `a` to a list of free regions `bs` such that
    `a <= b for all b in bs`
  - the free region map is populated during type check as we check
    each function. See the function `relate_free_regions` for
    more information.
- `cleanup_scopes` includes scopes where trans cleanups occur
  - this is intended to reflect the current state of trans, not
    necessarily how I think things ought to work
*/
pub struct RegionMaps {
    priv scope_map: HashMap<ast::NodeId, ast::NodeId>,
    priv free_region_map: HashMap<FreeRegion, ~[FreeRegion]>,
    priv cleanup_scopes: HashSet<ast::NodeId>
}

#[deriving(Clone)]
pub struct Context {
    // Scope where variables should be parented to
    var_parent: Option<ast::NodeId>,

    // Innermost enclosing expression
    parent: Option<ast::NodeId>,
}

struct RegionResolutionVisitor {
    sess: Session,

    // Generated maps:
    region_maps: @mut RegionMaps,
}


impl RegionMaps {
    pub fn relate_free_regions(&mut self, sub: FreeRegion, sup: FreeRegion) {
        match self.free_region_map.find_mut(&sub) {
            Some(sups) => {
                if !sups.iter().any(|x| x == &sup) {
                    sups.push(sup);
                }
                return;
            }
            None => {}
        }

        debug!("relate_free_regions(sub={:?}, sup={:?})", sub, sup);

        self.free_region_map.insert(sub, ~[sup]);
    }

    pub fn record_parent(&mut self, sub: ast::NodeId, sup: ast::NodeId) {
        debug!("record_parent(sub={:?}, sup={:?})", sub, sup);
        assert!(sub != sup);

        self.scope_map.insert(sub, sup);
    }

    pub fn record_cleanup_scope(&mut self, scope_id: ast::NodeId) {
        //! Records that a scope is a CLEANUP SCOPE.  This is invoked
        //! from within regionck.  We wait until regionck because we do
        //! not know which operators are overloaded until that point,
        //! and only overloaded operators result in cleanup scopes.

        self.cleanup_scopes.insert(scope_id);
    }

    pub fn opt_encl_scope(&self, id: ast::NodeId) -> Option<ast::NodeId> {
        //! Returns the narrowest scope that encloses `id`, if any.

        self.scope_map.find(&id).map(|x| *x)
    }

    pub fn encl_scope(&self, id: ast::NodeId) -> ast::NodeId {
        //! Returns the narrowest scope that encloses `id`, if any.

        match self.scope_map.find(&id) {
            Some(&r) => r,
            None => { fail!("No enclosing scope for id {:?}", id); }
        }
    }

    pub fn is_cleanup_scope(&self, scope_id: ast::NodeId) -> bool {
        self.cleanup_scopes.contains(&scope_id)
    }

    pub fn cleanup_scope(&self, expr_id: ast::NodeId) -> ast::NodeId {
        //! Returns the scope when temps in expr will be cleaned up

        let mut id = self.encl_scope(expr_id);
        while !self.cleanup_scopes.contains(&id) {
            id = self.encl_scope(id);
        }
        return id;
    }

    pub fn encl_region(&self, id: ast::NodeId) -> ty::Region {
        //! Returns the narrowest scope region that encloses `id`, if any.

        ty::ReScope(self.encl_scope(id))
    }

    pub fn scopes_intersect(&self, scope1: ast::NodeId, scope2: ast::NodeId)
                            -> bool {
        self.is_subscope_of(scope1, scope2) ||
        self.is_subscope_of(scope2, scope1)
    }

    pub fn is_subscope_of(&self,
                          subscope: ast::NodeId,
                          superscope: ast::NodeId)
                          -> bool {
        /*!
         * Returns true if `subscope` is equal to or is lexically
         * nested inside `superscope` and false otherwise.
         */

        let mut s = subscope;
        while superscope != s {
            match self.scope_map.find(&s) {
                None => {
                    debug!("is_subscope_of({:?}, {:?}, s={:?})=false",
                           subscope, superscope, s);

                    return false;
                }
                Some(&scope) => s = scope
            }
        }

        debug!("is_subscope_of({:?}, {:?})=true",
               subscope, superscope);

        return true;
    }

    pub fn sub_free_region(&self, sub: FreeRegion, sup: FreeRegion) -> bool {
        /*!
         * Determines whether two free regions have a subregion relationship
         * by walking the graph encoded in `free_region_map`.  Note that
         * it is possible that `sub != sup` and `sub <= sup` and `sup <= sub`
         * (that is, the user can give two different names to the same lifetime).
         */

        if sub == sup {
            return true;
        }

        // Do a little breadth-first-search here.  The `queue` list
        // doubles as a way to detect if we've seen a particular FR
        // before.  Note that we expect this graph to be an *extremely
        // shallow* tree.
        let mut queue = ~[sub];
        let mut i = 0;
        while i < queue.len() {
            match self.free_region_map.find(&queue[i]) {
                Some(parents) => {
                    for parent in parents.iter() {
                        if *parent == sup {
                            return true;
                        }

                        if !queue.iter().any(|x| x == parent) {
                            queue.push(*parent);
                        }
                    }
                }
                None => {}
            }
            i += 1;
        }
        return false;
    }

    pub fn is_subregion_of(&self,
                           sub_region: ty::Region,
                           super_region: ty::Region)
                           -> bool {
        /*!
         * Determines whether one region is a subregion of another.  This is
         * intended to run *after inference* and sadly the logic is somewhat
         * duplicated with the code in infer.rs.
         */

        debug!("is_subregion_of(sub_region={:?}, super_region={:?})",
               sub_region, super_region);

        sub_region == super_region || {
            match (sub_region, super_region) {
                (_, ty::ReStatic) => {
                    true
                }

                (ty::ReScope(sub_scope), ty::ReScope(super_scope)) => {
                    self.is_subscope_of(sub_scope, super_scope)
                }

                (ty::ReScope(sub_scope), ty::ReFree(ref fr)) => {
                    self.is_subscope_of(sub_scope, fr.scope_id)
                }

                (ty::ReFree(sub_fr), ty::ReFree(super_fr)) => {
                    self.sub_free_region(sub_fr, super_fr)
                }

                _ => {
                    false
                }
            }
        }
    }

    pub fn nearest_common_ancestor(&self,
                                   scope_a: ast::NodeId,
                                   scope_b: ast::NodeId)
                                   -> Option<ast::NodeId> {
        /*!
         * Finds the nearest common ancestor (if any) of two scopes.  That
         * is, finds the smallest scope which is greater than or equal to
         * both `scope_a` and `scope_b`.
         */

        if scope_a == scope_b { return Some(scope_a); }

        let a_ancestors = ancestors_of(self, scope_a);
        let b_ancestors = ancestors_of(self, scope_b);
        let mut a_index = a_ancestors.len() - 1u;
        let mut b_index = b_ancestors.len() - 1u;

        // Here, ~[ab]_ancestors is a vector going from narrow to broad.
        // The end of each vector will be the item where the scope is
        // defined; if there are any common ancestors, then the tails of
        // the vector will be the same.  So basically we want to walk
        // backwards from the tail of each vector and find the first point
        // where they diverge.  If one vector is a suffix of the other,
        // then the corresponding scope is a superscope of the other.

        if a_ancestors[a_index] != b_ancestors[b_index] {
            return None;
        }

        loop {
            // Loop invariant: a_ancestors[a_index] == b_ancestors[b_index]
            // for all indices between a_index and the end of the array
            if a_index == 0u { return Some(scope_a); }
            if b_index == 0u { return Some(scope_b); }
            a_index -= 1u;
            b_index -= 1u;
            if a_ancestors[a_index] != b_ancestors[b_index] {
                return Some(a_ancestors[a_index + 1u]);
            }
        }

        fn ancestors_of(this: &RegionMaps, scope: ast::NodeId)
            -> ~[ast::NodeId]
        {
            // debug!("ancestors_of(scope={})", scope);
            let mut result = ~[scope];
            let mut scope = scope;
            loop {
                match this.scope_map.find(&scope) {
                    None => return result,
                    Some(&superscope) => {
                        result.push(superscope);
                        scope = superscope;
                    }
                }
                // debug!("ancestors_of_loop(scope={})", scope);
            }
        }
    }
}

/// Records the current parent (if any) as the parent of `child_id`.
fn parent_to_expr(visitor: &mut RegionResolutionVisitor,
                  cx: Context, child_id: ast::NodeId, sp: Span) {
    debug!("region::parent_to_expr(span={:?})",
           visitor.sess.codemap.span_to_str(sp));
    for parent_id in cx.parent.iter() {
        visitor.region_maps.record_parent(child_id, *parent_id);
    }
}

fn resolve_block(visitor: &mut RegionResolutionVisitor,
                 blk: ast::P<ast::Block>,
                 cx: Context) {
    // Record the parent of this block.
    parent_to_expr(visitor, cx, blk.id, blk.span);

    // Descend.
    let new_cx = Context {var_parent: Some(blk.id),
                          parent: Some(blk.id)};
    visit::walk_block(visitor, blk, new_cx);
}

fn resolve_arm(visitor: &mut RegionResolutionVisitor,
               arm: &ast::Arm,
               cx: Context) {
    visit::walk_arm(visitor, arm, cx);
}

fn resolve_pat(visitor: &mut RegionResolutionVisitor,
               pat: &ast::Pat,
               cx: Context) {
    assert_eq!(cx.var_parent, cx.parent);
    parent_to_expr(visitor, cx, pat.id, pat.span);
    visit::walk_pat(visitor, pat, cx);
}

fn resolve_stmt(visitor: &mut RegionResolutionVisitor,
                stmt: @ast::Stmt,
                cx: Context) {
    match stmt.node {
        ast::StmtDecl(..) => {
            visit::walk_stmt(visitor, stmt, cx);
        }
        ast::StmtExpr(_, stmt_id) |
        ast::StmtSemi(_, stmt_id) => {
            parent_to_expr(visitor, cx, stmt_id, stmt.span);
            let expr_cx = Context {parent: Some(stmt_id), ..cx};
            visit::walk_stmt(visitor, stmt, expr_cx);
        }
        ast::StmtMac(..) => visitor.sess.bug("unexpanded macro")
    }
}

fn resolve_expr(visitor: &mut RegionResolutionVisitor,
                expr: @ast::Expr,
                cx: Context) {
    parent_to_expr(visitor, cx, expr.id, expr.span);

    let mut new_cx = cx;
    new_cx.parent = Some(expr.id);
    match expr.node {
        ast::ExprAssignOp(..) | ast::ExprIndex(..) | ast::ExprBinary(..) |
        ast::ExprUnary(..) | ast::ExprCall(..) | ast::ExprMethodCall(..) => {
            // FIXME(#6268) Nested method calls
            //
            // The lifetimes for a call or method call look as follows:
            //
            // call.id
            // - arg0.id
            // - ...
            // - argN.id
            // - call.callee_id
            //
            // The idea is that call.callee_id represents *the time when
            // the invoked function is actually running* and call.id
            // represents *the time to prepare the arguments and make the
            // call*.  See the section "Borrows in Calls" borrowck/doc.rs
            // for an extended explanantion of why this distinction is
            // important.
            //
            // parent_to_expr(new_cx, expr.callee_id);
        }

        ast::ExprMatch(..) => {
            new_cx.var_parent = Some(expr.id);
        }

        _ => {}
    };


    visit::walk_expr(visitor, expr, new_cx);
}

fn resolve_local(visitor: &mut RegionResolutionVisitor,
                 local: @ast::Local,
                 cx: Context) {
    assert_eq!(cx.var_parent, cx.parent);
    parent_to_expr(visitor, cx, local.id, local.span);
    visit::walk_local(visitor, local, cx);
}

fn resolve_item(visitor: &mut RegionResolutionVisitor,
                item: @ast::item,
                cx: Context) {
    // Items create a new outer block scope as far as we're concerned.
    let new_cx = Context {var_parent: None, parent: None, ..cx};
    visit::walk_item(visitor, item, new_cx);
}

fn resolve_fn(visitor: &mut RegionResolutionVisitor,
              fk: &visit::fn_kind,
              decl: &ast::fn_decl,
              body: ast::P<ast::Block>,
              sp: Span,
              id: ast::NodeId,
              cx: Context) {
    debug!("region::resolve_fn(id={:?}, \
                               span={:?}, \
                               body.id={:?}, \
                               cx.parent={:?})",
           id,
           visitor.sess.codemap.span_to_str(sp),
           body.id,
           cx.parent);

    // The arguments and `self` are parented to the body of the fn.
    let decl_cx = Context {parent: Some(body.id),
                           var_parent: Some(body.id),
                           ..cx};
    match *fk {
        visit::fk_method(_, _, method) => {
            visitor.region_maps.record_parent(method.self_id, body.id);
        }
        _ => {}
    }
    visit::walk_fn_decl(visitor, decl, decl_cx);

    // The body of the fn itself is either a root scope (top-level fn)
    // or it continues with the inherited scope (closures).
    let body_cx = match *fk {
        visit::fk_item_fn(..) |
        visit::fk_method(..) => {
            Context {parent: None, var_parent: None, ..cx}
        }
        visit::fk_anon(..) |
        visit::fk_fn_block(..) => {
            cx
        }
    };
    visitor.visit_block(body, body_cx);
}

impl Visitor<Context> for RegionResolutionVisitor {

    fn visit_block(&mut self, b:P<Block>, cx:Context) {
        resolve_block(self, b, cx);
    }

    fn visit_item(&mut self, i:@item, cx:Context) {
        resolve_item(self, i, cx);
    }

    fn visit_fn(&mut self, fk:&fn_kind, fd:&fn_decl, b:P<Block>, s:Span, n:NodeId, cx:Context) {
        resolve_fn(self, fk, fd, b, s, n, cx);
    }
    fn visit_arm(&mut self, a:&Arm, cx:Context) {
        resolve_arm(self, a, cx);
    }
    fn visit_pat(&mut self, p:&Pat, cx:Context) {
        resolve_pat(self, p, cx);
    }
    fn visit_stmt(&mut self, s:@Stmt, cx:Context) {
        resolve_stmt(self, s, cx);
    }
    fn visit_expr(&mut self, ex:@Expr, cx:Context) {
        resolve_expr(self, ex, cx);
    }
    fn visit_local(&mut self, l:@Local, cx:Context) {
        resolve_local(self, l, cx);
    }
}

pub fn resolve_crate(sess: Session,
                     crate: &ast::Crate) -> @mut RegionMaps
{
    let region_maps = @mut RegionMaps {
        scope_map: HashMap::new(),
        free_region_map: HashMap::new(),
        cleanup_scopes: HashSet::new(),
    };
    let cx = Context {parent: None,
                      var_parent: None};
    let mut visitor = RegionResolutionVisitor {
        sess: sess,
        region_maps: region_maps,
    };
    visit::walk_crate(&mut visitor, crate, cx);
    return region_maps;
}

