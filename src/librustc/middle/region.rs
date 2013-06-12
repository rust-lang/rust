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

*/

use core::prelude::*;

use driver::session::Session;
use metadata::csearch;
use middle::resolve;
use middle::ty::{region_variance, rv_covariant, rv_invariant};
use middle::ty::{rv_contravariant, FreeRegion};
use middle::ty;

use core::iterator::IteratorUtil;
use core::hashmap::{HashMap, HashSet};
use syntax::ast_map;
use syntax::codemap::span;
use syntax::print::pprust;
use syntax::parse::token;
use syntax::parse::token::special_idents;
use syntax::{ast, visit};

pub type parent = Option<ast::node_id>;

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
    priv scope_map: HashMap<ast::node_id, ast::node_id>,
    priv free_region_map: HashMap<FreeRegion, ~[FreeRegion]>,
    priv cleanup_scopes: HashSet<ast::node_id>
}

pub struct Context {
    sess: Session,
    def_map: resolve::DefMap,

    // Generated maps:
    region_maps: @mut RegionMaps,

    // Scope where variables should be parented to
    var_parent: parent,

    // Innermost enclosing expression
    parent: parent,
}

impl RegionMaps {
    pub fn relate_free_regions(&mut self, sub: FreeRegion, sup: FreeRegion) {
        match self.free_region_map.find_mut(&sub) {
            Some(sups) => {
                if !sups.contains(&sup) {
                    sups.push(sup);
                }
                return;
            }
            None => {}
        }

        debug!("relate_free_regions(sub=%?, sup=%?)", sub, sup);

        self.free_region_map.insert(sub, ~[sup]);
    }

    pub fn record_parent(&mut self, sub: ast::node_id, sup: ast::node_id) {
        debug!("record_parent(sub=%?, sup=%?)", sub, sup);
        assert!(sub != sup);

        self.scope_map.insert(sub, sup);
    }

    pub fn record_cleanup_scope(&mut self, scope_id: ast::node_id) {
        //! Records that a scope is a CLEANUP SCOPE.  This is invoked
        //! from within regionck.  We wait until regionck because we do
        //! not know which operators are overloaded until that point,
        //! and only overloaded operators result in cleanup scopes.

        self.cleanup_scopes.insert(scope_id);
    }

    pub fn opt_encl_scope(&self, id: ast::node_id) -> Option<ast::node_id> {
        //! Returns the narrowest scope that encloses `id`, if any.

        self.scope_map.find(&id).map(|&x| *x)
    }

    pub fn encl_scope(&self, id: ast::node_id) -> ast::node_id {
        //! Returns the narrowest scope that encloses `id`, if any.

        match self.scope_map.find(&id) {
            Some(&r) => r,
            None => { fail!("No enclosing scope for id %?", id); }
        }
    }

    pub fn is_cleanup_scope(&self, scope_id: ast::node_id) -> bool {
        self.cleanup_scopes.contains(&scope_id)
    }

    pub fn cleanup_scope(&self, expr_id: ast::node_id) -> ast::node_id {
        //! Returns the scope when temps in expr will be cleaned up

        let mut id = self.encl_scope(expr_id);
        while !self.cleanup_scopes.contains(&id) {
            id = self.encl_scope(id);
        }
        return id;
    }

    pub fn encl_region(&self, id: ast::node_id) -> ty::Region {
        //! Returns the narrowest scope region that encloses `id`, if any.

        ty::re_scope(self.encl_scope(id))
    }

    pub fn scopes_intersect(&self, scope1: ast::node_id, scope2: ast::node_id)
                            -> bool {
        self.is_subscope_of(scope1, scope2) ||
        self.is_subscope_of(scope2, scope1)
    }

    pub fn is_subscope_of(&self,
                          subscope: ast::node_id,
                          superscope: ast::node_id)
                          -> bool {
        /*!
         * Returns true if `subscope` is equal to or is lexically
         * nested inside `superscope` and false otherwise.
         */

        let mut s = subscope;
        while superscope != s {
            match self.scope_map.find(&s) {
                None => {
                    debug!("is_subscope_of(%?, %?, s=%?)=false",
                           subscope, superscope, s);

                    return false;
                }
                Some(&scope) => s = scope
            }
        }

        debug!("is_subscope_of(%?, %?)=true",
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
                    for parents.each |parent| {
                        if *parent == sup {
                            return true;
                        }

                        if !queue.contains(parent) {
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

        debug!("is_subregion_of(sub_region=%?, super_region=%?)",
               sub_region, super_region);

        sub_region == super_region || {
            match (sub_region, super_region) {
                (_, ty::re_static) => {
                    true
                }

                (ty::re_scope(sub_scope), ty::re_scope(super_scope)) => {
                    self.is_subscope_of(sub_scope, super_scope)
                }

                (ty::re_scope(sub_scope), ty::re_free(ref fr)) => {
                    self.is_subscope_of(sub_scope, fr.scope_id)
                }

                (ty::re_free(sub_fr), ty::re_free(super_fr)) => {
                    self.sub_free_region(sub_fr, super_fr)
                }

                _ => {
                    false
                }
            }
        }
    }

    pub fn nearest_common_ancestor(&self,
                                   scope_a: ast::node_id,
                                   scope_b: ast::node_id)
                                   -> Option<ast::node_id> {
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

        fn ancestors_of(this: &RegionMaps, scope: ast::node_id)
            -> ~[ast::node_id]
        {
            // debug!("ancestors_of(scope=%d)", scope);
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
                // debug!("ancestors_of_loop(scope=%d)", scope);
            }
        }
    }
}

/// Extracts that current parent from cx, failing if there is none.
pub fn parent_id(cx: Context, span: span) -> ast::node_id {
    match cx.parent {
      None => {
        cx.sess.span_bug(span, "crate should not be parent here");
      }
      Some(parent_id) => {
        parent_id
      }
    }
}

/// Records the current parent (if any) as the parent of `child_id`.
pub fn parent_to_expr(cx: Context, child_id: ast::node_id, sp: span) {
    debug!("region::parent_to_expr(span=%?)",
           cx.sess.codemap.span_to_str(sp));
    for cx.parent.iter().advance |parent_id| {
        cx.region_maps.record_parent(child_id, *parent_id);
    }
}

pub fn resolve_block(blk: &ast::blk, (cx, visitor): (Context, visit::vt<Context>)) {
    // Record the parent of this block.
    parent_to_expr(cx, blk.node.id, blk.span);

    // Descend.
    let new_cx = Context {var_parent: Some(blk.node.id),
                          parent: Some(blk.node.id),
                          ..cx};
    visit::visit_block(blk, (new_cx, visitor));
}

pub fn resolve_arm(arm: &ast::arm, (cx, visitor): (Context, visit::vt<Context>)) {
    visit::visit_arm(arm, (cx, visitor));
}

pub fn resolve_pat(pat: @ast::pat, (cx, visitor): (Context, visit::vt<Context>)) {
    assert_eq!(cx.var_parent, cx.parent);
    parent_to_expr(cx, pat.id, pat.span);
    visit::visit_pat(pat, (cx, visitor));
}

pub fn resolve_stmt(stmt: @ast::stmt, (cx, visitor): (Context, visit::vt<Context>)) {
    match stmt.node {
        ast::stmt_decl(*) => {
            visit::visit_stmt(stmt, (cx, visitor));
        }
        ast::stmt_expr(_, stmt_id) |
        ast::stmt_semi(_, stmt_id) => {
            parent_to_expr(cx, stmt_id, stmt.span);
            let expr_cx = Context {parent: Some(stmt_id), ..cx};
            visit::visit_stmt(stmt, (expr_cx, visitor));
        }
        ast::stmt_mac(*) => cx.sess.bug("unexpanded macro")
    }
}

pub fn resolve_expr(expr: @ast::expr, (cx, visitor): (Context, visit::vt<Context>)) {
    parent_to_expr(cx, expr.id, expr.span);

    let mut new_cx = cx;
    new_cx.parent = Some(expr.id);
    match expr.node {
        ast::expr_assign_op(*) | ast::expr_index(*) | ast::expr_binary(*) |
        ast::expr_unary(*) | ast::expr_call(*) | ast::expr_method_call(*) => {
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

        ast::expr_match(*) => {
            new_cx.var_parent = Some(expr.id);
        }

        _ => {}
    };


    visit::visit_expr(expr, (new_cx, visitor));
}

pub fn resolve_local(local: @ast::local,
                     (cx, visitor) : (Context,
                                      visit::vt<Context>)) {
    assert_eq!(cx.var_parent, cx.parent);
    parent_to_expr(cx, local.node.id, local.span);
    visit::visit_local(local, (cx, visitor));
}

pub fn resolve_item(item: @ast::item, (cx, visitor): (Context, visit::vt<Context>)) {
    // Items create a new outer block scope as far as we're concerned.
    let new_cx = Context {var_parent: None, parent: None, ..cx};
    visit::visit_item(item, (new_cx, visitor));
}

pub fn resolve_fn(fk: &visit::fn_kind,
                  decl: &ast::fn_decl,
                  body: &ast::blk,
                  sp: span,
                  id: ast::node_id,
                  (cx, visitor): (Context,
                                  visit::vt<Context>)) {
    debug!("region::resolve_fn(id=%?, \
                               span=%?, \
                               body.node.id=%?, \
                               cx.parent=%?)",
           id,
           cx.sess.codemap.span_to_str(sp),
           body.node.id,
           cx.parent);

    // The arguments and `self` are parented to the body of the fn.
    let decl_cx = Context {parent: Some(body.node.id),
                           var_parent: Some(body.node.id),
                           ..cx};
    match *fk {
        visit::fk_method(_, _, method) => {
            cx.region_maps.record_parent(method.self_id, body.node.id);
        }
        _ => {}
    }
    visit::visit_fn_decl(decl, (decl_cx, visitor));

    // The body of the fn itself is either a root scope (top-level fn)
    // or it continues with the inherited scope (closures).
    let body_cx = match *fk {
        visit::fk_item_fn(*) |
        visit::fk_method(*) => {
            Context {parent: None, var_parent: None, ..cx}
        }
        visit::fk_anon(*) |
        visit::fk_fn_block(*) => {
            cx
        }
    };
    (visitor.visit_block)(body, (body_cx, visitor));
}

pub fn resolve_crate(sess: Session,
                     def_map: resolve::DefMap,
                     crate: @ast::crate) -> @mut RegionMaps
{
    let region_maps = @mut RegionMaps {
        scope_map: HashMap::new(),
        free_region_map: HashMap::new(),
        cleanup_scopes: HashSet::new(),
    };
    let cx = Context {sess: sess,
                      def_map: def_map,
                      region_maps: region_maps,
                      parent: None,
                      var_parent: None};
    let visitor = visit::mk_vt(@visit::Visitor {
        visit_block: resolve_block,
        visit_item: resolve_item,
        visit_fn: resolve_fn,
        visit_arm: resolve_arm,
        visit_pat: resolve_pat,
        visit_stmt: resolve_stmt,
        visit_expr: resolve_expr,
        visit_local: resolve_local,
        .. *visit::default_visitor()
    });
    visit::visit_crate(crate, (cx, visitor));
    return region_maps;
}

// ___________________________________________________________________________
// Determining region parameterization
//
// Infers which type defns must be region parameterized---this is done
// by scanning their contents to see whether they reference a region
// type, directly or indirectly.  This is a fixed-point computation.
//
// We do it in two passes.  First we walk the AST and construct a map
// from each type defn T1 to other defns which make use of it.  For example,
// if we have a type like:
//
//    type S = *int;
//    type T = S;
//
// Then there would be a map entry from S to T.  During the same walk,
// we also construct add any types that reference regions to a set and
// a worklist.  We can then process the worklist, propagating indirect
// dependencies until a fixed point is reached.

pub type region_paramd_items = @mut HashMap<ast::node_id, region_variance>;

#[deriving(Eq)]
pub struct region_dep {
    ambient_variance: region_variance,
    id: ast::node_id
}

pub type dep_map = @mut HashMap<ast::node_id, @mut ~[region_dep]>;

pub struct DetermineRpCtxt {
    sess: Session,
    ast_map: ast_map::map,
    def_map: resolve::DefMap,
    region_paramd_items: region_paramd_items,
    dep_map: dep_map,
    worklist: ~[ast::node_id],

    // the innermost enclosing item id
    item_id: ast::node_id,

    // true when we are within an item but not within a method.
    // see long discussion on region_is_relevant().
    anon_implies_rp: bool,

    // encodes the context of the current type; invariant if
    // mutable, covariant otherwise
    ambient_variance: region_variance,
}

pub fn join_variance(variance1: region_variance,
                     variance2: region_variance)
                  -> region_variance {
    match (variance1, variance2) {
      (rv_invariant, _) => {rv_invariant}
      (_, rv_invariant) => {rv_invariant}
      (rv_covariant, rv_contravariant) => {rv_invariant}
      (rv_contravariant, rv_covariant) => {rv_invariant}
      (rv_covariant, rv_covariant) => {rv_covariant}
      (rv_contravariant, rv_contravariant) => {rv_contravariant}
    }
}

/// Combines the ambient variance with the variance of a
/// particular site to yield the final variance of the reference.
///
/// Example: if we are checking function arguments then the ambient
/// variance is contravariant.  If we then find a `&'r T` pointer, `r`
/// appears in a co-variant position.  This implies that this
/// occurrence of `r` is contra-variant with respect to the current
/// item, and hence the function returns `rv_contravariant`.
pub fn add_variance(ambient_variance: region_variance,
                    variance: region_variance)
                 -> region_variance {
    match (ambient_variance, variance) {
      (rv_invariant, _) => rv_invariant,
      (_, rv_invariant) => rv_invariant,
      (rv_covariant, c) => c,
      (c, rv_covariant) => c,
      (rv_contravariant, rv_contravariant) => rv_covariant
    }
}

impl DetermineRpCtxt {
    pub fn add_variance(&self, variance: region_variance) -> region_variance {
        add_variance(self.ambient_variance, variance)
    }

    /// Records that item `id` is region-parameterized with the
    /// variance `variance`.  If `id` was already parameterized, then
    /// the new variance is joined with the old variance.
    pub fn add_rp(&mut self, id: ast::node_id, variance: region_variance) {
        assert!(id != 0);
        let old_variance = self.region_paramd_items.find(&id).
                                map_consume(|x| *x);
        let joined_variance = match old_variance {
          None => variance,
          Some(v) => join_variance(v, variance)
        };

        debug!("add_rp() variance for %s: %? == %? ^ %?",
               ast_map::node_id_to_str(self.ast_map, id,
                                       token::get_ident_interner()),
               joined_variance, old_variance, variance);

        if Some(joined_variance) != old_variance {
            let region_paramd_items = self.region_paramd_items;
            region_paramd_items.insert(id, joined_variance);
            self.worklist.push(id);
        }
    }

    /// Indicates that the region-parameterization of the current item
    /// is dependent on the region-parameterization of the item
    /// `from`.  Put another way, it indicates that the current item
    /// contains a value of type `from`, so if `from` is
    /// region-parameterized, so is the current item.
    pub fn add_dep(&mut self, from: ast::node_id) {
        debug!("add dependency from %d -> %d (%s -> %s) with variance %?",
               from, self.item_id,
               ast_map::node_id_to_str(self.ast_map, from,
                                       token::get_ident_interner()),
               ast_map::node_id_to_str(self.ast_map, self.item_id,
                                       token::get_ident_interner()),
               copy self.ambient_variance);
        let vec = match self.dep_map.find(&from) {
            Some(&vec) => vec,
            None => {
                let vec = @mut ~[];
                let dep_map = self.dep_map;
                dep_map.insert(from, vec);
                vec
            }
        };
        let dep = region_dep {
            ambient_variance: self.ambient_variance,
            id: self.item_id
        };
        if !vec.contains(&dep) { vec.push(dep); }
    }

    // Determines whether a reference to a region that appears in the
    // AST implies that the enclosing type is region-parameterized (RP).
    // This point is subtle.  Here are some examples to make it more
    // concrete.
    //
    // 1. impl foo for &int { ... }
    // 2. impl foo for &'self int { ... }
    // 3. impl foo for bar { fn m(@self) -> &'self int { ... } }
    // 4. impl foo for bar { fn m(&self) -> &'self int { ... } }
    // 5. impl foo for bar { fn m(&self) -> &int { ... } }
    //
    // In case 1, the anonymous region is being referenced,
    // but it appears in a context where the anonymous region
    // resolves to self, so the impl foo is RP.
    //
    // In case 2, the self parameter is written explicitly.
    //
    // In case 3, the method refers to the region `self`, so that
    // implies that the impl must be region parameterized.  (If the
    // type bar is not region parameterized, that is an error, because
    // the self region is effectively unconstrained, but that is
    // detected elsewhere).
    //
    // In case 4, the method refers to the region `self`, but the
    // `self` region is bound by the `&self` receiver, and so this
    // does not require that `bar` be RP.
    //
    // In case 5, the anonymous region is referenced, but it
    // bound by the method, so it does not refer to self.  This impl
    // need not be region parameterized.
    //
    // Normally, & or &self implies that the enclosing item is RP.
    // However, within a function, & is always bound.  Within a method
    // with &self type, &self is also bound.  We detect those last two
    // cases via flags (anon_implies_rp and self_implies_rp) that are
    // true when the anon or self region implies RP.
    pub fn region_is_relevant(&self, r: Option<@ast::Lifetime>) -> bool {
        match r {
            None => {
                self.anon_implies_rp
            }
            Some(ref l) if l.ident == special_idents::statik => {
                false
            }
            Some(ref l) if l.ident == special_idents::self_ => {
                true
            }
            Some(_) => {
                false
            }
        }
    }

    pub fn with(@mut self,
                item_id: ast::node_id,
                anon_implies_rp: bool,
                f: &fn()) {
        let old_item_id = self.item_id;
        let old_anon_implies_rp = self.anon_implies_rp;
        self.item_id = item_id;
        self.anon_implies_rp = anon_implies_rp;
        debug!("with_item_id(%d, %b)",
               item_id,
               anon_implies_rp);
        let _i = ::util::common::indenter();
        f();
        self.item_id = old_item_id;
        self.anon_implies_rp = old_anon_implies_rp;
    }

    pub fn with_ambient_variance(@mut self,
                                 variance: region_variance,
                                 f: &fn()) {
        let old_ambient_variance = self.ambient_variance;
        self.ambient_variance = self.add_variance(variance);
        f();
        self.ambient_variance = old_ambient_variance;
    }
}

pub fn determine_rp_in_item(item: @ast::item,
                            (cx, visitor): (@mut DetermineRpCtxt,
                                            visit::vt<@mut DetermineRpCtxt>)) {
    do cx.with(item.id, true) {
        visit::visit_item(item, (cx, visitor));
    }
}

pub fn determine_rp_in_fn(fk: &visit::fn_kind,
                          decl: &ast::fn_decl,
                          body: &ast::blk,
                          _: span,
                          _: ast::node_id,
                          (cx, visitor): (@mut DetermineRpCtxt,
                                          visit::vt<@mut DetermineRpCtxt>)) {
    do cx.with(cx.item_id, false) {
        do cx.with_ambient_variance(rv_contravariant) {
            for decl.inputs.each |a| {
                (visitor.visit_ty)(a.ty, (cx, visitor));
            }
        }
        (visitor.visit_ty)(decl.output, (cx, visitor));
        let generics = visit::generics_of_fn(fk);
        (visitor.visit_generics)(&generics, (cx, visitor));
        (visitor.visit_block)(body, (cx, visitor));
    }
}

pub fn determine_rp_in_ty_method(ty_m: &ast::ty_method,
                                 (cx, visitor): (@mut DetermineRpCtxt,
                                                 visit::vt<@mut DetermineRpCtxt>)) {
    do cx.with(cx.item_id, false) {
        visit::visit_ty_method(ty_m, (cx, visitor));
    }
}

pub fn determine_rp_in_ty(ty: @ast::Ty,
                          (cx, visitor): (@mut DetermineRpCtxt,
                                          visit::vt<@mut DetermineRpCtxt>)) {
    // we are only interested in types that will require an item to
    // be region-parameterized.  if cx.item_id is zero, then this type
    // is not a member of a type defn nor is it a constitutent of an
    // impl etc.  So we can ignore it and its components.
    if cx.item_id == 0 { return; }

    // if this type directly references a region pointer like &'r ty,
    // add to the worklist/set.  Note that &'r ty is contravariant with
    // respect to &r, because &'r ty can be used whereever a *smaller*
    // region is expected (and hence is a supertype of those
    // locations)
    let sess = cx.sess;
    match ty.node {
        ast::ty_rptr(r, _) => {
            debug!("referenced rptr type %s",
                   pprust::ty_to_str(ty, sess.intr()));

            if cx.region_is_relevant(r) {
                let rv = cx.add_variance(rv_contravariant);
                cx.add_rp(cx.item_id, rv)
            }
        }

        ast::ty_closure(ref f) => {
            debug!("referenced fn type: %s",
                   pprust::ty_to_str(ty, sess.intr()));
            match f.region {
                Some(_) => {
                    if cx.region_is_relevant(f.region) {
                        let rv = cx.add_variance(rv_contravariant);
                        cx.add_rp(cx.item_id, rv)
                    }
                }
                None => {
                    if f.sigil == ast::BorrowedSigil && cx.anon_implies_rp {
                        let rv = cx.add_variance(rv_contravariant);
                        cx.add_rp(cx.item_id, rv)
                    }
                }
            }
        }

        _ => {}
    }

    // if this references another named type, add the dependency
    // to the dep_map.  If the type is not defined in this crate,
    // then check whether it is region-parameterized and consider
    // that as a direct dependency.
    match ty.node {
      ast::ty_path(path, id) => {
        match cx.def_map.find(&id) {
          Some(&ast::def_ty(did)) |
          Some(&ast::def_trait(did)) |
          Some(&ast::def_struct(did)) => {
            if did.crate == ast::local_crate {
                if cx.region_is_relevant(path.rp) {
                    cx.add_dep(did.node);
                }
            } else {
                let cstore = sess.cstore;
                match csearch::get_region_param(cstore, did) {
                  None => {}
                  Some(variance) => {
                    debug!("reference to external, rp'd type %s",
                           pprust::ty_to_str(ty, sess.intr()));
                    if cx.region_is_relevant(path.rp) {
                        let rv = cx.add_variance(variance);
                        cx.add_rp(cx.item_id, rv)
                    }
                  }
                }
            }
          }
          _ => {}
        }
      }
      _ => {}
    }

    match ty.node {
      ast::ty_box(mt) | ast::ty_uniq(mt) | ast::ty_vec(mt) |
      ast::ty_rptr(_, mt) | ast::ty_ptr(mt) => {
        visit_mt(mt, (cx, visitor));
      }

      ast::ty_path(path, _) => {
        // type parameters are---for now, anyway---always invariant
        do cx.with_ambient_variance(rv_invariant) {
            for path.types.each |tp| {
                (visitor.visit_ty)(*tp, (cx, visitor));
            }
        }
      }

      ast::ty_closure(@ast::TyClosure {decl: ref decl, _}) |
      ast::ty_bare_fn(@ast::TyBareFn {decl: ref decl, _}) => {
        // fn() binds the & region, so do not consider &T types that
        // appear *inside* a fn() type to affect the enclosing item:
        do cx.with(cx.item_id, false) {
            // parameters are contravariant
            do cx.with_ambient_variance(rv_contravariant) {
                for decl.inputs.each |a| {
                    (visitor.visit_ty)(a.ty, (cx, visitor));
                }
            }
            (visitor.visit_ty)(decl.output, (cx, visitor));
        }
      }

      _ => {
        visit::visit_ty(ty, (cx, visitor));
      }
    }

    fn visit_mt(mt: ast::mt,
                (cx, visitor): (@mut DetermineRpCtxt,
                                visit::vt<@mut DetermineRpCtxt>)) {
        // mutability is invariant
        if mt.mutbl == ast::m_mutbl {
            do cx.with_ambient_variance(rv_invariant) {
                (visitor.visit_ty)(mt.ty, (cx, visitor));
            }
        } else {
            (visitor.visit_ty)(mt.ty, (cx, visitor));
        }
    }
}

pub fn determine_rp_in_struct_field(
        cm: @ast::struct_field,
        (cx, visitor): (@mut DetermineRpCtxt,
                        visit::vt<@mut DetermineRpCtxt>)) {
    visit::visit_struct_field(cm, (cx, visitor));
}

pub fn determine_rp_in_crate(sess: Session,
                             ast_map: ast_map::map,
                             def_map: resolve::DefMap,
                             crate: @ast::crate)
                          -> region_paramd_items {
    let cx = @mut DetermineRpCtxt {
        sess: sess,
        ast_map: ast_map,
        def_map: def_map,
        region_paramd_items: @mut HashMap::new(),
        dep_map: @mut HashMap::new(),
        worklist: ~[],
        item_id: 0,
        anon_implies_rp: false,
        ambient_variance: rv_covariant
    };

    // Gather up the base set, worklist and dep_map
    let visitor = visit::mk_vt(@visit::Visitor {
        visit_fn: determine_rp_in_fn,
        visit_item: determine_rp_in_item,
        visit_ty: determine_rp_in_ty,
        visit_ty_method: determine_rp_in_ty_method,
        visit_struct_field: determine_rp_in_struct_field,
        .. *visit::default_visitor()
    });
    visit::visit_crate(crate, (cx, visitor));

    // Propagate indirect dependencies
    //
    // Each entry in the worklist is the id of an item C whose region
    // parameterization has been updated.  So we pull ids off of the
    // worklist, find the current variance, and then iterate through
    // all of the dependent items (that is, those items that reference
    // C).  For each dependent item D, we combine the variance of C
    // with the ambient variance where the reference occurred and then
    // update the region-parameterization of D to reflect the result.
    {
        let cx = &mut *cx;
        while cx.worklist.len() != 0 {
            let c_id = cx.worklist.pop();
            let c_variance = cx.region_paramd_items.get_copy(&c_id);
            debug!("popped %d from worklist", c_id);
            match cx.dep_map.find(&c_id) {
              None => {}
              Some(deps) => {
                for deps.each |dep| {
                    let v = add_variance(dep.ambient_variance, c_variance);
                    cx.add_rp(dep.id, v);
                }
              }
            }
        }
    }

    debug!("%s", {
        debug!("Region variance results:");
        let region_paramd_items = cx.region_paramd_items;
        for region_paramd_items.each |&key, &value| {
            debug!("item %? (%s) is parameterized with variance %?",
                   key,
                   ast_map::node_id_to_str(ast_map, key,
                                           token::get_ident_interner()),
                   value);
        }
        "----"
    });

    // return final set
    return cx.region_paramd_items;
}
