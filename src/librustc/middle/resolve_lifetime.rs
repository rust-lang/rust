// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Name resolution for lifetimes.
 *
 * Name resolution for lifetimes follows MUCH simpler rules than the
 * full resolve. For example, lifetime names are never exported or
 * used between functions, and they operate in a purely top-down
 * way. Therefore we break lifetime name resolution into a separate pass.
 */

use driver::session;
use std::hashmap::HashMap;
use syntax::ast;
use syntax::codemap::Span;
use syntax::opt_vec;
use syntax::opt_vec::OptVec;
use syntax::parse::token::special_idents;
use syntax::print::pprust::{lifetime_to_str};
use syntax::visit;
use syntax::visit::Visitor;

// maps the id of each lifetime reference to the lifetime decl
// that it corresponds to
pub type NamedRegionMap = HashMap<ast::NodeId, ast::DefRegion>;

struct LifetimeContext {
    sess: session::Session,
    named_region_map: @mut NamedRegionMap,
}

enum ScopeChain<'self> {
    EarlyScope(uint, &'self OptVec<ast::Lifetime>, Scope<'self>),
    LateScope(ast::NodeId, &'self OptVec<ast::Lifetime>, Scope<'self>),
    BlockScope(ast::NodeId, Scope<'self>),
    RootScope
}

type Scope<'self> = &'self ScopeChain<'self>;

pub fn crate(sess: session::Session,
             crate: &ast::Crate)
             -> @mut NamedRegionMap {
    let mut ctxt = LifetimeContext {
        sess: sess,
        named_region_map: @mut HashMap::new()
    };
    visit::walk_crate(&mut ctxt, crate, &RootScope);
    sess.abort_if_errors();
    ctxt.named_region_map
}

impl<'self> Visitor<Scope<'self>> for LifetimeContext {
    fn visit_item(&mut self,
                  item: @ast::item,
                  _: Scope<'self>) {
        let scope = match item.node {
            ast::item_fn(*) | // fn lifetimes get added in visit_fn below
            ast::item_mod(*) |
            ast::item_mac(*) |
            ast::item_foreign_mod(*) |
            ast::item_static(*) => {
                RootScope
            }
            ast::item_ty(_, ref generics) |
            ast::item_enum(_, ref generics) |
            ast::item_struct(_, ref generics) |
            ast::item_impl(ref generics, _, _, _) |
            ast::item_trait(ref generics, _, _) => {
                self.check_lifetime_names(&generics.lifetimes);
                EarlyScope(0, &generics.lifetimes, &RootScope)
            }
        };
        debug!("entering scope {:?}", scope);
        visit::walk_item(self, item, &scope);
        debug!("exiting scope {:?}", scope);
    }

    fn visit_fn(&mut self,
                fk: &visit::fn_kind,
                fd: &ast::fn_decl,
                b: &ast::Block,
                s: Span,
                n: ast::NodeId,
                scope: Scope<'self>) {
        match *fk {
            visit::fk_item_fn(_, generics, _, _) |
            visit::fk_method(_, generics, _) => {
                self.visit_fn_decl(
                    n, generics, scope,
                    |this, scope1| visit::walk_fn(this, fk, fd, b,
                                                  s, n, scope1))
            }
            visit::fk_anon(*) | visit::fk_fn_block(*) => {
                visit::walk_fn(self, fk, fd, b, s, n, scope)
            }
        }
    }

    fn visit_ty(&mut self,
                ty: &ast::Ty,
                scope: Scope<'self>) {
        match ty.node {
            ast::ty_closure(@ast::TyClosure { lifetimes: ref lifetimes, _ }) |
            ast::ty_bare_fn(@ast::TyBareFn { lifetimes: ref lifetimes, _ }) => {
                let scope1 = LateScope(ty.id, lifetimes, scope);
                self.check_lifetime_names(lifetimes);
                debug!("pushing fn ty scope id={} due to type", ty.id);
                visit::walk_ty(self, ty, &scope1);
                debug!("popping fn ty scope id={} due to type", ty.id);
            }
            _ => {
                visit::walk_ty(self, ty, scope);
            }
        }
    }

    fn visit_ty_method(&mut self,
                       m: &ast::TypeMethod,
                       scope: Scope<'self>) {
        self.visit_fn_decl(
            m.id, &m.generics, scope,
            |this, scope1| visit::walk_ty_method(this, m, scope1))
    }

    fn visit_block(&mut self,
                   b: &ast::Block,
                   scope: Scope<'self>) {
        let scope1 = BlockScope(b.id, scope);
        debug!("pushing block scope {}", b.id);
        visit::walk_block(self, b, &scope1);
        debug!("popping block scope {}", b.id);
    }

    fn visit_lifetime_ref(&mut self,
                          lifetime_ref: &ast::Lifetime,
                          scope: Scope<'self>) {
        if lifetime_ref.ident == special_idents::statik {
            self.insert_lifetime(lifetime_ref, ast::DefStaticRegion);
            return;
        }
        self.resolve_lifetime_ref(lifetime_ref, scope);
    }
}

impl<'self> ScopeChain<'self> {
    fn count_early_params(&self) -> uint {
        /*!
         * Counts the number of early parameters that are in scope.
         * Used when checking methods so that we assign the early
         * lifetime parameters declared on the method indices that
         * come after those from the type (e.g., if there is something
         * like `impl<'a> Foo { ... fn bar<'b>(...) }` then `'a` would
         * have index 0 and `'b` would have index 1.
         */

        match *self {
            RootScope => 0,
            EarlyScope(base, lifetimes, _) => base + lifetimes.len(),
            LateScope(_, _, s) => s.count_early_params(),
            BlockScope(_, _) => 0,
        }
    }
}

impl LifetimeContext {
    fn visit_fn_decl(&mut self,
                     n: ast::NodeId,
                     generics: &ast::Generics,
                     scope: Scope,
                     walk: |&mut LifetimeContext, Scope|) {
        /*!
         * Handles visiting fns and methods. These are a bit
         * complicated because we must distinguish early- vs late-bound
         * lifetime parameters. We do this by checking which lifetimes
         * appear within type bounds; those are early bound lifetimes,
         * and the rest are late bound.
         *
         * For example:
         *
         *    fn foo<'a,'b,'c,T:Trait<'b>>(...)
         *
         * Here `'a` and `'c` are late bound but `'b` is early
         * bound. Note that early- and late-bound lifetimes may be
         * interspersed together.
         *
         * If early bound lifetimes are present, we separate them into
         * their own list (and likewise for late bound). They will be
         * numbered sequentially, starting from the lowest index that
         * is already in scope (for a fn item, that will be 0, but for
         * a method it might not be). Late bound lifetimes are
         * resolved by name and associated with a binder id (`n`), so
         * the ordering is not important there.
         */

        self.check_lifetime_names(&generics.lifetimes);

        let early_count = scope.count_early_params();
        let referenced_idents =
            free_lifetimes(&generics.ty_params);
        debug!("pushing fn scope id={} due to item/method \
               referenced_idents={:?} \
               early_count={}",
               n,
               referenced_idents.map(|&i| self.sess.str_of(i)),
               early_count);
        if referenced_idents.is_empty() {
            let scope1 = LateScope(n, &generics.lifetimes, scope);

            walk(self, &scope1)
        } else {
            let early: OptVec<ast::Lifetime> =
                generics.lifetimes.iter()
                .filter(|l| referenced_idents.iter().any(|i| i == &l.ident))
                .map(|l| *l)
                .collect();
            let scope1 = EarlyScope(early_count, &early, scope);
            debug!("early names = {:?}",
                   early.map(|l| self.sess.str_of(l.ident)));

            let late: OptVec<ast::Lifetime> =
                generics.lifetimes.iter()
                .filter(|l| !referenced_idents.iter().any(|i| i == &l.ident))
                .map(|l| *l)
                .collect();
            let scope2 = LateScope(n, &late, &scope1);

            walk(self, &scope2)
        }
        debug!("popping fn scope id={} due to item/method", n);
    }

    fn resolve_lifetime_ref(&self,
                            lifetime_ref: &ast::Lifetime,
                            scope: &ScopeChain) {
        // Walk up the scope chain, tracking the number of fn scopes
        // that we pass through, until we find a lifetime with the
        // given name or we run out of scopes. If we encounter a code
        // block, then the lifetime is not bound but free, so switch
        // over to `resolve_free_lifetime_ref()` to complete the
        // search.
        let mut depth = 0;
        let mut scope = scope;
        loop {
            match *scope {
                BlockScope(id, s) => {
                    return self.resolve_free_lifetime_ref(id, lifetime_ref, s);
                }

                RootScope => {
                    break;
                }

                EarlyScope(base, lifetimes, s) => {
                    match search_lifetimes(lifetimes, lifetime_ref) {
                        Some((offset, decl_id)) => {
                            let index = base + offset;
                            let def = ast::DefEarlyBoundRegion(index, decl_id);
                            self.insert_lifetime(lifetime_ref, def);
                            return;
                        }
                        None => {
                            depth += 1;
                            scope = s;
                        }
                    }
                }

                LateScope(binder_id, lifetimes, s) => {
                    match search_lifetimes(lifetimes, lifetime_ref) {
                        Some((_index, decl_id)) => {
                            let def = ast::DefLateBoundRegion(binder_id, depth, decl_id);
                            self.insert_lifetime(lifetime_ref, def);
                            return;
                        }

                        None => {
                            depth += 1;
                            scope = s;
                        }
                    }
                }
            }
        }

        self.unresolved_lifetime_ref(lifetime_ref);
    }

    fn resolve_free_lifetime_ref(&self,
                                 scope_id: ast::NodeId,
                                 lifetime_ref: &ast::Lifetime,
                                 scope: &ScopeChain) {
        // Walk up the scope chain, tracking the outermost free scope,
        // until we encounter a scope that contains the named lifetime
        // or we run out of scopes.
        let mut scope_id = scope_id;
        let mut scope = scope;
        let mut search_result = None;
        loop {
            match *scope {
                BlockScope(id, s) => {
                    scope_id = id;
                    scope = s;
                }

                RootScope => {
                    break;
                }

                EarlyScope(_, lifetimes, s) | LateScope(_, lifetimes, s) => {
                    search_result = search_lifetimes(lifetimes, lifetime_ref);
                    if search_result.is_some() {
                        break;
                    }
                    scope = s;
                }
            }
        }

        match search_result {
            Some((_depth, decl_id)) => {
                let def = ast::DefFreeRegion(scope_id, decl_id);
                self.insert_lifetime(lifetime_ref, def);
            }

            None => {
                self.unresolved_lifetime_ref(lifetime_ref);
            }
        }

    }

    fn unresolved_lifetime_ref(&self,
                               lifetime_ref: &ast::Lifetime) {
        self.sess.span_err(
            lifetime_ref.span,
            format!("use of undeclared lifetime name `'{}`",
                    self.sess.str_of(lifetime_ref.ident)));
    }

    fn check_lifetime_names(&self, lifetimes: &OptVec<ast::Lifetime>) {
        for i in range(0, lifetimes.len()) {
            let lifetime_i = lifetimes.get(i);

            let special_idents = [special_idents::statik];
            for lifetime in lifetimes.iter() {
                if special_idents.iter().any(|&i| i == lifetime.ident) {
                    self.sess.span_err(
                        lifetime.span,
                        format!("illegal lifetime parameter name: `{}`",
                                self.sess.str_of(lifetime.ident)));
                }
            }

            for j in range(i + 1, lifetimes.len()) {
                let lifetime_j = lifetimes.get(j);

                if lifetime_i.ident == lifetime_j.ident {
                    self.sess.span_err(
                        lifetime_j.span,
                        format!("lifetime name `'{}` declared twice in \
                                the same scope",
                                self.sess.str_of(lifetime_j.ident)));
                }
            }
        }
    }

    fn insert_lifetime(&self,
                       lifetime_ref: &ast::Lifetime,
                       def: ast::DefRegion) {
        if lifetime_ref.id == ast::DUMMY_NODE_ID {
            self.sess.span_bug(lifetime_ref.span,
                               "Lifetime reference not renumbered, \
                               probably a bug in syntax::fold");
        }

        debug!("lifetime_ref={} id={} resolved to {:?}",
                lifetime_to_str(lifetime_ref,
                                self.sess.intr()),
                lifetime_ref.id,
                def);
        self.named_region_map.insert(lifetime_ref.id, def);
    }
}

fn search_lifetimes(lifetimes: &OptVec<ast::Lifetime>,
                    lifetime_ref: &ast::Lifetime)
                    -> Option<(uint, ast::NodeId)> {
    for (i, lifetime_decl) in lifetimes.iter().enumerate() {
        if lifetime_decl.ident == lifetime_ref.ident {
            return Some((i, lifetime_decl.id));
        }
    }
    return None;
}

///////////////////////////////////////////////////////////////////////////

pub fn early_bound_lifetimes<'a>(generics: &'a ast::Generics) -> OptVec<ast::Lifetime> {
    let referenced_idents = free_lifetimes(&generics.ty_params);
    if referenced_idents.is_empty() {
        return opt_vec::Empty;
    }

    generics.lifetimes.iter()
        .filter(|l| referenced_idents.iter().any(|i| i == &l.ident))
        .map(|l| *l)
        .collect()
}

pub fn free_lifetimes(ty_params: &OptVec<ast::TyParam>) -> OptVec<ast::Ident> {
    /*!
     * Gathers up and returns the names of any lifetimes that appear
     * free in `ty_params`. Of course, right now, all lifetimes appear
     * free, since we don't have any binders in type parameter
     * declarations, but I just to be forwards compatible for future
     * extensions with my terminology. =)
     */

    let mut collector = FreeLifetimeCollector { names: opt_vec::Empty };
    for ty_param in ty_params.iter() {
        visit::walk_ty_param_bounds(&mut collector, &ty_param.bounds, ());
    }
    return collector.names;

    struct FreeLifetimeCollector {
        names: OptVec<ast::Ident>,
    }

    impl Visitor<()> for FreeLifetimeCollector {
        fn visit_ty(&mut self, t:&ast::Ty, _:()) {
            // for some weird reason visitor doesn't descend into
            // types by default
            visit::walk_ty(self, t, ());
        }

        fn visit_lifetime_ref(&mut self,
                              lifetime_ref: &ast::Lifetime,
                              _: ()) {
            self.names.push(lifetime_ref.ident);
        }
    }
}
