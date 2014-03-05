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
use std::cell::RefCell;
use collections::HashMap;
use syntax::ast;
use syntax::codemap::Span;
use syntax::opt_vec::OptVec;
use syntax::parse::token::special_idents;
use syntax::parse::token;
use syntax::print::pprust::{lifetime_to_str};
use syntax::visit;
use syntax::visit::Visitor;

// maps the id of each lifetime reference to the lifetime decl
// that it corresponds to
pub type NamedRegionMap = HashMap<ast::NodeId, ast::DefRegion>;

struct LifetimeContext {
    sess: session::Session,
    named_region_map: @RefCell<NamedRegionMap>,
}

enum ScopeChain<'a> {
    ItemScope(&'a OptVec<ast::Lifetime>),
    FnScope(ast::NodeId, &'a OptVec<ast::Lifetime>, &'a ScopeChain<'a>),
    BlockScope(ast::NodeId, &'a ScopeChain<'a>),
    RootScope
}

pub fn krate(sess: session::Session, krate: &ast::Crate)
             -> @RefCell<NamedRegionMap> {
    let mut ctxt = LifetimeContext {
        sess: sess,
        named_region_map: @RefCell::new(HashMap::new())
    };
    visit::walk_crate(&mut ctxt, krate, &RootScope);
    sess.abort_if_errors();
    ctxt.named_region_map
}

impl<'a> Visitor<&'a ScopeChain<'a>> for LifetimeContext {
    fn visit_item(&mut self,
                  item: &ast::Item,
                  _: &'a ScopeChain<'a>) {
        let scope = match item.node {
            ast::ItemFn(..) | // fn lifetimes get added in visit_fn below
            ast::ItemMod(..) |
            ast::ItemMac(..) |
            ast::ItemForeignMod(..) |
            ast::ItemStatic(..) => {
                RootScope
            }
            ast::ItemTy(_, ref generics) |
            ast::ItemEnum(_, ref generics) |
            ast::ItemStruct(_, ref generics) |
            ast::ItemImpl(ref generics, _, _, _) |
            ast::ItemTrait(ref generics, _, _) => {
                self.check_lifetime_names(&generics.lifetimes);
                ItemScope(&generics.lifetimes)
            }
        };
        debug!("entering scope {:?}", scope);
        visit::walk_item(self, item, &scope);
        debug!("exiting scope {:?}", scope);
    }

    fn visit_fn(&mut self, fk: &visit::FnKind, fd: &ast::FnDecl,
                b: &ast::Block, s: Span, n: ast::NodeId,
                scope: &'a ScopeChain<'a>) {
        match *fk {
            visit::FkItemFn(_, generics, _, _) |
            visit::FkMethod(_, generics, _) => {
                let scope1 = FnScope(n, &generics.lifetimes, scope);
                self.check_lifetime_names(&generics.lifetimes);
                debug!("pushing fn scope id={} due to item/method", n);
                visit::walk_fn(self, fk, fd, b, s, n, &scope1);
                debug!("popping fn scope id={} due to item/method", n);
            }
            visit::FkFnBlock(..) => {
                visit::walk_fn(self, fk, fd, b, s, n, scope);
            }
        }
    }

    fn visit_ty(&mut self, ty: &ast::Ty,
                scope: &'a ScopeChain<'a>) {
        match ty.node {
            ast::TyClosure(closure) => {
                let scope1 = FnScope(ty.id, &closure.lifetimes, scope);
                self.check_lifetime_names(&closure.lifetimes);
                debug!("pushing fn scope id={} due to type", ty.id);
                visit::walk_ty(self, ty, &scope1);
                debug!("popping fn scope id={} due to type", ty.id);
            }
            ast::TyBareFn(bare_fn) => {
                let scope1 = FnScope(ty.id, &bare_fn.lifetimes, scope);
                self.check_lifetime_names(&bare_fn.lifetimes);
                debug!("pushing fn scope id={} due to type", ty.id);
                visit::walk_ty(self, ty, &scope1);
                debug!("popping fn scope id={} due to type", ty.id);
            }
            _ => {
                visit::walk_ty(self, ty, scope);
            }
        }
    }

    fn visit_ty_method(&mut self,
                       m: &ast::TypeMethod,
                       scope: &'a ScopeChain<'a>) {
        let scope1 = FnScope(m.id, &m.generics.lifetimes, scope);
        self.check_lifetime_names(&m.generics.lifetimes);
        debug!("pushing fn scope id={} due to ty_method", m.id);
        visit::walk_ty_method(self, m, &scope1);
        debug!("popping fn scope id={} due to ty_method", m.id);
    }

    fn visit_block(&mut self,
                   b: &ast::Block,
                   scope: &'a ScopeChain<'a>) {
        let scope1 = BlockScope(b.id, scope);
        debug!("pushing block scope {}", b.id);
        visit::walk_block(self, b, &scope1);
        debug!("popping block scope {}", b.id);
    }

    fn visit_lifetime_ref(&mut self,
                          lifetime_ref: &ast::Lifetime,
                          scope: &'a ScopeChain<'a>) {
        if lifetime_ref.ident == special_idents::statik.name {
            self.insert_lifetime(lifetime_ref, ast::DefStaticRegion);
            return;
        }
        self.resolve_lifetime_ref(lifetime_ref, scope);
    }
}

impl LifetimeContext {
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

                ItemScope(lifetimes) => {
                    match search_lifetimes(lifetimes, lifetime_ref) {
                        Some((index, decl_id)) => {
                            let def = ast::DefEarlyBoundRegion(index, decl_id);
                            self.insert_lifetime(lifetime_ref, def);
                            return;
                        }
                        None => {
                            break;
                        }
                    }
                }

                FnScope(id, lifetimes, s) => {
                    match search_lifetimes(lifetimes, lifetime_ref) {
                        Some((_index, decl_id)) => {
                            let def = ast::DefLateBoundRegion(id, depth, decl_id);
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

                ItemScope(lifetimes) => {
                    search_result = search_lifetimes(lifetimes, lifetime_ref);
                    break;
                }

                FnScope(_, lifetimes, s) => {
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
                    token::get_name(lifetime_ref.ident)));
    }

    fn check_lifetime_names(&self, lifetimes: &OptVec<ast::Lifetime>) {
        for i in range(0, lifetimes.len()) {
            let lifetime_i = lifetimes.get(i);

            let special_idents = [special_idents::statik];
            for lifetime in lifetimes.iter() {
                if special_idents.iter().any(|&i| i.name == lifetime.ident) {
                    self.sess.span_err(
                        lifetime.span,
                        format!("illegal lifetime parameter name: `{}`",
                                token::get_name(lifetime.ident)));
                }
            }

            for j in range(i + 1, lifetimes.len()) {
                let lifetime_j = lifetimes.get(j);

                if lifetime_i.ident == lifetime_j.ident {
                    self.sess.span_err(
                        lifetime_j.span,
                        format!("lifetime name `'{}` declared twice in \
                                the same scope",
                                token::get_name(lifetime_j.ident)));
                }
            }
        }
    }

    fn insert_lifetime(&self,
                       lifetime_ref: &ast::Lifetime,
                       def: ast::DefRegion) {
        if lifetime_ref.id == ast::DUMMY_NODE_ID {
            self.sess.span_bug(lifetime_ref.span,
                               "lifetime reference not renumbered, \
                               probably a bug in syntax::fold");
        }

        debug!("lifetime_ref={} id={} resolved to {:?}",
                lifetime_to_str(lifetime_ref),
                lifetime_ref.id,
                def);
        let mut named_region_map = self.named_region_map.borrow_mut();
        named_region_map.get().insert(lifetime_ref.id, def);
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
