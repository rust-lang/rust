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

use driver::session::Session;
use util::nodemap::NodeMap;
use syntax::ast;
use syntax::codemap::Span;
use syntax::owned_slice::OwnedSlice;
use syntax::parse::token::special_idents;
use syntax::parse::token;
use syntax::print::pprust::{lifetime_to_str};
use syntax::visit;
use syntax::visit::Visitor;

// maps the id of each lifetime reference to the lifetime decl
// that it corresponds to
pub type NamedRegionMap = NodeMap<ast::DefRegion>;

// Returns an instance of some type that implements std::fmt::Show
fn lifetime_show(lt_name: &ast::Name) -> token::InternedString {
    token::get_name(*lt_name)
}

struct LifetimeContext<'a> {
    sess: &'a Session,
    named_region_map: NamedRegionMap,
}

enum ScopeChain<'a> {
    /// EarlyScope(i, ['a, 'b, ...], s) extends s with early-bound
    /// lifetimes, assigning indexes 'a => i, 'b => i+1, ... etc.
    EarlyScope(uint, &'a Vec<ast::Lifetime>, Scope<'a>),
    /// LateScope(binder_id, ['a, 'b, ...], s) extends s with late-bound
    /// lifetimes introduced by the declaration binder_id.
    LateScope(ast::NodeId, &'a Vec<ast::Lifetime>, Scope<'a>),
    /// lifetimes introduced by items within a code block are scoped
    /// to that block.
    BlockScope(ast::NodeId, Scope<'a>),
    RootScope
}

type Scope<'a> = &'a ScopeChain<'a>;

pub fn krate(sess: &Session, krate: &ast::Crate) -> NamedRegionMap {
    let mut ctxt = LifetimeContext {
        sess: sess,
        named_region_map: NodeMap::new()
    };
    visit::walk_crate(&mut ctxt, krate, &RootScope);
    sess.abort_if_errors();
    ctxt.named_region_map
}

impl<'a, 'b> Visitor<Scope<'a>> for LifetimeContext<'b> {
    fn visit_item(&mut self,
                  item: &ast::Item,
                  _: Scope<'a>) {
        let root = RootScope;
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
            ast::ItemTrait(ref generics, _, _, _) => {
                self.check_lifetime_names(&generics.lifetimes);
                EarlyScope(0, &generics.lifetimes, &root)
            }
        };
        debug!("entering scope {:?}", scope);
        visit::walk_item(self, item, &scope);
        debug!("exiting scope {:?}", scope);
    }

    fn visit_fn(&mut self, fk: &visit::FnKind, fd: &ast::FnDecl,
                b: &ast::Block, s: Span, n: ast::NodeId,
                scope: Scope<'a>) {
        match *fk {
            visit::FkItemFn(_, generics, _, _) |
            visit::FkMethod(_, generics, _) => {
                self.visit_fn_decl(
                    n, generics, scope,
                    |this, scope1| visit::walk_fn(this, fk, fd, b, s, scope1))
            }
            visit::FkFnBlock(..) => {
                visit::walk_fn(self, fk, fd, b, s, scope)
            }
        }
    }

    fn visit_ty(&mut self, ty: &ast::Ty, scope: Scope<'a>) {
        match ty.node {
            ast::TyClosure(c, _) | ast::TyProc(c) => {
                push_fn_scope(self, ty, scope, &c.lifetimes);
            }
            ast::TyBareFn(c) => push_fn_scope(self, ty, scope, &c.lifetimes),
            _ => visit::walk_ty(self, ty, scope),
        }

        fn push_fn_scope(this: &mut LifetimeContext,
                         ty: &ast::Ty,
                         scope: Scope,
                         lifetimes: &Vec<ast::Lifetime>) {
            let scope1 = LateScope(ty.id, lifetimes, scope);
            this.check_lifetime_names(lifetimes);
            debug!("pushing fn scope id={} due to type", ty.id);
            visit::walk_ty(this, ty, &scope1);
            debug!("popping fn scope id={} due to type", ty.id);
        }
    }

    fn visit_ty_method(&mut self,
                       m: &ast::TypeMethod,
                       scope: Scope<'a>) {
        self.visit_fn_decl(
            m.id, &m.generics, scope,
            |this, scope1| visit::walk_ty_method(this, m, scope1))
    }

    fn visit_block(&mut self,
                   b: &ast::Block,
                   scope: Scope<'a>) {
        let scope1 = BlockScope(b.id, scope);
        debug!("pushing block scope {}", b.id);
        visit::walk_block(self, b, &scope1);
        debug!("popping block scope {}", b.id);
    }

    fn visit_lifetime_ref(&mut self,
                          lifetime_ref: &ast::Lifetime,
                          scope: Scope<'a>) {
        if lifetime_ref.name == special_idents::statik.name {
            self.insert_lifetime(lifetime_ref, ast::DefStaticRegion);
            return;
        }
        self.resolve_lifetime_ref(lifetime_ref, scope);
    }
}

impl<'a> ScopeChain<'a> {
    fn count_early_params(&self) -> uint {
        /*!
         * Counts the number of early-bound parameters that are in
         * scope.  Used when checking methods: the early-bound
         * lifetime parameters declared on the method are assigned
         * indices that come after the indices from the type.  Given
         * something like `impl<'a> Foo { ... fn bar<'b>(...) }`
         * then `'a` gets index 0 and `'b` gets index 1.
         */

        match *self {
            RootScope => 0,
            EarlyScope(base, lifetimes, _) => base + lifetimes.len(),
            LateScope(_, _, s) => s.count_early_params(),
            BlockScope(_, _) => 0,
        }
    }
}

impl<'a> LifetimeContext<'a> {
    /// Visits self by adding a scope and handling recursive walk over the contents with `walk`.
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
        let referenced_idents = free_lifetimes(&generics.ty_params);
        debug!("pushing fn scope id={} due to fn item/method\
               referenced_idents={:?} \
               early_count={}",
               n,
               referenced_idents.iter().map(lifetime_show).collect::<Vec<token::InternedString>>(),
               early_count);
        if referenced_idents.is_empty() {
            let scope1 = LateScope(n, &generics.lifetimes, scope);
            walk(self, &scope1)
        } else {
            let (early, late) = generics.lifetimes.clone().partition(
                |l| referenced_idents.iter().any(|&i| i == l.name));

            let scope1 = EarlyScope(early_count, &early, scope);
            let scope2 = LateScope(n, &late, &scope1);

            walk(self, &scope2);
        }
        debug!("popping fn scope id={} due to fn item/method", n);
    }

    fn resolve_lifetime_ref(&mut self,
                            lifetime_ref: &ast::Lifetime,
                            scope: Scope) {
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

    fn resolve_free_lifetime_ref(&mut self,
                                 scope_id: ast::NodeId,
                                 lifetime_ref: &ast::Lifetime,
                                 scope: Scope) {
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

                EarlyScope(_, lifetimes, s) |
                LateScope(_, lifetimes, s) => {
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
                    token::get_name(lifetime_ref.name)));
    }

    fn check_lifetime_names(&self, lifetimes: &Vec<ast::Lifetime>) {
        for i in range(0, lifetimes.len()) {
            let lifetime_i = lifetimes.get(i);

            let special_idents = [special_idents::statik];
            for lifetime in lifetimes.iter() {
                if special_idents.iter().any(|&i| i.name == lifetime.name) {
                    self.sess.span_err(
                        lifetime.span,
                        format!("illegal lifetime parameter name: `{}`",
                                token::get_name(lifetime.name)));
                }
            }

            for j in range(i + 1, lifetimes.len()) {
                let lifetime_j = lifetimes.get(j);

                if lifetime_i.name == lifetime_j.name {
                    self.sess.span_err(
                        lifetime_j.span,
                        format!("lifetime name `'{}` declared twice in \
                                the same scope",
                                token::get_name(lifetime_j.name)));
                }
            }
        }
    }

    fn insert_lifetime(&mut self,
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
        self.named_region_map.insert(lifetime_ref.id, def);
    }
}

fn search_lifetimes(lifetimes: &Vec<ast::Lifetime>,
                    lifetime_ref: &ast::Lifetime)
                    -> Option<(uint, ast::NodeId)> {
    for (i, lifetime_decl) in lifetimes.iter().enumerate() {
        if lifetime_decl.name == lifetime_ref.name {
            return Some((i, lifetime_decl.id));
        }
    }
    return None;
}

///////////////////////////////////////////////////////////////////////////

pub fn early_bound_lifetimes<'a>(generics: &'a ast::Generics) -> Vec<ast::Lifetime> {
    let referenced_idents = free_lifetimes(&generics.ty_params);
    if referenced_idents.is_empty() {
        return Vec::new();
    }

    generics.lifetimes.iter()
        .filter(|l| referenced_idents.iter().any(|&i| i == l.name))
        .map(|l| *l)
        .collect()
}

pub fn free_lifetimes(ty_params: &OwnedSlice<ast::TyParam>) -> Vec<ast::Name> {
    /*!
     * Gathers up and returns the names of any lifetimes that appear
     * free in `ty_params`. Of course, right now, all lifetimes appear
     * free, since we don't currently have any binders in type parameter
     * declarations; just being forwards compatible with future extensions.
     */

    let mut collector = FreeLifetimeCollector { names: vec!() };
    for ty_param in ty_params.iter() {
        visit::walk_ty_param_bounds(&mut collector, &ty_param.bounds, ());
    }
    return collector.names;

    struct FreeLifetimeCollector {
        names: Vec<ast::Name>,
    }

    impl Visitor<()> for FreeLifetimeCollector {
        fn visit_lifetime_ref(&mut self,
                              lifetime_ref: &ast::Lifetime,
                              _: ()) {
            self.names.push(lifetime_ref.name);
        }
    }
}
