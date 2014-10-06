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
use middle::subst;
use syntax::ast;
use syntax::codemap::Span;
use syntax::owned_slice::OwnedSlice;
use syntax::parse::token::special_idents;
use syntax::parse::token;
use syntax::print::pprust::{lifetime_to_string};
use syntax::visit;
use syntax::visit::Visitor;
use util::nodemap::NodeMap;

#[deriving(Clone, PartialEq, Eq, Hash, Encodable, Decodable, Show)]
pub enum DefRegion {
    DefStaticRegion,
    DefEarlyBoundRegion(/* space */ subst::ParamSpace,
                        /* index */ uint,
                        /* lifetime decl */ ast::NodeId),
    DefLateBoundRegion(/* binder_id */ ast::NodeId,
                       /* depth */ uint,
                       /* lifetime decl */ ast::NodeId),
    DefFreeRegion(/* block scope */ ast::NodeId,
                  /* lifetime decl */ ast::NodeId),
}

// maps the id of each lifetime reference to the lifetime decl
// that it corresponds to
pub type NamedRegionMap = NodeMap<DefRegion>;

// Returns an instance of some type that implements std::fmt::Show
fn lifetime_show(lt_name: &ast::Name) -> token::InternedString {
    token::get_name(*lt_name)
}

struct LifetimeContext<'a> {
    sess: &'a Session,
    named_region_map: &'a mut NamedRegionMap,
    scope: Scope<'a>
}

enum ScopeChain<'a> {
    /// EarlyScope(i, ['a, 'b, ...], s) extends s with early-bound
    /// lifetimes, assigning indexes 'a => i, 'b => i+1, ... etc.
    EarlyScope(subst::ParamSpace, &'a Vec<ast::LifetimeDef>, Scope<'a>),
    /// LateScope(binder_id, ['a, 'b, ...], s) extends s with late-bound
    /// lifetimes introduced by the declaration binder_id.
    LateScope(ast::NodeId, &'a Vec<ast::LifetimeDef>, Scope<'a>),
    /// lifetimes introduced by items within a code block are scoped
    /// to that block.
    BlockScope(ast::NodeId, Scope<'a>),
    RootScope
}

type Scope<'a> = &'a ScopeChain<'a>;

static ROOT_SCOPE: ScopeChain<'static> = RootScope;

pub fn krate(sess: &Session, krate: &ast::Crate) -> NamedRegionMap {
    let mut named_region_map = NodeMap::new();
    visit::walk_crate(&mut LifetimeContext {
        sess: sess,
        named_region_map: &mut named_region_map,
        scope: &ROOT_SCOPE
    }, krate);
    sess.abort_if_errors();
    named_region_map
}

impl<'a, 'v> Visitor<'v> for LifetimeContext<'a> {
    fn visit_item(&mut self, item: &ast::Item) {
        let lifetimes = match item.node {
            ast::ItemFn(..) | // fn lifetimes get added in visit_fn below
            ast::ItemMod(..) |
            ast::ItemMac(..) |
            ast::ItemForeignMod(..) |
            ast::ItemStatic(..) | ast::ItemConst(..) => {
                self.with(|_, f| f(RootScope), |v| visit::walk_item(v, item));
                return;
            }
            ast::ItemTy(_, ref generics) |
            ast::ItemEnum(_, ref generics) |
            ast::ItemStruct(_, ref generics) |
            ast::ItemTrait(ref generics, _, _, _) => {
                self.with(|scope, f| {
                    f(EarlyScope(subst::TypeSpace,
                                 &generics.lifetimes,
                                 scope))
                }, |v| v.check_lifetime_defs(&generics.lifetimes));
                &generics.lifetimes
            }
            ast::ItemImpl(ref generics, _, _, _) => {
                self.with(|scope, f| {
                    f(EarlyScope(subst::TypeSpace,
                                 &generics.lifetimes,
                                 scope))
                }, |v| v.check_lifetime_defs(&generics.lifetimes));
                &generics.lifetimes
            }
        };

        self.with(|_, f| f(EarlyScope(subst::TypeSpace, lifetimes, &ROOT_SCOPE)), |v| {
            debug!("entering scope {:?}", v.scope);
            v.check_lifetime_defs(lifetimes);
            visit::walk_item(v, item);
            debug!("exiting scope {:?}", v.scope);
        });
    }

    fn visit_fn(&mut self, fk: visit::FnKind<'v>, fd: &'v ast::FnDecl,
                b: &'v ast::Block, s: Span, n: ast::NodeId) {
        match fk {
            visit::FkItemFn(_, generics, _, _) |
            visit::FkMethod(_, generics, _) => {
                self.visit_fn_decl(n, generics, |v| visit::walk_fn(v, fk, fd, b, s))
            }
            visit::FkFnBlock(..) => {
                visit::walk_fn(self, fk, fd, b, s)
            }
        }
    }

    fn visit_ty(&mut self, ty: &ast::Ty) {
        let lifetimes = match ty.node {
            ast::TyClosure(ref c) | ast::TyProc(ref c) => &c.lifetimes,
            ast::TyBareFn(ref c) => &c.lifetimes,
            _ => return visit::walk_ty(self, ty)
        };

        self.with(|scope, f| f(LateScope(ty.id, lifetimes, scope)), |v| {
            v.check_lifetime_defs(lifetimes);
            debug!("pushing fn scope id={} due to type", ty.id);
            visit::walk_ty(v, ty);
            debug!("popping fn scope id={} due to type", ty.id);
        });
    }

    fn visit_ty_method(&mut self, m: &ast::TypeMethod) {
        self.visit_fn_decl(m.id, &m.generics, |v| visit::walk_ty_method(v, m))
    }

    fn visit_block(&mut self, b: &ast::Block) {
        debug!("pushing block scope {}", b.id);
        self.with(|scope, f| f(BlockScope(b.id, scope)), |v| visit::walk_block(v, b));
        debug!("popping block scope {}", b.id);
    }

    fn visit_lifetime_ref(&mut self, lifetime_ref: &ast::Lifetime) {
        if lifetime_ref.name == special_idents::static_lifetime.name {
            self.insert_lifetime(lifetime_ref, DefStaticRegion);
            return;
        }
        self.resolve_lifetime_ref(lifetime_ref);
    }

    fn visit_generics(&mut self, generics: &ast::Generics) {
        for ty_param in generics.ty_params.iter() {
            self.visit_ty_param_bounds(&ty_param.bounds);
            match ty_param.default {
                Some(ref ty) => self.visit_ty(&**ty),
                None => {}
            }
        }
        for predicate in generics.where_clause.predicates.iter() {
            self.visit_ident(predicate.span, predicate.ident);
            self.visit_ty_param_bounds(&predicate.bounds);
        }
    }
}

impl<'a> LifetimeContext<'a> {
    fn with(&mut self, wrap_scope: |Scope, |ScopeChain||, f: |&mut LifetimeContext|) {
        let LifetimeContext { sess, ref mut named_region_map, scope} = *self;
        wrap_scope(scope, |scope1| f(&mut LifetimeContext {
            sess: sess,
            named_region_map: *named_region_map,
            scope: &scope1
        }))
    }

    fn visit_ty_param_bounds(&mut self,
                             bounds: &OwnedSlice<ast::TyParamBound>) {
        for bound in bounds.iter() {
            match *bound {
                ast::TraitTyParamBound(ref trait_ref) => {
                    self.visit_trait_ref(trait_ref);
                }
                ast::UnboxedFnTyParamBound(ref fn_decl) => {
                    self.visit_unboxed_fn_ty_param_bound(&**fn_decl);
                }
                ast::RegionTyParamBound(ref lifetime) => {
                    self.visit_lifetime_ref(lifetime);
                }
            }
        }
    }

    fn visit_trait_ref(&mut self, trait_ref: &ast::TraitRef) {
        self.with(|scope, f| {
            f(LateScope(trait_ref.ref_id, &trait_ref.lifetimes, scope))
        }, |v| {
            v.check_lifetime_defs(&trait_ref.lifetimes);
            for lifetime in trait_ref.lifetimes.iter() {
                v.visit_lifetime_decl(lifetime);
            }
            v.visit_path(&trait_ref.path, trait_ref.ref_id);
        })
    }

    fn visit_unboxed_fn_ty_param_bound(&mut self,
                                       bound: &ast::UnboxedFnBound) {
        self.with(|scope, f| {
            f(LateScope(bound.ref_id, &bound.lifetimes, scope))
        }, |v| {
            for argument in bound.decl.inputs.iter() {
                v.visit_ty(&*argument.ty);
            }
            v.visit_ty(&*bound.decl.output);
        })
    }

    /// Visits self by adding a scope and handling recursive walk over the contents with `walk`.
    fn visit_fn_decl(&mut self,
                     n: ast::NodeId,
                     generics: &ast::Generics,
                     walk: |&mut LifetimeContext|) {
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

        let referenced_idents = early_bound_lifetime_names(generics);
        debug!("pushing fn scope id={} due to fn item/method\
               referenced_idents={:?}",
               n,
               referenced_idents.iter().map(lifetime_show).collect::<Vec<token::InternedString>>());
        let lifetimes = &generics.lifetimes;
        if referenced_idents.is_empty() {
            self.with(|scope, f| f(LateScope(n, lifetimes, scope)), |v| {
                v.check_lifetime_defs(lifetimes);
                walk(v);
            });
        } else {
            let (early, late) = lifetimes.clone().partition(
                |l| referenced_idents.iter().any(|&i| i == l.lifetime.name));

            self.with(|scope, f| f(EarlyScope(subst::FnSpace, &early, scope)), |v| {
                v.with(|scope1, f| f(LateScope(n, &late, scope1)), |v| {
                    v.check_lifetime_defs(lifetimes);
                    walk(v);
                });
            });
        }
        debug!("popping fn scope id={} due to fn item/method", n);
    }

    fn resolve_lifetime_ref(&mut self, lifetime_ref: &ast::Lifetime) {
        // Walk up the scope chain, tracking the number of fn scopes
        // that we pass through, until we find a lifetime with the
        // given name or we run out of scopes. If we encounter a code
        // block, then the lifetime is not bound but free, so switch
        // over to `resolve_free_lifetime_ref()` to complete the
        // search.
        let mut depth = 0;
        let mut scope = self.scope;
        loop {
            match *scope {
                BlockScope(id, s) => {
                    return self.resolve_free_lifetime_ref(id, lifetime_ref, s);
                }

                RootScope => {
                    break;
                }

                EarlyScope(space, lifetimes, s) => {
                    match search_lifetimes(lifetimes, lifetime_ref) {
                        Some((index, decl_id)) => {
                            let def = DefEarlyBoundRegion(space, index, decl_id);
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
                            let def = DefLateBoundRegion(binder_id, depth, decl_id);
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
                let def = DefFreeRegion(scope_id, decl_id);
                self.insert_lifetime(lifetime_ref, def);
            }

            None => {
                self.unresolved_lifetime_ref(lifetime_ref);
            }
        }

    }

    fn unresolved_lifetime_ref(&self, lifetime_ref: &ast::Lifetime) {
        self.sess.span_err(
            lifetime_ref.span,
            format!("use of undeclared lifetime name `{}`",
                    token::get_name(lifetime_ref.name)).as_slice());
    }

    fn check_lifetime_defs(&mut self, lifetimes: &Vec<ast::LifetimeDef>) {
        for i in range(0, lifetimes.len()) {
            let lifetime_i = lifetimes.get(i);

            let special_idents = [special_idents::static_lifetime];
            for lifetime in lifetimes.iter() {
                if special_idents.iter().any(|&i| i.name == lifetime.lifetime.name) {
                    self.sess.span_err(
                        lifetime.lifetime.span,
                        format!("illegal lifetime parameter name: `{}`",
                                token::get_name(lifetime.lifetime.name))
                            .as_slice());
                }
            }

            for j in range(i + 1, lifetimes.len()) {
                let lifetime_j = lifetimes.get(j);

                if lifetime_i.lifetime.name == lifetime_j.lifetime.name {
                    self.sess.span_err(
                        lifetime_j.lifetime.span,
                        format!("lifetime name `{}` declared twice in \
                                the same scope",
                                token::get_name(lifetime_j.lifetime.name))
                            .as_slice());
                }
            }

            for bound in lifetime_i.bounds.iter() {
                self.resolve_lifetime_ref(bound);
            }
        }
    }

    fn insert_lifetime(&mut self,
                       lifetime_ref: &ast::Lifetime,
                       def: DefRegion) {
        if lifetime_ref.id == ast::DUMMY_NODE_ID {
            self.sess.span_bug(lifetime_ref.span,
                               "lifetime reference not renumbered, \
                               probably a bug in syntax::fold");
        }

        debug!("lifetime_ref={} id={} resolved to {:?}",
                lifetime_to_string(lifetime_ref),
                lifetime_ref.id,
                def);
        self.named_region_map.insert(lifetime_ref.id, def);
    }
}

fn search_lifetimes(lifetimes: &Vec<ast::LifetimeDef>,
                    lifetime_ref: &ast::Lifetime)
                    -> Option<(uint, ast::NodeId)> {
    for (i, lifetime_decl) in lifetimes.iter().enumerate() {
        if lifetime_decl.lifetime.name == lifetime_ref.name {
            return Some((i, lifetime_decl.lifetime.id));
        }
    }
    return None;
}

///////////////////////////////////////////////////////////////////////////

pub fn early_bound_lifetimes<'a>(generics: &'a ast::Generics) -> Vec<ast::LifetimeDef> {
    let referenced_idents = early_bound_lifetime_names(generics);
    if referenced_idents.is_empty() {
        return Vec::new();
    }

    generics.lifetimes.iter()
        .filter(|l| referenced_idents.iter().any(|&i| i == l.lifetime.name))
        .map(|l| (*l).clone())
        .collect()
}

fn early_bound_lifetime_names(generics: &ast::Generics) -> Vec<ast::Name> {
    /*!
     * Given a set of generic declarations, returns a list of names
     * containing all early bound lifetime names for those
     * generics. (In fact, this list may also contain other names.)
     */

    // Create two lists, dividing the lifetimes into early/late bound.
    // Initially, all of them are considered late, but we will move
    // things from late into early as we go if we find references to
    // them.
    let mut early_bound = Vec::new();
    let mut late_bound = generics.lifetimes.iter()
                                           .map(|l| l.lifetime.name)
                                           .collect();

    // Any lifetime that appears in a type bound is early.
    {
        let mut collector =
            FreeLifetimeCollector { early_bound: &mut early_bound,
                                    late_bound: &mut late_bound };
        for ty_param in generics.ty_params.iter() {
            visit::walk_ty_param_bounds(&mut collector, &ty_param.bounds);
        }
        for predicate in generics.where_clause.predicates.iter() {
            visit::walk_ty_param_bounds(&mut collector, &predicate.bounds);
        }
    }

    // Any lifetime that either has a bound or is referenced by a
    // bound is early.
    for lifetime_def in generics.lifetimes.iter() {
        if !lifetime_def.bounds.is_empty() {
            shuffle(&mut early_bound, &mut late_bound,
                    lifetime_def.lifetime.name);
            for bound in lifetime_def.bounds.iter() {
                shuffle(&mut early_bound, &mut late_bound,
                        bound.name);
            }
        }
    }
    return early_bound;

    struct FreeLifetimeCollector<'a> {
        early_bound: &'a mut Vec<ast::Name>,
        late_bound: &'a mut Vec<ast::Name>,
    }

    impl<'a, 'v> Visitor<'v> for FreeLifetimeCollector<'a> {
        fn visit_lifetime_ref(&mut self, lifetime_ref: &ast::Lifetime) {
            shuffle(self.early_bound, self.late_bound,
                    lifetime_ref.name);
        }
    }

    fn shuffle(early_bound: &mut Vec<ast::Name>,
               late_bound: &mut Vec<ast::Name>,
               name: ast::Name) {
        match late_bound.iter().position(|n| *n == name) {
            Some(index) => {
                late_bound.swap_remove(index);
                early_bound.push(name);
            }
            None => { }
        }
    }
}
