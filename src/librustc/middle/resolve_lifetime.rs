// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Name resolution for lifetimes.
//!
//! Name resolution for lifetimes follows MUCH simpler rules than the
//! full resolve. For example, lifetime names are never exported or
//! used between functions, and they operate in a purely top-down
//! way. Therefore we break lifetime name resolution into a separate pass.

pub use self::DefRegion::*;
use self::ScopeChain::*;

use session::Session;
use middle::def::{self, DefMap};
use middle::region;
use middle::subst;
use middle::ty;
use std::fmt;
use syntax::ast;
use syntax::codemap::Span;
use syntax::parse::token::special_idents;
use syntax::parse::token;
use syntax::print::pprust::{lifetime_to_string};
use syntax::visit;
use syntax::visit::Visitor;
use util::nodemap::NodeMap;

#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, Debug)]
pub enum DefRegion {
    DefStaticRegion,
    DefEarlyBoundRegion(/* space */ subst::ParamSpace,
                        /* index */ u32,
                        /* lifetime decl */ ast::NodeId),
    DefLateBoundRegion(ty::DebruijnIndex,
                       /* lifetime decl */ ast::NodeId),
    DefFreeRegion(/* block scope */ region::DestructionScopeData,
                  /* lifetime decl */ ast::NodeId),
}

// Maps the id of each lifetime reference to the lifetime decl
// that it corresponds to.
pub type NamedRegionMap = NodeMap<DefRegion>;

struct LifetimeContext<'a> {
    sess: &'a Session,
    named_region_map: &'a mut NamedRegionMap,
    scope: Scope<'a>,
    def_map: &'a DefMap,
    // Deep breath. Our representation for poly trait refs contains a single
    // binder and thus we only allow a single level of quantification. However,
    // the syntax of Rust permits quantification in two places, e.g., `T: for <'a> Foo<'a>`
    // and `for <'a, 'b> &'b T: Foo<'a>`. In order to get the de Bruijn indices
    // correct when representing these constraints, we should only introduce one
    // scope. However, we want to support both locations for the quantifier and
    // during lifetime resolution we want precise information (so we can't
    // desugar in an earlier phase).

    // SO, if we encounter a quantifier at the outer scope, we set
    // trait_ref_hack to true (and introduce a scope), and then if we encounter
    // a quantifier at the inner scope, we error. If trait_ref_hack is false,
    // then we introduce the scope at the inner quantifier.

    // I'm sorry.
    trait_ref_hack: bool,
}

enum ScopeChain<'a> {
    /// EarlyScope(i, ['a, 'b, ...], s) extends s with early-bound
    /// lifetimes, assigning indexes 'a => i, 'b => i+1, ... etc.
    EarlyScope(subst::ParamSpace, &'a Vec<ast::LifetimeDef>, Scope<'a>),
    /// LateScope(['a, 'b, ...], s) extends s with late-bound
    /// lifetimes introduced by the declaration binder_id.
    LateScope(&'a Vec<ast::LifetimeDef>, Scope<'a>),
    /// lifetimes introduced by items within a code block are scoped
    /// to that block.
    BlockScope(region::DestructionScopeData, Scope<'a>),
    RootScope
}

type Scope<'a> = &'a ScopeChain<'a>;

static ROOT_SCOPE: ScopeChain<'static> = RootScope;

pub fn krate(sess: &Session, krate: &ast::Crate, def_map: &DefMap) -> NamedRegionMap {
    let mut named_region_map = NodeMap();
    visit::walk_crate(&mut LifetimeContext {
        sess: sess,
        named_region_map: &mut named_region_map,
        scope: &ROOT_SCOPE,
        def_map: def_map,
        trait_ref_hack: false,
    }, krate);
    sess.abort_if_errors();
    named_region_map
}

impl<'a, 'v> Visitor<'v> for LifetimeContext<'a> {
    fn visit_item(&mut self, item: &ast::Item) {
        // Items always introduce a new root scope
        self.with(RootScope, |_, this| {
            match item.node {
                ast::ItemFn(..) => {
                    // Fn lifetimes get added in visit_fn below:
                    visit::walk_item(this, item);
                }
                ast::ItemExternCrate(_) |
                ast::ItemUse(_) |
                ast::ItemMod(..) |
                ast::ItemMac(..) |
                ast::ItemForeignMod(..) |
                ast::ItemStatic(..) |
                ast::ItemConst(..) => {
                    // These sorts of items have no lifetime parameters at all.
                    visit::walk_item(this, item);
                }
                ast::ItemTy(_, ref generics) |
                ast::ItemEnum(_, ref generics) |
                ast::ItemStruct(_, ref generics) |
                ast::ItemTrait(_, ref generics, _, _) |
                ast::ItemImpl(_, _, ref generics, _, _, _) => {
                    // These kinds of items have only early bound lifetime parameters.
                    let lifetimes = &generics.lifetimes;
                    let early_scope = EarlyScope(subst::TypeSpace, lifetimes, &ROOT_SCOPE);
                    this.with(early_scope, |old_scope, this| {
                        this.check_lifetime_defs(old_scope, lifetimes);
                        visit::walk_item(this, item);
                    });
                }
            }
        });
    }

    fn visit_fn(&mut self, fk: visit::FnKind<'v>, fd: &'v ast::FnDecl,
                b: &'v ast::Block, s: Span, _: ast::NodeId) {
        match fk {
            visit::FkItemFn(_, generics, _, _) |
            visit::FkMethod(_, generics, _) => {
                self.visit_early_late(subst::FnSpace, generics, |this| {
                    visit::walk_fn(this, fk, fd, b, s)
                })
            }
            visit::FkFnBlock(..) => {
                visit::walk_fn(self, fk, fd, b, s)
            }
        }
    }

    fn visit_ty(&mut self, ty: &ast::Ty) {
        match ty.node {
            ast::TyBareFn(ref c) => {
                visit::walk_lifetime_decls_helper(self, &c.lifetimes);
                self.with(LateScope(&c.lifetimes, self.scope), |old_scope, this| {
                    // a bare fn has no bounds, so everything
                    // contained within is scoped within its binder.
                    this.check_lifetime_defs(old_scope, &c.lifetimes);
                    visit::walk_ty(this, ty);
                });
            }
            ast::TyPath(ref path, id) => {
                // if this path references a trait, then this will resolve to
                // a trait ref, which introduces a binding scope.
                match self.def_map.borrow().get(&id) {
                    Some(&def::DefTrait(..)) => {
                        self.with(LateScope(&Vec::new(), self.scope), |_, this| {
                            this.visit_path(path, id);
                        });
                    }
                    _ => {
                        visit::walk_ty(self, ty);
                    }
                }
            }
            _ => {
                visit::walk_ty(self, ty)
            }
        }
    }

    fn visit_ty_method(&mut self, m: &ast::TypeMethod) {
        self.visit_early_late(
            subst::FnSpace, &m.generics,
            |this| visit::walk_ty_method(this, m))
    }

    fn visit_block(&mut self, b: &ast::Block) {
        self.with(BlockScope(region::DestructionScopeData::new(b.id),
                             self.scope),
                  |_, this| visit::walk_block(this, b));
    }

    fn visit_lifetime_ref(&mut self, lifetime_ref: &ast::Lifetime) {
        if lifetime_ref.name == special_idents::static_lifetime.name {
            self.insert_lifetime(lifetime_ref, DefStaticRegion);
            return;
        }
        self.resolve_lifetime_ref(lifetime_ref);
    }

    fn visit_generics(&mut self, generics: &ast::Generics) {
        for ty_param in &*generics.ty_params {
            visit::walk_ty_param_bounds_helper(self, &ty_param.bounds);
            match ty_param.default {
                Some(ref ty) => self.visit_ty(&**ty),
                None => {}
            }
        }
        for predicate in &generics.where_clause.predicates {
            match predicate {
                &ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate{ ref bounded_ty,
                                                                               ref bounds,
                                                                               ref bound_lifetimes,
                                                                               .. }) => {
                    if bound_lifetimes.len() > 0 {
                        self.trait_ref_hack = true;
                        let result = self.with(LateScope(bound_lifetimes, self.scope),
                                               |old_scope, this| {
                            this.check_lifetime_defs(old_scope, bound_lifetimes);
                            this.visit_ty(&**bounded_ty);
                            visit::walk_ty_param_bounds_helper(this, bounds);
                        });
                        self.trait_ref_hack = false;
                        result
                    } else {
                        self.visit_ty(&**bounded_ty);
                        visit::walk_ty_param_bounds_helper(self, bounds);
                    }
                }
                &ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate{ref lifetime,
                                                                                ref bounds,
                                                                                .. }) => {

                    self.visit_lifetime_ref(lifetime);
                    for bound in bounds {
                        self.visit_lifetime_ref(bound);
                    }
                }
                &ast::WherePredicate::EqPredicate(ast::WhereEqPredicate{ id,
                                                                         ref path,
                                                                         ref ty,
                                                                         .. }) => {
                    self.visit_path(path, id);
                    self.visit_ty(&**ty);
                }
            }
        }
    }

    fn visit_poly_trait_ref(&mut self,
                            trait_ref: &ast::PolyTraitRef,
                            _modifier: &ast::TraitBoundModifier) {
        debug!("visit_poly_trait_ref trait_ref={:?}", trait_ref);

        if !self.trait_ref_hack || trait_ref.bound_lifetimes.len() > 0 {
            if self.trait_ref_hack {
                println!("{:?}", trait_ref.span);
                span_err!(self.sess, trait_ref.span, E0316,
                          "nested quantification of lifetimes");
            }
            self.with(LateScope(&trait_ref.bound_lifetimes, self.scope), |old_scope, this| {
                this.check_lifetime_defs(old_scope, &trait_ref.bound_lifetimes);
                for lifetime in &trait_ref.bound_lifetimes {
                    this.visit_lifetime_def(lifetime);
                }
                this.visit_trait_ref(&trait_ref.trait_ref)
            })
        } else {
            self.visit_trait_ref(&trait_ref.trait_ref)
        }
    }

    fn visit_trait_ref(&mut self, trait_ref: &ast::TraitRef) {
        self.visit_path(&trait_ref.path, trait_ref.ref_id);
    }
}

impl<'a> LifetimeContext<'a> {
    fn with<F>(&mut self, wrap_scope: ScopeChain, f: F) where
        F: FnOnce(Scope, &mut LifetimeContext),
    {
        let LifetimeContext {sess, ref mut named_region_map, ..} = *self;
        let mut this = LifetimeContext {
            sess: sess,
            named_region_map: *named_region_map,
            scope: &wrap_scope,
            def_map: self.def_map,
            trait_ref_hack: self.trait_ref_hack,
        };
        debug!("entering scope {:?}", this.scope);
        f(self.scope, &mut this);
        debug!("exiting scope {:?}", this.scope);
    }

    /// Visits self by adding a scope and handling recursive walk over the contents with `walk`.
    ///
    /// Handles visiting fns and methods. These are a bit complicated because we must distinguish
    /// early- vs late-bound lifetime parameters. We do this by checking which lifetimes appear
    /// within type bounds; those are early bound lifetimes, and the rest are late bound.
    ///
    /// For example:
    ///
    ///    fn foo<'a,'b,'c,T:Trait<'b>>(...)
    ///
    /// Here `'a` and `'c` are late bound but `'b` is early bound. Note that early- and late-bound
    /// lifetimes may be interspersed together.
    ///
    /// If early bound lifetimes are present, we separate them into their own list (and likewise
    /// for late bound). They will be numbered sequentially, starting from the lowest index that is
    /// already in scope (for a fn item, that will be 0, but for a method it might not be). Late
    /// bound lifetimes are resolved by name and associated with a binder id (`binder_id`), so the
    /// ordering is not important there.
    fn visit_early_late<F>(&mut self,
                           early_space: subst::ParamSpace,
                           generics: &ast::Generics,
                           walk: F) where
        F: FnOnce(&mut LifetimeContext),
    {
        let referenced_idents = early_bound_lifetime_names(generics);

        debug!("visit_early_late: referenced_idents={:?}",
               referenced_idents);

        let (early, late): (Vec<_>, _) = generics.lifetimes.iter().cloned().partition(
            |l| referenced_idents.iter().any(|&i| i == l.lifetime.name));

        self.with(EarlyScope(early_space, &early, self.scope), move |old_scope, this| {
            this.with(LateScope(&late, this.scope), move |_, this| {
                this.check_lifetime_defs(old_scope, &generics.lifetimes);
                walk(this);
            });
        });
    }

    fn resolve_lifetime_ref(&mut self, lifetime_ref: &ast::Lifetime) {
        // Walk up the scope chain, tracking the number of fn scopes
        // that we pass through, until we find a lifetime with the
        // given name or we run out of scopes. If we encounter a code
        // block, then the lifetime is not bound but free, so switch
        // over to `resolve_free_lifetime_ref()` to complete the
        // search.
        let mut late_depth = 0;
        let mut scope = self.scope;
        loop {
            match *scope {
                BlockScope(blk_scope, s) => {
                    return self.resolve_free_lifetime_ref(blk_scope, lifetime_ref, s);
                }

                RootScope => {
                    break;
                }

                EarlyScope(space, lifetimes, s) => {
                    match search_lifetimes(lifetimes, lifetime_ref) {
                        Some((index, lifetime_def)) => {
                            let decl_id = lifetime_def.id;
                            let def = DefEarlyBoundRegion(space, index, decl_id);
                            self.insert_lifetime(lifetime_ref, def);
                            return;
                        }
                        None => {
                            scope = s;
                        }
                    }
                }

                LateScope(lifetimes, s) => {
                    match search_lifetimes(lifetimes, lifetime_ref) {
                        Some((_index, lifetime_def)) => {
                            let decl_id = lifetime_def.id;
                            let debruijn = ty::DebruijnIndex::new(late_depth + 1);
                            let def = DefLateBoundRegion(debruijn, decl_id);
                            self.insert_lifetime(lifetime_ref, def);
                            return;
                        }

                        None => {
                            late_depth += 1;
                            scope = s;
                        }
                    }
                }
            }
        }

        self.unresolved_lifetime_ref(lifetime_ref);
    }

    fn resolve_free_lifetime_ref(&mut self,
                                 scope_data: region::DestructionScopeData,
                                 lifetime_ref: &ast::Lifetime,
                                 scope: Scope) {
        debug!("resolve_free_lifetime_ref \
                scope_data: {:?} lifetime_ref: {:?} scope: {:?}",
               scope_data, lifetime_ref, scope);

        // Walk up the scope chain, tracking the outermost free scope,
        // until we encounter a scope that contains the named lifetime
        // or we run out of scopes.
        let mut scope_data = scope_data;
        let mut scope = scope;
        let mut search_result = None;
        loop {
            debug!("resolve_free_lifetime_ref \
                    scope_data: {:?} scope: {:?} search_result: {:?}",
                   scope_data, scope, search_result);
            match *scope {
                BlockScope(blk_scope_data, s) => {
                    scope_data = blk_scope_data;
                    scope = s;
                }

                RootScope => {
                    break;
                }

                EarlyScope(_, lifetimes, s) |
                LateScope(lifetimes, s) => {
                    search_result = search_lifetimes(lifetimes, lifetime_ref);
                    if search_result.is_some() {
                        break;
                    }
                    scope = s;
                }
            }
        }

        match search_result {
            Some((_depth, lifetime)) => {
                let def = DefFreeRegion(scope_data, lifetime.id);
                self.insert_lifetime(lifetime_ref, def);
            }

            None => {
                self.unresolved_lifetime_ref(lifetime_ref);
            }
        }

    }

    fn unresolved_lifetime_ref(&self, lifetime_ref: &ast::Lifetime) {
        span_err!(self.sess, lifetime_ref.span, E0261,
            "use of undeclared lifetime name `{}`",
                    token::get_name(lifetime_ref.name));
    }

    fn check_lifetime_defs(&mut self, old_scope: Scope, lifetimes: &Vec<ast::LifetimeDef>) {
        for i in 0..lifetimes.len() {
            let lifetime_i = &lifetimes[i];

            let special_idents = [special_idents::static_lifetime];
            for lifetime in lifetimes {
                if special_idents.iter().any(|&i| i.name == lifetime.lifetime.name) {
                    span_err!(self.sess, lifetime.lifetime.span, E0262,
                        "illegal lifetime parameter name: `{}`",
                                token::get_name(lifetime.lifetime.name));
                }
            }

            // It is a hard error to shadow a lifetime within the same scope.
            for j in i + 1..lifetimes.len() {
                let lifetime_j = &lifetimes[j];

                if lifetime_i.lifetime.name == lifetime_j.lifetime.name {
                    span_err!(self.sess, lifetime_j.lifetime.span, E0263,
                        "lifetime name `{}` declared twice in \
                                the same scope",
                                token::get_name(lifetime_j.lifetime.name));
                }
            }

            // It is a soft error to shadow a lifetime within a parent scope.
            self.check_lifetime_def_for_shadowing(old_scope, &lifetime_i.lifetime);

            for bound in &lifetime_i.bounds {
                self.resolve_lifetime_ref(bound);
            }
        }
    }

    fn check_lifetime_def_for_shadowing(&self,
                                        mut old_scope: Scope,
                                        lifetime: &ast::Lifetime)
    {
        loop {
            match *old_scope {
                BlockScope(_, s) => {
                    old_scope = s;
                }

                RootScope => {
                    return;
                }

                EarlyScope(_, lifetimes, s) |
                LateScope(lifetimes, s) => {
                    if let Some((_, lifetime_def)) = search_lifetimes(lifetimes, lifetime) {
                        self.sess.span_warn(
                            lifetime.span,
                            &format!("lifetime name `{}` shadows another \
                                     lifetime name that is already in scope",
                                     token::get_name(lifetime.name)));
                        self.sess.span_note(
                            lifetime_def.span,
                            &format!("shadowed lifetime `{}` declared here",
                                     token::get_name(lifetime.name)));
                        self.sess.span_note(
                            lifetime.span,
                            "shadowed lifetimes are deprecated \
                             and will become a hard error before 1.0");
                        return;
                    }

                    old_scope = s;
                }
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

        debug!("lifetime_ref={:?} id={:?} resolved to {:?}",
                lifetime_to_string(lifetime_ref),
                lifetime_ref.id,
                def);
        self.named_region_map.insert(lifetime_ref.id, def);
    }
}

fn search_lifetimes<'a>(lifetimes: &'a Vec<ast::LifetimeDef>,
                    lifetime_ref: &ast::Lifetime)
                    -> Option<(u32, &'a ast::Lifetime)> {
    for (i, lifetime_decl) in lifetimes.iter().enumerate() {
        if lifetime_decl.lifetime.name == lifetime_ref.name {
            return Some((i as u32, &lifetime_decl.lifetime));
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
        .cloned()
        .collect()
}

/// Given a set of generic declarations, returns a list of names containing all early bound
/// lifetime names for those generics. (In fact, this list may also contain other names.)
fn early_bound_lifetime_names(generics: &ast::Generics) -> Vec<ast::Name> {
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
        for ty_param in &*generics.ty_params {
            visit::walk_ty_param_bounds_helper(&mut collector, &ty_param.bounds);
        }
        for predicate in &generics.where_clause.predicates {
            match predicate {
                &ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate{ref bounds,
                                                                              ref bounded_ty,
                                                                              ..}) => {
                    collector.visit_ty(&**bounded_ty);
                    visit::walk_ty_param_bounds_helper(&mut collector, bounds);
                }
                &ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate{ref lifetime,
                                                                                ref bounds,
                                                                                ..}) => {
                    collector.visit_lifetime_ref(lifetime);

                    for bound in bounds {
                        collector.visit_lifetime_ref(bound);
                    }
                }
                &ast::WherePredicate::EqPredicate(_) => unimplemented!()
            }
        }
    }

    // Any lifetime that either has a bound or is referenced by a
    // bound is early.
    for lifetime_def in &generics.lifetimes {
        if !lifetime_def.bounds.is_empty() {
            shuffle(&mut early_bound, &mut late_bound,
                    lifetime_def.lifetime.name);
            for bound in &lifetime_def.bounds {
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

impl<'a> fmt::Debug for ScopeChain<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            EarlyScope(space, defs, _) => write!(fmt, "EarlyScope({:?}, {:?})", space, defs),
            LateScope(defs, _) => write!(fmt, "LateScope({:?})", defs),
            BlockScope(id, _) => write!(fmt, "BlockScope({:?})", id),
            RootScope => write!(fmt, "RootScope"),
        }
    }
}
