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

use dep_graph::DepNode;
use hir::map::Map;
use session::Session;
use hir::def::Def;
use hir::def_id::DefId;
use middle::region;
use ty;
use std::mem::replace;
use syntax::ast;
use syntax::symbol::keywords;
use syntax_pos::Span;
use util::nodemap::{NodeMap, FxHashSet, FxHashMap};

use hir;
use hir::intravisit::{self, Visitor, NestedVisitorMap};

#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, Debug)]
pub enum Region {
    Static,
    EarlyBound(/* index */ u32, /* lifetime decl */ ast::NodeId),
    LateBound(ty::DebruijnIndex, /* lifetime decl */ ast::NodeId),
    Free(region::CallSiteScopeData, /* lifetime decl */ ast::NodeId),
}

impl Region {
    fn early(index: &mut u32, def: &hir::LifetimeDef) -> (ast::Name, Region) {
        let i = *index;
        *index += 1;
        (def.lifetime.name, Region::EarlyBound(i, def.lifetime.id))
    }

    fn late(def: &hir::LifetimeDef) -> (ast::Name, Region) {
        let depth = ty::DebruijnIndex::new(1);
        (def.lifetime.name, Region::LateBound(depth, def.lifetime.id))
    }

    fn id(&self) -> Option<ast::NodeId> {
        match *self {
            Region::Static => None,
            Region::EarlyBound(_, id) |
            Region::LateBound(_, id) |
            Region::Free(_, id) => Some(id)
        }
    }

    fn shifted(self, amount: u32) -> Region {
        match self {
            Region::LateBound(depth, id) => {
                Region::LateBound(depth.shifted(amount), id)
            }
            _ => self
        }
    }
}

// Maps the id of each lifetime reference to the lifetime decl
// that it corresponds to.
pub struct NamedRegionMap {
    // maps from every use of a named (not anonymous) lifetime to a
    // `Region` describing how that region is bound
    pub defs: NodeMap<Region>,

    // the set of lifetime def ids that are late-bound; late-bound ids
    // are named regions appearing in fn arguments that do not appear
    // in where-clauses
    pub late_bound: NodeMap<ty::Issue32330>,
}

struct LifetimeContext<'a, 'tcx: 'a> {
    sess: &'a Session,
    hir_map: &'a Map<'tcx>,
    map: &'a mut NamedRegionMap,
    scope: ScopeRef<'a>,
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

    // List of labels in the function/method currently under analysis.
    labels_in_fn: Vec<(ast::Name, Span)>,
}

#[derive(Debug)]
enum Scope<'a> {
    /// Declares lifetimes, and each can be early-bound or late-bound.
    /// The `DebruijnIndex` of late-bound lifetimes starts at `1` and
    /// it should be shifted by the number of `Binder`s in between the
    /// declaration `Binder` and the location it's referenced from.
    Binder {
        lifetimes: FxHashMap<ast::Name, Region>,
        s: ScopeRef<'a>
    },

    /// Lifetimes introduced by a fn are scoped to the call-site for that fn,
    /// if this is a fn body, otherwise the original definitions are used.
    Body {
        id: hir::BodyId,
        s: ScopeRef<'a>
    },

    Root
}

type ScopeRef<'a> = &'a Scope<'a>;

const ROOT_SCOPE: ScopeRef<'static> = &Scope::Root;

pub fn krate(sess: &Session,
             hir_map: &Map)
             -> Result<NamedRegionMap, usize> {
    let _task = hir_map.dep_graph.in_task(DepNode::ResolveLifetimes);
    let krate = hir_map.krate();
    let mut map = NamedRegionMap {
        defs: NodeMap(),
        late_bound: NodeMap(),
    };
    sess.track_errors(|| {
        let mut visitor = LifetimeContext {
            sess: sess,
            hir_map: hir_map,
            map: &mut map,
            scope: ROOT_SCOPE,
            trait_ref_hack: false,
            labels_in_fn: vec![],
        };
        for (_, item) in &krate.items {
            visitor.visit_item(item);
        }
    })?;
    Ok(map)
}

impl<'a, 'tcx> Visitor<'tcx> for LifetimeContext<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(self.hir_map)
    }

    // We want to nest trait/impl items in their parent, but nothing else.
    fn visit_nested_item(&mut self, _: hir::ItemId) {}

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        // Each body has their own set of labels, save labels.
        let saved = replace(&mut self.labels_in_fn, vec![]);
        let body = self.hir_map.body(body);
        extract_labels(self, body);
        self.with(Scope::Body { id: body.id(), s: self.scope }, |_, this| {
            this.visit_body(body);
        });
        replace(&mut self.labels_in_fn, saved);
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        match item.node {
            hir::ItemFn(ref decl, _, _, _, ref generics, _) => {
                self.visit_early_late(item.id, None, decl, generics, |this| {
                    intravisit::walk_item(this, item);
                });
            }
            hir::ItemExternCrate(_) |
            hir::ItemUse(..) |
            hir::ItemMod(..) |
            hir::ItemDefaultImpl(..) |
            hir::ItemForeignMod(..) |
            hir::ItemStatic(..) |
            hir::ItemConst(..) => {
                // These sorts of items have no lifetime parameters at all.
                intravisit::walk_item(self, item);
            }
            hir::ItemTy(_, ref generics) |
            hir::ItemEnum(_, ref generics) |
            hir::ItemStruct(_, ref generics) |
            hir::ItemUnion(_, ref generics) |
            hir::ItemTrait(_, ref generics, ..) |
            hir::ItemImpl(_, _, ref generics, ..) => {
                // These kinds of items have only early bound lifetime parameters.
                let mut index = if let hir::ItemTrait(..) = item.node {
                    1 // Self comes before lifetimes
                } else {
                    0
                };
                let lifetimes = generics.lifetimes.iter().map(|def| {
                    Region::early(&mut index, def)
                }).collect();
                let scope = Scope::Binder {
                    lifetimes: lifetimes,
                    s: ROOT_SCOPE
                };
                self.with(scope, |old_scope, this| {
                    this.check_lifetime_defs(old_scope, &generics.lifetimes);
                    intravisit::walk_item(this, item);
                });
            }
        }
    }

    fn visit_foreign_item(&mut self, item: &'tcx hir::ForeignItem) {
        match item.node {
            hir::ForeignItemFn(ref decl, _, ref generics) => {
                self.visit_early_late(item.id, None, decl, generics, |this| {
                    intravisit::walk_foreign_item(this, item);
                })
            }
            hir::ForeignItemStatic(..) => {
                intravisit::walk_foreign_item(self, item);
            }
        }
    }

    fn visit_ty(&mut self, ty: &'tcx hir::Ty) {
        match ty.node {
            hir::TyBareFn(ref c) => {
                let scope = Scope::Binder {
                    lifetimes: c.lifetimes.iter().map(Region::late).collect(),
                    s: self.scope
                };
                self.with(scope, |old_scope, this| {
                    // a bare fn has no bounds, so everything
                    // contained within is scoped within its binder.
                    this.check_lifetime_defs(old_scope, &c.lifetimes);
                    intravisit::walk_ty(this, ty);
                });
            }
            hir::TyPath(hir::QPath::Resolved(None, ref path)) => {
                // if this path references a trait, then this will resolve to
                // a trait ref, which introduces a binding scope.
                match path.def {
                    Def::Trait(..) => {
                        let scope = Scope::Binder {
                            lifetimes: FxHashMap(),
                            s: self.scope
                        };
                        self.with(scope, |_, this| {
                            this.visit_path(path, ty.id);
                        });
                    }
                    _ => {
                        intravisit::walk_ty(self, ty);
                    }
                }
            }
            _ => {
                intravisit::walk_ty(self, ty)
            }
        }
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem) {
        if let hir::TraitItemKind::Method(ref sig, _) = trait_item.node {
            self.visit_early_late(
                trait_item.id,
                Some(self.hir_map.get_parent(trait_item.id)),
                &sig.decl, &sig.generics,
                |this| intravisit::walk_trait_item(this, trait_item))
        } else {
            intravisit::walk_trait_item(self, trait_item);
        }
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem) {
        if let hir::ImplItemKind::Method(ref sig, _) = impl_item.node {
            self.visit_early_late(
                impl_item.id,
                Some(self.hir_map.get_parent(impl_item.id)),
                &sig.decl, &sig.generics,
                |this| intravisit::walk_impl_item(this, impl_item))
        } else {
            intravisit::walk_impl_item(self, impl_item);
        }
    }

    fn visit_lifetime(&mut self, lifetime_ref: &'tcx hir::Lifetime) {
        if lifetime_ref.is_elided() {
            return;
        }
        if lifetime_ref.name == keywords::StaticLifetime.name() {
            self.insert_lifetime(lifetime_ref, Region::Static);
            return;
        }
        self.resolve_lifetime_ref(lifetime_ref);
    }

    fn visit_generics(&mut self, generics: &'tcx hir::Generics) {
        for ty_param in generics.ty_params.iter() {
            walk_list!(self, visit_ty_param_bound, &ty_param.bounds);
            if let Some(ref ty) = ty_param.default {
                self.visit_ty(&ty);
            }
        }
        for predicate in &generics.where_clause.predicates {
            match predicate {
                &hir::WherePredicate::BoundPredicate(hir::WhereBoundPredicate{ ref bounded_ty,
                                                                               ref bounds,
                                                                               ref bound_lifetimes,
                                                                               .. }) => {
                    if !bound_lifetimes.is_empty() {
                        self.trait_ref_hack = true;
                        let scope = Scope::Binder {
                            lifetimes: bound_lifetimes.iter().map(Region::late).collect(),
                            s: self.scope
                        };
                        let result = self.with(scope, |old_scope, this| {
                            this.check_lifetime_defs(old_scope, bound_lifetimes);
                            this.visit_ty(&bounded_ty);
                            walk_list!(this, visit_ty_param_bound, bounds);
                        });
                        self.trait_ref_hack = false;
                        result
                    } else {
                        self.visit_ty(&bounded_ty);
                        walk_list!(self, visit_ty_param_bound, bounds);
                    }
                }
                &hir::WherePredicate::RegionPredicate(hir::WhereRegionPredicate{ref lifetime,
                                                                                ref bounds,
                                                                                .. }) => {

                    self.visit_lifetime(lifetime);
                    for bound in bounds {
                        self.visit_lifetime(bound);
                    }
                }
                &hir::WherePredicate::EqPredicate(hir::WhereEqPredicate{ref lhs_ty,
                                                                        ref rhs_ty,
                                                                        .. }) => {
                    self.visit_ty(lhs_ty);
                    self.visit_ty(rhs_ty);
                }
            }
        }
    }

    fn visit_poly_trait_ref(&mut self,
                            trait_ref: &'tcx hir::PolyTraitRef,
                            _modifier: &'tcx hir::TraitBoundModifier) {
        debug!("visit_poly_trait_ref trait_ref={:?}", trait_ref);

        if !self.trait_ref_hack || !trait_ref.bound_lifetimes.is_empty() {
            if self.trait_ref_hack {
                span_err!(self.sess, trait_ref.span, E0316,
                          "nested quantification of lifetimes");
            }
            let scope = Scope::Binder {
                lifetimes: trait_ref.bound_lifetimes.iter().map(Region::late).collect(),
                s: self.scope
            };
            self.with(scope, |old_scope, this| {
                this.check_lifetime_defs(old_scope, &trait_ref.bound_lifetimes);
                for lifetime in &trait_ref.bound_lifetimes {
                    this.visit_lifetime_def(lifetime);
                }
                intravisit::walk_path(this, &trait_ref.trait_ref.path)
            })
        } else {
            self.visit_trait_ref(&trait_ref.trait_ref)
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
enum ShadowKind { Label, Lifetime }
struct Original { kind: ShadowKind, span: Span }
struct Shadower { kind: ShadowKind, span: Span }

fn original_label(span: Span) -> Original {
    Original { kind: ShadowKind::Label, span: span }
}
fn shadower_label(span: Span) -> Shadower {
    Shadower { kind: ShadowKind::Label, span: span }
}
fn original_lifetime(span: Span) -> Original {
    Original { kind: ShadowKind::Lifetime, span: span }
}
fn shadower_lifetime(l: &hir::Lifetime) -> Shadower {
    Shadower { kind: ShadowKind::Lifetime, span: l.span }
}

impl ShadowKind {
    fn desc(&self) -> &'static str {
        match *self {
            ShadowKind::Label => "label",
            ShadowKind::Lifetime => "lifetime",
        }
    }
}

fn signal_shadowing_problem(sess: &Session, name: ast::Name, orig: Original, shadower: Shadower) {
    let mut err = if let (ShadowKind::Lifetime, ShadowKind::Lifetime) = (orig.kind, shadower.kind) {
        // lifetime/lifetime shadowing is an error
        struct_span_err!(sess, shadower.span, E0496,
                         "{} name `{}` shadows a \
                          {} name that is already in scope",
                         shadower.kind.desc(), name, orig.kind.desc())
    } else {
        // shadowing involving a label is only a warning, due to issues with
        // labels and lifetimes not being macro-hygienic.
        sess.struct_span_warn(shadower.span,
                              &format!("{} name `{}` shadows a \
                                        {} name that is already in scope",
                                       shadower.kind.desc(), name, orig.kind.desc()))
    };
    err.span_label(orig.span, &"first declared here");
    err.span_label(shadower.span,
                   &format!("lifetime {} already in scope", name));
    err.emit();
}

// Adds all labels in `b` to `ctxt.labels_in_fn`, signalling a warning
// if one of the label shadows a lifetime or another label.
fn extract_labels(ctxt: &mut LifetimeContext, body: &hir::Body) {
    struct GatherLabels<'a, 'tcx: 'a> {
        sess: &'a Session,
        hir_map: &'a Map<'tcx>,
        scope: ScopeRef<'a>,
        labels_in_fn: &'a mut Vec<(ast::Name, Span)>,
    }

    let mut gather = GatherLabels {
        sess: ctxt.sess,
        hir_map: ctxt.hir_map,
        scope: ctxt.scope,
        labels_in_fn: &mut ctxt.labels_in_fn,
    };
    gather.visit_body(body);

    impl<'v, 'a, 'tcx> Visitor<'v> for GatherLabels<'a, 'tcx> {
        fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'v> {
            NestedVisitorMap::None
        }

        fn visit_expr(&mut self, ex: &hir::Expr) {
            if let Some((label, label_span)) = expression_label(ex) {
                for &(prior, prior_span) in &self.labels_in_fn[..] {
                    // FIXME (#24278): non-hygienic comparison
                    if label == prior {
                        signal_shadowing_problem(self.sess,
                                                 label,
                                                 original_label(prior_span),
                                                 shadower_label(label_span));
                    }
                }

                check_if_label_shadows_lifetime(self.sess,
                                                self.hir_map,
                                                self.scope,
                                                label,
                                                label_span);

                self.labels_in_fn.push((label, label_span));
            }
            intravisit::walk_expr(self, ex)
        }

        fn visit_item(&mut self, _: &hir::Item) {
            // do not recurse into items defined in the block
        }
    }

    fn expression_label(ex: &hir::Expr) -> Option<(ast::Name, Span)> {
        match ex.node {
            hir::ExprWhile(.., Some(label)) |
            hir::ExprLoop(_, Some(label), _) => Some((label.node, label.span)),
            _ => None,
        }
    }

    fn check_if_label_shadows_lifetime<'a>(sess: &'a Session,
                                           hir_map: &Map,
                                           mut scope: ScopeRef<'a>,
                                           label: ast::Name,
                                           label_span: Span) {
        loop {
            match *scope {
                Scope::Body { s, .. } => { scope = s; }
                Scope::Root => { return; }

                Scope::Binder { ref lifetimes, s } => {
                    // FIXME (#24278): non-hygienic comparison
                    if let Some(def) = lifetimes.get(&label) {
                        signal_shadowing_problem(
                            sess,
                            label,
                            original_lifetime(hir_map.span(def.id().unwrap())),
                            shadower_label(label_span));
                        return;
                    }
                    scope = s;
                }
            }
        }
    }
}

impl<'a, 'tcx> LifetimeContext<'a, 'tcx> {
    // FIXME(#37666) this works around a limitation in the region inferencer
    fn hack<F>(&mut self, f: F) where
        F: for<'b> FnOnce(&mut LifetimeContext<'b, 'tcx>),
    {
        f(self)
    }

    fn with<F>(&mut self, wrap_scope: Scope, f: F) where
        F: for<'b> FnOnce(ScopeRef, &mut LifetimeContext<'b, 'tcx>),
    {
        let LifetimeContext {sess, hir_map, ref mut map, ..} = *self;
        let labels_in_fn = replace(&mut self.labels_in_fn, vec![]);
        let mut this = LifetimeContext {
            sess: sess,
            hir_map: hir_map,
            map: *map,
            scope: &wrap_scope,
            trait_ref_hack: self.trait_ref_hack,
            labels_in_fn: labels_in_fn,
        };
        debug!("entering scope {:?}", this.scope);
        f(self.scope, &mut this);
        debug!("exiting scope {:?}", this.scope);
        self.labels_in_fn = this.labels_in_fn;
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
                           fn_id: ast::NodeId,
                           parent_id: Option<ast::NodeId>,
                           decl: &'tcx hir::FnDecl,
                           generics: &'tcx hir::Generics,
                           walk: F) where
        F: for<'b, 'c> FnOnce(&'b mut LifetimeContext<'c, 'tcx>),
    {
        let fn_def_id = self.hir_map.local_def_id(fn_id);
        insert_late_bound_lifetimes(self.map,
                                    fn_def_id,
                                    decl,
                                    generics);

        // Find the start of nested early scopes, e.g. in methods.
        let mut index = 0;
        if let Some(parent_id) = parent_id {
            let parent = self.hir_map.expect_item(parent_id);
            if let hir::ItemTrait(..) = parent.node {
                index += 1; // Self comes first.
            }
            match parent.node {
                hir::ItemTrait(_, ref generics, ..) |
                hir::ItemImpl(_, _, ref generics, ..) => {
                    index += (generics.lifetimes.len() + generics.ty_params.len()) as u32;
                }
                _ => {}
            }
        }

        let lifetimes = generics.lifetimes.iter().map(|def| {
            if self.map.late_bound.contains_key(&def.lifetime.id) {
                Region::late(def)
            } else {
                Region::early(&mut index, def)
            }
        }).collect();

        let scope = Scope::Binder {
            lifetimes: lifetimes,
            s: self.scope
        };
        self.with(scope, move |old_scope, this| {
            this.check_lifetime_defs(old_scope, &generics.lifetimes);
            this.hack(walk); // FIXME(#37666) workaround in place of `walk(this)`
        });
    }

    fn resolve_lifetime_ref(&mut self, lifetime_ref: &hir::Lifetime) {
        // Walk up the scope chain, tracking the number of fn scopes
        // that we pass through, until we find a lifetime with the
        // given name or we run out of scopes.
        // search.
        let mut late_depth = 0;
        let mut scope = self.scope;
        let mut outermost_body = None;
        let result = loop {
            match *scope {
                Scope::Body { id, s } => {
                    outermost_body = Some(id);
                    scope = s;
                }

                Scope::Root => {
                    break None;
                }

                Scope::Binder { ref lifetimes, s } => {
                    if let Some(&def) = lifetimes.get(&lifetime_ref.name) {
                        break Some(def.shifted(late_depth));
                    } else {
                        late_depth += 1;
                        scope = s;
                    }
                }
            }
        };

        if let Some(mut def) = result {
            if let Some(body_id) = outermost_body {
                let fn_id = self.hir_map.body_owner(body_id);
                let scope_data = region::CallSiteScopeData {
                    fn_id: fn_id, body_id: body_id.node_id
                };
                match self.hir_map.get(fn_id) {
                    hir::map::NodeItem(&hir::Item {
                        node: hir::ItemFn(..), ..
                    }) |
                    hir::map::NodeTraitItem(&hir::TraitItem {
                        node: hir::TraitItemKind::Method(..), ..
                    }) |
                    hir::map::NodeImplItem(&hir::ImplItem {
                        node: hir::ImplItemKind::Method(..), ..
                    }) => {
                        def = Region::Free(scope_data, def.id().unwrap());
                    }
                    _ => {}
                }
            }
            self.insert_lifetime(lifetime_ref, def);
        } else {
            struct_span_err!(self.sess, lifetime_ref.span, E0261,
                "use of undeclared lifetime name `{}`", lifetime_ref.name)
                .span_label(lifetime_ref.span, &format!("undeclared lifetime"))
                .emit();
        }
    }

    fn check_lifetime_defs(&mut self, old_scope: ScopeRef, lifetimes: &[hir::LifetimeDef]) {
        for i in 0..lifetimes.len() {
            let lifetime_i = &lifetimes[i];

            for lifetime in lifetimes {
                if lifetime.lifetime.name == keywords::StaticLifetime.name() {
                    let lifetime = lifetime.lifetime;
                    let mut err = struct_span_err!(self.sess, lifetime.span, E0262,
                                  "invalid lifetime parameter name: `{}`", lifetime.name);
                    err.span_label(lifetime.span,
                                   &format!("{} is a reserved lifetime name", lifetime.name));
                    err.emit();
                }
            }

            // It is a hard error to shadow a lifetime within the same scope.
            for j in i + 1..lifetimes.len() {
                let lifetime_j = &lifetimes[j];

                if lifetime_i.lifetime.name == lifetime_j.lifetime.name {
                    struct_span_err!(self.sess, lifetime_j.lifetime.span, E0263,
                                     "lifetime name `{}` declared twice in the same scope",
                                     lifetime_j.lifetime.name)
                        .span_label(lifetime_j.lifetime.span,
                                    &format!("declared twice"))
                        .span_label(lifetime_i.lifetime.span,
                                   &format!("previous declaration here"))
                        .emit();
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
                                        mut old_scope: ScopeRef,
                                        lifetime: &hir::Lifetime)
    {
        for &(label, label_span) in &self.labels_in_fn {
            // FIXME (#24278): non-hygienic comparison
            if lifetime.name == label {
                signal_shadowing_problem(self.sess,
                                         lifetime.name,
                                         original_label(label_span),
                                         shadower_lifetime(&lifetime));
                return;
            }
        }

        loop {
            match *old_scope {
                Scope::Body { s, .. } => {
                    old_scope = s;
                }

                Scope::Root => {
                    return;
                }

                Scope::Binder { ref lifetimes, s } => {
                    if let Some(&def) = lifetimes.get(&lifetime.name) {
                        signal_shadowing_problem(
                            self.sess,
                            lifetime.name,
                            original_lifetime(self.hir_map.span(def.id().unwrap())),
                            shadower_lifetime(&lifetime));
                        return;
                    }

                    old_scope = s;
                }
            }
        }
    }

    fn insert_lifetime(&mut self,
                       lifetime_ref: &hir::Lifetime,
                       def: Region) {
        if lifetime_ref.id == ast::DUMMY_NODE_ID {
            span_bug!(lifetime_ref.span,
                      "lifetime reference not renumbered, \
                       probably a bug in syntax::fold");
        }

        debug!("{} resolved to {:?} span={:?}",
               self.hir_map.node_to_string(lifetime_ref.id),
               def,
               self.sess.codemap().span_to_string(lifetime_ref.span));
        self.map.defs.insert(lifetime_ref.id, def);
    }
}

///////////////////////////////////////////////////////////////////////////

/// Detects late-bound lifetimes and inserts them into
/// `map.late_bound`.
///
/// A region declared on a fn is **late-bound** if:
/// - it is constrained by an argument type;
/// - it does not appear in a where-clause.
///
/// "Constrained" basically means that it appears in any type but
/// not amongst the inputs to a projection.  In other words, `<&'a
/// T as Trait<''b>>::Foo` does not constrain `'a` or `'b`.
fn insert_late_bound_lifetimes(map: &mut NamedRegionMap,
                               fn_def_id: DefId,
                               decl: &hir::FnDecl,
                               generics: &hir::Generics) {
    debug!("insert_late_bound_lifetimes(decl={:?}, generics={:?})", decl, generics);

    let mut constrained_by_input = ConstrainedCollector { regions: FxHashSet() };
    for arg_ty in &decl.inputs {
        constrained_by_input.visit_ty(arg_ty);
    }

    let mut appears_in_output = AllCollector {
        regions: FxHashSet(),
        impl_trait: false
    };
    intravisit::walk_fn_ret_ty(&mut appears_in_output, &decl.output);

    debug!("insert_late_bound_lifetimes: constrained_by_input={:?}",
           constrained_by_input.regions);

    // Walk the lifetimes that appear in where clauses.
    //
    // Subtle point: because we disallow nested bindings, we can just
    // ignore binders here and scrape up all names we see.
    let mut appears_in_where_clause = AllCollector {
        regions: FxHashSet(),
        impl_trait: false
    };
    for ty_param in generics.ty_params.iter() {
        walk_list!(&mut appears_in_where_clause,
                   visit_ty_param_bound,
                   &ty_param.bounds);
    }
    walk_list!(&mut appears_in_where_clause,
               visit_where_predicate,
               &generics.where_clause.predicates);
    for lifetime_def in &generics.lifetimes {
        if !lifetime_def.bounds.is_empty() {
            // `'a: 'b` means both `'a` and `'b` are referenced
            appears_in_where_clause.visit_lifetime_def(lifetime_def);
        }
    }

    debug!("insert_late_bound_lifetimes: appears_in_where_clause={:?}",
           appears_in_where_clause.regions);

    // Late bound regions are those that:
    // - appear in the inputs
    // - do not appear in the where-clauses
    // - are not implicitly captured by `impl Trait`
    for lifetime in &generics.lifetimes {
        let name = lifetime.lifetime.name;

        // appears in the where clauses? early-bound.
        if appears_in_where_clause.regions.contains(&name) { continue; }

        // any `impl Trait` in the return type? early-bound.
        if appears_in_output.impl_trait { continue; }

        // does not appear in the inputs, but appears in the return
        // type? eventually this will be early-bound, but for now we
        // just mark it so we can issue warnings.
        let constrained_by_input = constrained_by_input.regions.contains(&name);
        let appears_in_output = appears_in_output.regions.contains(&name);
        let will_change = !constrained_by_input && appears_in_output;
        let issue_32330 = if will_change {
            ty::Issue32330::WillChange {
                fn_def_id: fn_def_id,
                region_name: name,
            }
        } else {
            ty::Issue32330::WontChange
        };

        debug!("insert_late_bound_lifetimes: \
                lifetime {:?} with id {:?} is late-bound ({:?}",
               lifetime.lifetime.name, lifetime.lifetime.id, issue_32330);

        let prev = map.late_bound.insert(lifetime.lifetime.id, issue_32330);
        assert!(prev.is_none(), "visited lifetime {:?} twice", lifetime.lifetime.id);
    }

    return;

    struct ConstrainedCollector {
        regions: FxHashSet<ast::Name>,
    }

    impl<'v> Visitor<'v> for ConstrainedCollector {
        fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'v> {
            NestedVisitorMap::None
        }

        fn visit_ty(&mut self, ty: &'v hir::Ty) {
            match ty.node {
                hir::TyPath(hir::QPath::Resolved(Some(_), _)) |
                hir::TyPath(hir::QPath::TypeRelative(..)) => {
                    // ignore lifetimes appearing in associated type
                    // projections, as they are not *constrained*
                    // (defined above)
                }

                hir::TyPath(hir::QPath::Resolved(None, ref path)) => {
                    // consider only the lifetimes on the final
                    // segment; I am not sure it's even currently
                    // valid to have them elsewhere, but even if it
                    // is, those would be potentially inputs to
                    // projections
                    if let Some(last_segment) = path.segments.last() {
                        self.visit_path_segment(path.span, last_segment);
                    }
                }

                _ => {
                    intravisit::walk_ty(self, ty);
                }
            }
        }

        fn visit_lifetime(&mut self, lifetime_ref: &'v hir::Lifetime) {
            self.regions.insert(lifetime_ref.name);
        }
    }

    struct AllCollector {
        regions: FxHashSet<ast::Name>,
        impl_trait: bool
    }

    impl<'v> Visitor<'v> for AllCollector {
        fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'v> {
            NestedVisitorMap::None
        }

        fn visit_lifetime(&mut self, lifetime_ref: &'v hir::Lifetime) {
            self.regions.insert(lifetime_ref.name);
        }

        fn visit_ty(&mut self, ty: &hir::Ty) {
            if let hir::TyImplTrait(_) = ty.node {
                self.impl_trait = true;
            }
            intravisit::walk_ty(self, ty);
        }
    }
}
