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

use hir::map::Map;
use hir::def::Def;
use hir::def_id::DefId;
use middle::cstore::CrateStore;
use session::Session;
use ty;

use std::cell::Cell;
use std::mem::replace;
use syntax::ast;
use syntax::attr;
use syntax::ptr::P;
use syntax_pos::Span;
use errors::DiagnosticBuilder;
use util::common::ErrorReported;
use util::nodemap::{NodeMap, NodeSet, FxHashSet, FxHashMap, DefIdMap};
use rustc_back::slice;

use hir;
use hir::intravisit::{self, Visitor, NestedVisitorMap};

#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, Debug)]
pub enum Region {
    Static,
    EarlyBound(/* index */ u32, /* lifetime decl */ DefId),
    LateBound(ty::DebruijnIndex, /* lifetime decl */ DefId),
    LateBoundAnon(ty::DebruijnIndex, /* anon index */ u32),
    Free(DefId, /* lifetime decl */ DefId),
}

impl Region {
    fn early(hir_map: &Map, index: &mut u32, def: &hir::LifetimeDef) -> (ast::Name, Region) {
        let i = *index;
        *index += 1;
        let def_id = hir_map.local_def_id(def.lifetime.id);
        (def.lifetime.name, Region::EarlyBound(i, def_id))
    }

    fn late(hir_map: &Map, def: &hir::LifetimeDef) -> (ast::Name, Region) {
        let depth = ty::DebruijnIndex::new(1);
        let def_id = hir_map.local_def_id(def.lifetime.id);
        (def.lifetime.name, Region::LateBound(depth, def_id))
    }

    fn late_anon(index: &Cell<u32>) -> Region {
        let i = index.get();
        index.set(i + 1);
        let depth = ty::DebruijnIndex::new(1);
        Region::LateBoundAnon(depth, i)
    }

    fn id(&self) -> Option<DefId> {
        match *self {
            Region::Static |
            Region::LateBoundAnon(..) => None,

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
            Region::LateBoundAnon(depth, index) => {
                Region::LateBoundAnon(depth.shifted(amount), index)
            }
            _ => self
        }
    }

    fn from_depth(self, depth: u32) -> Region {
        match self {
            Region::LateBound(debruijn, id) => {
                Region::LateBound(ty::DebruijnIndex {
                    depth: debruijn.depth - (depth - 1)
                }, id)
            }
            Region::LateBoundAnon(debruijn, index) => {
                Region::LateBoundAnon(ty::DebruijnIndex {
                    depth: debruijn.depth - (depth - 1)
                }, index)
            }
            _ => self
        }
    }

    fn subst(self, params: &[hir::Lifetime], map: &NamedRegionMap)
             -> Option<Region> {
        if let Region::EarlyBound(index, _) = self {
            params.get(index as usize).and_then(|lifetime| {
                map.defs.get(&lifetime.id).cloned()
            })
        } else {
            Some(self)
        }
    }
}

/// A set containing, at most, one known element.
/// If two distinct values are inserted into a set, then it
/// becomes `Many`, which can be used to detect ambiguities.
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Debug)]
pub enum Set1<T> {
    Empty,
    One(T),
    Many
}

impl<T: PartialEq> Set1<T> {
    pub fn insert(&mut self, value: T) {
        if let Set1::Empty = *self {
            *self = Set1::One(value);
            return;
        }
        if let Set1::One(ref old) = *self {
            if *old == value {
                return;
            }
        }
        *self = Set1::Many;
    }
}

pub type ObjectLifetimeDefault = Set1<Region>;

// Maps the id of each lifetime reference to the lifetime decl
// that it corresponds to.
pub struct NamedRegionMap {
    // maps from every use of a named (not anonymous) lifetime to a
    // `Region` describing how that region is bound
    pub defs: NodeMap<Region>,

    // the set of lifetime def ids that are late-bound; a region can
    // be late-bound if (a) it does NOT appear in a where-clause and
    // (b) it DOES appear in the arguments.
    pub late_bound: NodeSet,

    // For each type and trait definition, maps type parameters
    // to the trait object lifetime defaults computed from them.
    pub object_lifetime_defaults: NodeMap<Vec<ObjectLifetimeDefault>>,
}

struct LifetimeContext<'a, 'tcx: 'a> {
    sess: &'a Session,
    cstore: &'a CrateStore,
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

    // Cache for cross-crate per-definition object lifetime defaults.
    xcrate_object_lifetime_defaults: DefIdMap<Vec<ObjectLifetimeDefault>>,
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
    /// Unspecified lifetimes are inferred, unless an elision scope is nested,
    /// e.g. `(&T, fn(&T) -> &T);` becomes `(&'_ T, for<'a> fn(&'a T) -> &'a T)`.
    Body {
        id: hir::BodyId,
        s: ScopeRef<'a>
    },

    /// A scope which either determines unspecified lifetimes or errors
    /// on them (e.g. due to ambiguity). For more details, see `Elide`.
    Elision {
        elide: Elide,
        s: ScopeRef<'a>
    },

    /// Use a specific lifetime (if `Some`) or leave it unset (to be
    /// inferred in a function body or potentially error outside one),
    /// for the default choice of lifetime in a trait object type.
    ObjectLifetimeDefault {
        lifetime: Option<Region>,
        s: ScopeRef<'a>
    },

    Root
}

#[derive(Clone, Debug)]
enum Elide {
    /// Use a fresh anonymous late-bound lifetime each time, by
    /// incrementing the counter to generate sequential indices.
    FreshLateAnon(Cell<u32>),
    /// Always use this one lifetime.
    Exact(Region),
    /// Less or more than one lifetime were found, error on unspecified.
    Error(Vec<ElisionFailureInfo>)
}

#[derive(Clone, Debug)]
struct ElisionFailureInfo {
    /// Where we can find the argument pattern.
    parent: Option<hir::BodyId>,
    /// The index of the argument in the original definition.
    index: usize,
    lifetime_count: usize,
    have_bound_regions: bool
}

type ScopeRef<'a> = &'a Scope<'a>;

const ROOT_SCOPE: ScopeRef<'static> = &Scope::Root;

pub fn krate(sess: &Session,
             cstore: &CrateStore,
             hir_map: &Map)
             -> Result<NamedRegionMap, ErrorReported> {
    let krate = hir_map.krate();
    let mut map = NamedRegionMap {
        defs: NodeMap(),
        late_bound: NodeSet(),
        object_lifetime_defaults: compute_object_lifetime_defaults(sess, hir_map),
    };
    sess.track_errors(|| {
        let mut visitor = LifetimeContext {
            sess,
            cstore,
            hir_map,
            map: &mut map,
            scope: ROOT_SCOPE,
            trait_ref_hack: false,
            labels_in_fn: vec![],
            xcrate_object_lifetime_defaults: DefIdMap(),
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
                self.visit_early_late(None, decl, generics, |this| {
                    intravisit::walk_item(this, item);
                });
            }
            hir::ItemExternCrate(_) |
            hir::ItemUse(..) |
            hir::ItemMod(..) |
            hir::ItemDefaultImpl(..) |
            hir::ItemForeignMod(..) |
            hir::ItemGlobalAsm(..) => {
                // These sorts of items have no lifetime parameters at all.
                intravisit::walk_item(self, item);
            }
            hir::ItemStatic(..) |
            hir::ItemConst(..) => {
                // No lifetime parameters, but implied 'static.
                let scope = Scope::Elision {
                    elide: Elide::Exact(Region::Static),
                    s: ROOT_SCOPE
                };
                self.with(scope, |_, this| intravisit::walk_item(this, item));
            }
            hir::ItemTy(_, ref generics) |
            hir::ItemEnum(_, ref generics) |
            hir::ItemStruct(_, ref generics) |
            hir::ItemUnion(_, ref generics) |
            hir::ItemTrait(_, ref generics, ..) |
            hir::ItemImpl(_, _, _, ref generics, ..) => {
                // These kinds of items have only early bound lifetime parameters.
                let mut index = if let hir::ItemTrait(..) = item.node {
                    1 // Self comes before lifetimes
                } else {
                    0
                };
                let lifetimes = generics.lifetimes.iter().map(|def| {
                    Region::early(self.hir_map, &mut index, def)
                }).collect();
                let scope = Scope::Binder {
                    lifetimes,
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
                self.visit_early_late(None, decl, generics, |this| {
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
                    lifetimes: c.lifetimes.iter().map(|def| {
                            Region::late(self.hir_map, def)
                        }).collect(),
                    s: self.scope
                };
                self.with(scope, |old_scope, this| {
                    // a bare fn has no bounds, so everything
                    // contained within is scoped within its binder.
                    this.check_lifetime_defs(old_scope, &c.lifetimes);
                    intravisit::walk_ty(this, ty);
                });
            }
            hir::TyTraitObject(ref bounds, ref lifetime) => {
                for bound in bounds {
                    self.visit_poly_trait_ref(bound, hir::TraitBoundModifier::None);
                }
                if lifetime.is_elided() {
                    self.resolve_object_lifetime_default(lifetime)
                } else {
                    self.visit_lifetime(lifetime);
                }
            }
            hir::TyRptr(ref lifetime_ref, ref mt) => {
                self.visit_lifetime(lifetime_ref);
                let scope = Scope::ObjectLifetimeDefault {
                    lifetime: self.map.defs.get(&lifetime_ref.id).cloned(),
                    s: self.scope
                };
                self.with(scope, |_, this| this.visit_ty(&mt.ty));
            }
            _ => {
                intravisit::walk_ty(self, ty)
            }
        }
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem) {
        if let hir::TraitItemKind::Method(ref sig, _) = trait_item.node {
            self.visit_early_late(
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
                Some(self.hir_map.get_parent(impl_item.id)),
                &sig.decl, &sig.generics,
                |this| intravisit::walk_impl_item(this, impl_item))
        } else {
            intravisit::walk_impl_item(self, impl_item);
        }
    }

    fn visit_lifetime(&mut self, lifetime_ref: &'tcx hir::Lifetime) {
        if lifetime_ref.is_elided() {
            self.resolve_elided_lifetimes(slice::ref_slice(lifetime_ref));
            return;
        }
        if lifetime_ref.is_static() {
            self.insert_lifetime(lifetime_ref, Region::Static);
            return;
        }
        self.resolve_lifetime_ref(lifetime_ref);
    }

    fn visit_path(&mut self, path: &'tcx hir::Path, _: ast::NodeId) {
        for (i, segment) in path.segments.iter().enumerate() {
            let depth = path.segments.len() - i - 1;
            self.visit_segment_parameters(path.def, depth, &segment.parameters);
        }
    }

    fn visit_fn_decl(&mut self, fd: &'tcx hir::FnDecl) {
        let output = match fd.output {
            hir::DefaultReturn(_) => None,
            hir::Return(ref ty) => Some(ty)
        };
        self.visit_fn_like_elision(&fd.inputs, output);
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
                            lifetimes: bound_lifetimes.iter().map(|def| {
                                    Region::late(self.hir_map, def)
                                }).collect(),
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
                            _modifier: hir::TraitBoundModifier) {
        debug!("visit_poly_trait_ref trait_ref={:?}", trait_ref);

        if !self.trait_ref_hack || !trait_ref.bound_lifetimes.is_empty() {
            if self.trait_ref_hack {
                span_err!(self.sess, trait_ref.span, E0316,
                          "nested quantification of lifetimes");
            }
            let scope = Scope::Binder {
                lifetimes: trait_ref.bound_lifetimes.iter().map(|def| {
                        Region::late(self.hir_map, def)
                    }).collect(),
                s: self.scope
            };
            self.with(scope, |old_scope, this| {
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
    err.span_label(orig.span, "first declared here");
    err.span_label(shadower.span,
                   format!("lifetime {} already in scope", name));
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
                Scope::Body { s, .. } |
                Scope::Elision { s, .. } |
                Scope::ObjectLifetimeDefault { s, .. } => { scope = s; }

                Scope::Root => { return; }

                Scope::Binder { ref lifetimes, s } => {
                    // FIXME (#24278): non-hygienic comparison
                    if let Some(def) = lifetimes.get(&label) {
                        let node_id = hir_map.as_local_node_id(def.id().unwrap())
                                             .unwrap();

                        signal_shadowing_problem(
                            sess,
                            label,
                            original_lifetime(hir_map.span(node_id)),
                            shadower_label(label_span));
                        return;
                    }
                    scope = s;
                }
            }
        }
    }
}

fn compute_object_lifetime_defaults(sess: &Session, hir_map: &Map)
                                    -> NodeMap<Vec<ObjectLifetimeDefault>> {
    let mut map = NodeMap();
    for item in hir_map.krate().items.values() {
        match item.node {
            hir::ItemStruct(_, ref generics) |
            hir::ItemUnion(_, ref generics) |
            hir::ItemEnum(_, ref generics) |
            hir::ItemTy(_, ref generics) |
            hir::ItemTrait(_, ref generics, ..) => {
                let result = object_lifetime_defaults_for_item(hir_map, generics);

                // Debugging aid.
                if attr::contains_name(&item.attrs, "rustc_object_lifetime_default") {
                    let object_lifetime_default_reprs: String =
                        result.iter().map(|set| {
                            match *set {
                                Set1::Empty => "BaseDefault".to_string(),
                                Set1::One(Region::Static) => "'static".to_string(),
                                Set1::One(Region::EarlyBound(i, _)) => {
                                    generics.lifetimes[i as usize].lifetime.name.to_string()
                                }
                                Set1::One(_) => bug!(),
                                Set1::Many => "Ambiguous".to_string(),
                            }
                        }).collect::<Vec<String>>().join(",");
                    sess.span_err(item.span, &object_lifetime_default_reprs);
                }

                map.insert(item.id, result);
            }
            _ => {}
        }
    }
    map
}

/// Scan the bounds and where-clauses on parameters to extract bounds
/// of the form `T:'a` so as to determine the `ObjectLifetimeDefault`
/// for each type parameter.
fn object_lifetime_defaults_for_item(hir_map: &Map, generics: &hir::Generics)
                                     -> Vec<ObjectLifetimeDefault> {
    fn add_bounds(set: &mut Set1<ast::Name>, bounds: &[hir::TyParamBound]) {
        for bound in bounds {
            if let hir::RegionTyParamBound(ref lifetime) = *bound {
                set.insert(lifetime.name);
            }
        }
    }

    generics.ty_params.iter().map(|param| {
        let mut set = Set1::Empty;

        add_bounds(&mut set, &param.bounds);

        let param_def_id = hir_map.local_def_id(param.id);
        for predicate in &generics.where_clause.predicates {
            // Look for `type: ...` where clauses.
            let data = match *predicate {
                hir::WherePredicate::BoundPredicate(ref data) => data,
                _ => continue
            };

            // Ignore `for<'a> type: ...` as they can change what
            // lifetimes mean (although we could "just" handle it).
            if !data.bound_lifetimes.is_empty() {
                continue;
            }

            let def = match data.bounded_ty.node {
                hir::TyPath(hir::QPath::Resolved(None, ref path)) => path.def,
                _ => continue
            };

            if def == Def::TyParam(param_def_id) {
                add_bounds(&mut set, &data.bounds);
            }
        }

        match set {
            Set1::Empty => Set1::Empty,
            Set1::One(name) => {
                if name == "'static" {
                    Set1::One(Region::Static)
                } else {
                    generics.lifetimes.iter().enumerate().find(|&(_, def)| {
                        def.lifetime.name == name
                    }).map_or(Set1::Many, |(i, def)| {
                        let def_id = hir_map.local_def_id(def.lifetime.id);
                        Set1::One(Region::EarlyBound(i as u32, def_id))
                    })
                }
            }
            Set1::Many => Set1::Many
        }
    }).collect()
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
        let LifetimeContext {sess, cstore, hir_map, ref mut map, ..} = *self;
        let labels_in_fn = replace(&mut self.labels_in_fn, vec![]);
        let xcrate_object_lifetime_defaults =
            replace(&mut self.xcrate_object_lifetime_defaults, DefIdMap());
        let mut this = LifetimeContext {
            sess,
            cstore,
            hir_map,
            map: *map,
            scope: &wrap_scope,
            trait_ref_hack: self.trait_ref_hack,
            labels_in_fn,
            xcrate_object_lifetime_defaults,
        };
        debug!("entering scope {:?}", this.scope);
        f(self.scope, &mut this);
        debug!("exiting scope {:?}", this.scope);
        self.labels_in_fn = this.labels_in_fn;
        self.xcrate_object_lifetime_defaults = this.xcrate_object_lifetime_defaults;
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
                           parent_id: Option<ast::NodeId>,
                           decl: &'tcx hir::FnDecl,
                           generics: &'tcx hir::Generics,
                           walk: F) where
        F: for<'b, 'c> FnOnce(&'b mut LifetimeContext<'c, 'tcx>),
    {
        insert_late_bound_lifetimes(self.map, decl, generics);

        // Find the start of nested early scopes, e.g. in methods.
        let mut index = 0;
        if let Some(parent_id) = parent_id {
            let parent = self.hir_map.expect_item(parent_id);
            if let hir::ItemTrait(..) = parent.node {
                index += 1; // Self comes first.
            }
            match parent.node {
                hir::ItemTrait(_, ref generics, ..) |
                hir::ItemImpl(_, _, _, ref generics, ..) => {
                    index += (generics.lifetimes.len() + generics.ty_params.len()) as u32;
                }
                _ => {}
            }
        }

        let lifetimes = generics.lifetimes.iter().map(|def| {
            if self.map.late_bound.contains(&def.lifetime.id) {
                Region::late(self.hir_map, def)
            } else {
                Region::early(self.hir_map, &mut index, def)
            }
        }).collect();

        let scope = Scope::Binder {
            lifetimes,
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

                Scope::Elision { s, .. } |
                Scope::ObjectLifetimeDefault { s, .. } => {
                    scope = s;
                }
            }
        };

        if let Some(mut def) = result {
            if let Region::EarlyBound(..) = def {
                // Do not free early-bound regions, only late-bound ones.
            } else if let Some(body_id) = outermost_body {
                let fn_id = self.hir_map.body_owner(body_id);
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
                        let scope = self.hir_map.local_def_id(fn_id);
                        def = Region::Free(scope, def.id().unwrap());
                    }
                    _ => {}
                }
            }
            self.insert_lifetime(lifetime_ref, def);
        } else {
            struct_span_err!(self.sess, lifetime_ref.span, E0261,
                "use of undeclared lifetime name `{}`", lifetime_ref.name)
                .span_label(lifetime_ref.span, "undeclared lifetime")
                .emit();
        }
    }

    fn visit_segment_parameters(&mut self,
                                def: Def,
                                depth: usize,
                                params: &'tcx hir::PathParameters) {
        if params.parenthesized {
            self.visit_fn_like_elision(params.inputs(), Some(&params.bindings[0].ty));
            return;
        }

        if params.lifetimes.iter().all(|l| l.is_elided()) {
            self.resolve_elided_lifetimes(&params.lifetimes);
        } else {
            for l in &params.lifetimes { self.visit_lifetime(l); }
        }

        // Figure out if this is a type/trait segment,
        // which requires object lifetime defaults.
        let parent_def_id = |this: &mut Self, def_id: DefId| {
            let def_key = if def_id.is_local() {
                this.hir_map.def_key(def_id)
            } else {
                this.cstore.def_key(def_id)
            };
            DefId {
                krate: def_id.krate,
                index: def_key.parent.expect("missing parent")
            }
        };
        let type_def_id = match def {
            Def::AssociatedTy(def_id) if depth == 1 => {
                Some(parent_def_id(self, def_id))
            }
            Def::Variant(def_id) if depth == 0 => {
                Some(parent_def_id(self, def_id))
            }
            Def::Struct(def_id) |
            Def::Union(def_id) |
            Def::Enum(def_id) |
            Def::TyAlias(def_id) |
            Def::Trait(def_id) if depth == 0 => Some(def_id),
            _ => None
        };

        let object_lifetime_defaults = type_def_id.map_or(vec![], |def_id| {
            let in_body = {
                let mut scope = self.scope;
                loop {
                    match *scope {
                        Scope::Root => break false,

                        Scope::Body { .. } => break true,

                        Scope::Binder { s, .. } |
                        Scope::Elision { s, .. } |
                        Scope::ObjectLifetimeDefault { s, .. } => {
                            scope = s;
                        }
                    }
                }
            };

            let map = &self.map;
            let unsubst = if let Some(id) = self.hir_map.as_local_node_id(def_id) {
                &map.object_lifetime_defaults[&id]
            } else {
                let cstore = self.cstore;
                self.xcrate_object_lifetime_defaults.entry(def_id).or_insert_with(|| {
                    cstore.item_generics_cloned_untracked(def_id).types.into_iter().map(|def| {
                        def.object_lifetime_default
                    }).collect()
                })
            };
            unsubst.iter().map(|set| {
                match *set {
                    Set1::Empty => {
                        if in_body {
                            None
                        } else {
                            Some(Region::Static)
                        }
                    }
                    Set1::One(r) => r.subst(&params.lifetimes, map),
                    Set1::Many => None
                }
            }).collect()
        });

        for (i, ty) in params.types.iter().enumerate() {
            if let Some(&lt) = object_lifetime_defaults.get(i) {
                let scope = Scope::ObjectLifetimeDefault {
                    lifetime: lt,
                    s: self.scope
                };
                self.with(scope, |_, this| this.visit_ty(ty));
            } else {
                self.visit_ty(ty);
            }
        }

        for b in &params.bindings { self.visit_assoc_type_binding(b); }
    }

    fn visit_fn_like_elision(&mut self, inputs: &'tcx [P<hir::Ty>],
                             output: Option<&'tcx P<hir::Ty>>) {
        let mut arg_elide = Elide::FreshLateAnon(Cell::new(0));
        let arg_scope = Scope::Elision {
            elide: arg_elide.clone(),
            s: self.scope
        };
        self.with(arg_scope, |_, this| {
            for input in inputs {
                this.visit_ty(input);
            }
            match *this.scope {
                Scope::Elision { ref elide, .. } => {
                    arg_elide = elide.clone();
                }
                _ => bug!()
            }
        });

        let output = match output {
            Some(ty) => ty,
            None => return
        };

        // Figure out if there's a body we can get argument names from,
        // and whether there's a `self` argument (treated specially).
        let mut assoc_item_kind = None;
        let mut impl_self = None;
        let parent = self.hir_map.get_parent_node(output.id);
        let body = match self.hir_map.get(parent) {
            // `fn` definitions and methods.
            hir::map::NodeItem(&hir::Item {
                node: hir::ItemFn(.., body), ..
            })  => Some(body),

            hir::map::NodeTraitItem(&hir::TraitItem {
                node: hir::TraitItemKind::Method(_, ref m), ..
            }) => {
                match self.hir_map.expect_item(self.hir_map.get_parent(parent)).node {
                    hir::ItemTrait(.., ref trait_items) => {
                        assoc_item_kind = trait_items.iter().find(|ti| ti.id.node_id == parent)
                                                            .map(|ti| ti.kind);
                    }
                    _ => {}
                }
                match *m {
                    hir::TraitMethod::Required(_) => None,
                    hir::TraitMethod::Provided(body) => Some(body),
                }
            }

            hir::map::NodeImplItem(&hir::ImplItem {
                node: hir::ImplItemKind::Method(_, body), ..
            }) => {
                match self.hir_map.expect_item(self.hir_map.get_parent(parent)).node {
                    hir::ItemImpl(.., ref self_ty, ref impl_items) => {
                        impl_self = Some(self_ty);
                        assoc_item_kind = impl_items.iter().find(|ii| ii.id.node_id == parent)
                                                           .map(|ii| ii.kind);
                    }
                    _ => {}
                }
                Some(body)
            }

            // Foreign functions, `fn(...) -> R` and `Trait(...) -> R` (both types and bounds).
            hir::map::NodeForeignItem(_) | hir::map::NodeTy(_) | hir::map::NodeTraitRef(_) => None,

            // Everything else (only closures?) doesn't
            // actually enjoy elision in return types.
            _ => {
                self.visit_ty(output);
                return;
            }
        };

        let has_self = match assoc_item_kind {
            Some(hir::AssociatedItemKind::Method { has_self }) => has_self,
            _ => false
        };

        // In accordance with the rules for lifetime elision, we can determine
        // what region to use for elision in the output type in two ways.
        // First (determined here), if `self` is by-reference, then the
        // implied output region is the region of the self parameter.
        if has_self {
            // Look for `self: &'a Self` - also desugared from `&'a self`,
            // and if that matches, use it for elision and return early.
            let is_self_ty = |def: Def| {
                if let Def::SelfTy(..) = def {
                    return true;
                }

                // Can't always rely on literal (or implied) `Self` due
                // to the way elision rules were originally specified.
                let impl_self = impl_self.map(|ty| &ty.node);
                if let Some(&hir::TyPath(hir::QPath::Resolved(None, ref path))) = impl_self {
                    match path.def {
                        // Whitelist the types that unambiguously always
                        // result in the same type constructor being used
                        // (it can't differ between `Self` and `self`).
                        Def::Struct(_) |
                        Def::Union(_) |
                        Def::Enum(_) |
                        Def::PrimTy(_) => return def == path.def,
                        _ => {}
                    }
                }

                false
            };

            if let hir::TyRptr(lifetime_ref, ref mt) = inputs[0].node {
                if let hir::TyPath(hir::QPath::Resolved(None, ref path)) = mt.ty.node {
                    if is_self_ty(path.def) {
                        if let Some(&lifetime) = self.map.defs.get(&lifetime_ref.id) {
                            let scope = Scope::Elision {
                                elide: Elide::Exact(lifetime),
                                s: self.scope
                            };
                            self.with(scope, |_, this| this.visit_ty(output));
                            return;
                        }
                    }
                }
            }
        }

        // Second, if there was exactly one lifetime (either a substitution or a
        // reference) in the arguments, then any anonymous regions in the output
        // have that lifetime.
        let mut possible_implied_output_region = None;
        let mut lifetime_count = 0;
        let arg_lifetimes = inputs.iter().enumerate().skip(has_self as usize).map(|(i, input)| {
            let mut gather = GatherLifetimes {
                map: self.map,
                binder_depth: 1,
                have_bound_regions: false,
                lifetimes: FxHashSet()
            };
            gather.visit_ty(input);

            lifetime_count += gather.lifetimes.len();

            if lifetime_count == 1 && gather.lifetimes.len() == 1 {
                // there's a chance that the unique lifetime of this
                // iteration will be the appropriate lifetime for output
                // parameters, so lets store it.
                possible_implied_output_region = gather.lifetimes.iter().cloned().next();
            }

            ElisionFailureInfo {
                parent: body,
                index: i,
                lifetime_count: gather.lifetimes.len(),
                have_bound_regions: gather.have_bound_regions
            }
        }).collect();

        let elide = if lifetime_count == 1 {
            Elide::Exact(possible_implied_output_region.unwrap())
        } else {
            Elide::Error(arg_lifetimes)
        };

        let scope = Scope::Elision {
            elide,
            s: self.scope
        };
        self.with(scope, |_, this| this.visit_ty(output));

        struct GatherLifetimes<'a> {
            map: &'a NamedRegionMap,
            binder_depth: u32,
            have_bound_regions: bool,
            lifetimes: FxHashSet<Region>,
        }

        impl<'v, 'a> Visitor<'v> for GatherLifetimes<'a> {
            fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'v> {
                NestedVisitorMap::None
            }

            fn visit_ty(&mut self, ty: &hir::Ty) {
                if let hir::TyBareFn(_) = ty.node {
                    self.binder_depth += 1;
                }
                if let hir::TyTraitObject(ref bounds, ref lifetime) = ty.node {
                    for bound in bounds {
                        self.visit_poly_trait_ref(bound, hir::TraitBoundModifier::None);
                    }

                    // Stay on the safe side and don't include the object
                    // lifetime default (which may not end up being used).
                    if !lifetime.is_elided() {
                        self.visit_lifetime(lifetime);
                    }
                } else {
                    intravisit::walk_ty(self, ty);
                }
                if let hir::TyBareFn(_) = ty.node {
                    self.binder_depth -= 1;
                }
            }

            fn visit_poly_trait_ref(&mut self,
                                    trait_ref: &hir::PolyTraitRef,
                                    modifier: hir::TraitBoundModifier) {
                self.binder_depth += 1;
                intravisit::walk_poly_trait_ref(self, trait_ref, modifier);
                self.binder_depth -= 1;
            }

            fn visit_lifetime_def(&mut self, lifetime_def: &hir::LifetimeDef) {
                for l in &lifetime_def.bounds { self.visit_lifetime(l); }
            }

            fn visit_lifetime(&mut self, lifetime_ref: &hir::Lifetime) {
                if let Some(&lifetime) = self.map.defs.get(&lifetime_ref.id) {
                    match lifetime {
                        Region::LateBound(debruijn, _) |
                        Region::LateBoundAnon(debruijn, _)
                                if debruijn.depth < self.binder_depth => {
                            self.have_bound_regions = true;
                        }
                        _ => {
                            self.lifetimes.insert(lifetime.from_depth(self.binder_depth));
                        }
                    }
                }
            }
        }

    }

    fn resolve_elided_lifetimes(&mut self, lifetime_refs: &[hir::Lifetime]) {
        if lifetime_refs.is_empty() {
            return;
        }

        let span = lifetime_refs[0].span;
        let mut late_depth = 0;
        let mut scope = self.scope;
        let error = loop {
            match *scope {
                // Do not assign any resolution, it will be inferred.
                Scope::Body { .. } => return,

                Scope::Root => break None,

                Scope::Binder { s, .. } => {
                    late_depth += 1;
                    scope = s;
                }

                Scope::Elision { ref elide, .. } => {
                    let lifetime = match *elide {
                        Elide::FreshLateAnon(ref counter) => {
                            for lifetime_ref in lifetime_refs {
                                let lifetime = Region::late_anon(counter).shifted(late_depth);
                                self.insert_lifetime(lifetime_ref, lifetime);
                            }
                            return;
                        }
                        Elide::Exact(l) => l.shifted(late_depth),
                        Elide::Error(ref e) => break Some(e)
                    };
                    for lifetime_ref in lifetime_refs {
                        self.insert_lifetime(lifetime_ref, lifetime);
                    }
                    return;
                }

                Scope::ObjectLifetimeDefault { s, .. } => {
                    scope = s;
                }
            }
        };

        let mut err = struct_span_err!(self.sess, span, E0106,
            "missing lifetime specifier{}",
            if lifetime_refs.len() > 1 { "s" } else { "" });
        let msg = if lifetime_refs.len() > 1 {
            format!("expected {} lifetime parameters", lifetime_refs.len())
        } else {
            format!("expected lifetime parameter")
        };
        err.span_label(span, msg);

        if let Some(params) = error {
            if lifetime_refs.len() == 1 {
                self.report_elision_failure(&mut err, params);
            }
        }
        err.emit();
    }

    fn report_elision_failure(&mut self,
                              db: &mut DiagnosticBuilder,
                              params: &[ElisionFailureInfo]) {
        let mut m = String::new();
        let len = params.len();

        let elided_params: Vec<_> = params.iter().cloned()
                                          .filter(|info| info.lifetime_count > 0)
                                          .collect();

        let elided_len = elided_params.len();

        for (i, info) in elided_params.into_iter().enumerate() {
            let ElisionFailureInfo {
                parent, index, lifetime_count: n, have_bound_regions
            } = info;

            let help_name = if let Some(body) = parent {
                let arg = &self.hir_map.body(body).arguments[index];
                format!("`{}`", self.hir_map.node_to_pretty_string(arg.pat.id))
            } else {
                format!("argument {}", index + 1)
            };

            m.push_str(&(if n == 1 {
                help_name
            } else {
                format!("one of {}'s {} {}lifetimes", help_name, n,
                        if have_bound_regions { "free " } else { "" } )
            })[..]);

            if elided_len == 2 && i == 0 {
                m.push_str(" or ");
            } else if i + 2 == elided_len {
                m.push_str(", or ");
            } else if i != elided_len - 1 {
                m.push_str(", ");
            }

        }

        if len == 0 {
            help!(db,
                  "this function's return type contains a borrowed value, but \
                   there is no value for it to be borrowed from");
            help!(db,
                  "consider giving it a 'static lifetime");
        } else if elided_len == 0 {
            help!(db,
                  "this function's return type contains a borrowed value with \
                   an elided lifetime, but the lifetime cannot be derived from \
                   the arguments");
            help!(db,
                  "consider giving it an explicit bounded or 'static \
                   lifetime");
        } else if elided_len == 1 {
            help!(db,
                  "this function's return type contains a borrowed value, but \
                   the signature does not say which {} it is borrowed from",
                  m);
        } else {
            help!(db,
                  "this function's return type contains a borrowed value, but \
                   the signature does not say whether it is borrowed from {}",
                  m);
        }
    }

    fn resolve_object_lifetime_default(&mut self, lifetime_ref: &hir::Lifetime) {
        let mut late_depth = 0;
        let mut scope = self.scope;
        let lifetime = loop {
            match *scope {
                Scope::Binder { s, .. } => {
                    late_depth += 1;
                    scope = s;
                }

                Scope::Root |
                Scope::Elision { .. } => break Region::Static,

                Scope::Body { .. } |
                Scope::ObjectLifetimeDefault { lifetime: None, .. } => return,

                Scope::ObjectLifetimeDefault { lifetime: Some(l), .. } => break l
            }
        };
        self.insert_lifetime(lifetime_ref, lifetime.shifted(late_depth));
    }

    fn check_lifetime_defs(&mut self, old_scope: ScopeRef, lifetimes: &[hir::LifetimeDef]) {
        for i in 0..lifetimes.len() {
            let lifetime_i = &lifetimes[i];

            for lifetime in lifetimes {
                if lifetime.lifetime.is_static() {
                    let lifetime = lifetime.lifetime;
                    let mut err = struct_span_err!(self.sess, lifetime.span, E0262,
                                  "invalid lifetime parameter name: `{}`", lifetime.name);
                    err.span_label(lifetime.span,
                                   format!("{} is a reserved lifetime name", lifetime.name));
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
                                    "declared twice")
                        .span_label(lifetime_i.lifetime.span,
                                   "previous declaration here")
                        .emit();
                }
            }

            // It is a soft error to shadow a lifetime within a parent scope.
            self.check_lifetime_def_for_shadowing(old_scope, &lifetime_i.lifetime);

            for bound in &lifetime_i.bounds {
                if !bound.is_static() {
                    self.resolve_lifetime_ref(bound);
                } else {
                    self.insert_lifetime(bound, Region::Static);
                    self.sess.struct_span_warn(lifetime_i.lifetime.span.to(bound.span),
                        &format!("unnecessary lifetime parameter `{}`", lifetime_i.lifetime.name))
                        .help(&format!("you can use the `'static` lifetime directly, in place \
                                        of `{}`", lifetime_i.lifetime.name))
                        .emit();
                }
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
                Scope::Body { s, .. } |
                Scope::Elision { s, .. } |
                Scope::ObjectLifetimeDefault { s, .. } => {
                    old_scope = s;
                }

                Scope::Root => {
                    return;
                }

                Scope::Binder { ref lifetimes, s } => {
                    if let Some(&def) = lifetimes.get(&lifetime.name) {
                        let node_id = self.hir_map
                                          .as_local_node_id(def.id().unwrap())
                                          .unwrap();

                        signal_shadowing_problem(
                            self.sess,
                            lifetime.name,
                            original_lifetime(self.hir_map.span(node_id)),
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

        // does not appear in the inputs, but appears in the return type? early-bound.
        if !constrained_by_input.regions.contains(&name) &&
            appears_in_output.regions.contains(&name) {
            continue;
        }

        debug!("insert_late_bound_lifetimes: \
                lifetime {:?} with id {:?} is late-bound",
               lifetime.lifetime.name, lifetime.lifetime.id);

        let inserted = map.late_bound.insert(lifetime.lifetime.id);
        assert!(inserted, "visited lifetime {:?} twice", lifetime.lifetime.id);
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
