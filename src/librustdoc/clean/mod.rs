//! This module contains the "cleaned" pieces of the AST, and the functions
//! that clean them.

mod auto_trait;
mod blanket_impl;
crate mod cfg;
crate mod inline;
mod simplify;
crate mod types;
crate mod utils;

use rustc_ast as ast;
use rustc_attr as attr;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::{CrateNum, DefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_index::vec::{Idx, IndexVec};
use rustc_infer::infer::region_constraints::{Constraint, RegionConstraintData};
use rustc_middle::bug;
use rustc_middle::middle::resolve_lifetime as rl;
use rustc_middle::ty::fold::TypeFolder;
use rustc_middle::ty::subst::{InternalSubsts, Subst};
use rustc_middle::ty::{self, AdtKind, Lift, Ty, TyCtxt};
use rustc_mir::const_eval::{is_const_fn, is_min_const_fn, is_unstable_const_fn};
use rustc_span::hygiene::{AstPass, MacroKind};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{self, ExpnKind};
use rustc_typeck::hir_ty_to_ty;

use std::collections::hash_map::Entry;
use std::default::Default;
use std::hash::Hash;
use std::rc::Rc;
use std::{mem, vec};

use crate::core::{self, DocContext, ImplTraitParam};
use crate::doctree;

use utils::*;

crate use utils::{get_auto_trait_and_blanket_impls, krate, register_res};

crate use self::types::FnRetTy::*;
crate use self::types::ItemKind::*;
crate use self::types::SelfTy::*;
crate use self::types::Type::*;
crate use self::types::Visibility::{Inherited, Public};
crate use self::types::*;

crate trait Clean<T> {
    fn clean(&self, cx: &DocContext<'_>) -> T;
}

impl<T: Clean<U>, U> Clean<Vec<U>> for [T] {
    fn clean(&self, cx: &DocContext<'_>) -> Vec<U> {
        self.iter().map(|x| x.clean(cx)).collect()
    }
}

impl<T: Clean<U>, U, V: Idx> Clean<IndexVec<V, U>> for IndexVec<V, T> {
    fn clean(&self, cx: &DocContext<'_>) -> IndexVec<V, U> {
        self.iter().map(|x| x.clean(cx)).collect()
    }
}

impl<T: Clean<U>, U> Clean<U> for &T {
    fn clean(&self, cx: &DocContext<'_>) -> U {
        (**self).clean(cx)
    }
}

impl<T: Clean<U>, U> Clean<U> for Rc<T> {
    fn clean(&self, cx: &DocContext<'_>) -> U {
        (**self).clean(cx)
    }
}

impl<T: Clean<U>, U> Clean<Option<U>> for Option<T> {
    fn clean(&self, cx: &DocContext<'_>) -> Option<U> {
        self.as_ref().map(|v| v.clean(cx))
    }
}

impl Clean<ExternalCrate> for CrateNum {
    fn clean(&self, cx: &DocContext<'_>) -> ExternalCrate {
        let root = DefId { krate: *self, index: CRATE_DEF_INDEX };
        let krate_span = cx.tcx.def_span(root);
        let krate_src = cx.sess().source_map().span_to_filename(krate_span);

        // Collect all inner modules which are tagged as implementations of
        // primitives.
        //
        // Note that this loop only searches the top-level items of the crate,
        // and this is intentional. If we were to search the entire crate for an
        // item tagged with `#[doc(primitive)]` then we would also have to
        // search the entirety of external modules for items tagged
        // `#[doc(primitive)]`, which is a pretty inefficient process (decoding
        // all that metadata unconditionally).
        //
        // In order to keep the metadata load under control, the
        // `#[doc(primitive)]` feature is explicitly designed to only allow the
        // primitive tags to show up as the top level items in a crate.
        //
        // Also note that this does not attempt to deal with modules tagged
        // duplicately for the same primitive. This is handled later on when
        // rendering by delegating everything to a hash map.
        let as_primitive = |res: Res| {
            if let Res::Def(DefKind::Mod, def_id) = res {
                let attrs = cx.tcx.get_attrs(def_id).clean(cx);
                let mut prim = None;
                for attr in attrs.lists(sym::doc) {
                    if let Some(v) = attr.value_str() {
                        if attr.has_name(sym::primitive) {
                            prim = PrimitiveType::from_symbol(v);
                            if prim.is_some() {
                                break;
                            }
                            // FIXME: should warn on unknown primitives?
                        }
                    }
                }
                return prim.map(|p| (def_id, p));
            }
            None
        };
        let primitives = if root.is_local() {
            cx.tcx
                .hir()
                .krate()
                .item
                .module
                .item_ids
                .iter()
                .filter_map(|&id| {
                    let item = cx.tcx.hir().expect_item(id.id);
                    match item.kind {
                        hir::ItemKind::Mod(_) => as_primitive(Res::Def(
                            DefKind::Mod,
                            cx.tcx.hir().local_def_id(id.id).to_def_id(),
                        )),
                        hir::ItemKind::Use(ref path, hir::UseKind::Single)
                            if item.vis.node.is_pub() =>
                        {
                            as_primitive(path.res).map(|(_, prim)| {
                                // Pretend the primitive is local.
                                (cx.tcx.hir().local_def_id(id.id).to_def_id(), prim)
                            })
                        }
                        _ => None,
                    }
                })
                .collect()
        } else {
            cx.tcx
                .item_children(root)
                .iter()
                .map(|item| item.res)
                .filter_map(as_primitive)
                .collect()
        };

        let as_keyword = |res: Res| {
            if let Res::Def(DefKind::Mod, def_id) = res {
                let attrs = cx.tcx.get_attrs(def_id).clean(cx);
                let mut keyword = None;
                for attr in attrs.lists(sym::doc) {
                    if attr.has_name(sym::keyword) {
                        if let Some(v) = attr.value_str() {
                            keyword = Some(v);
                            break;
                        }
                    }
                }
                return keyword.map(|p| (def_id, p));
            }
            None
        };
        let keywords = if root.is_local() {
            cx.tcx
                .hir()
                .krate()
                .item
                .module
                .item_ids
                .iter()
                .filter_map(|&id| {
                    let item = cx.tcx.hir().expect_item(id.id);
                    match item.kind {
                        hir::ItemKind::Mod(_) => as_keyword(Res::Def(
                            DefKind::Mod,
                            cx.tcx.hir().local_def_id(id.id).to_def_id(),
                        )),
                        hir::ItemKind::Use(ref path, hir::UseKind::Single)
                            if item.vis.node.is_pub() =>
                        {
                            as_keyword(path.res).map(|(_, prim)| {
                                (cx.tcx.hir().local_def_id(id.id).to_def_id(), prim)
                            })
                        }
                        _ => None,
                    }
                })
                .collect()
        } else {
            cx.tcx.item_children(root).iter().map(|item| item.res).filter_map(as_keyword).collect()
        };

        ExternalCrate {
            name: cx.tcx.crate_name(*self),
            src: krate_src,
            attrs: cx.tcx.get_attrs(root).clean(cx),
            primitives,
            keywords,
        }
    }
}

impl Clean<Item> for doctree::Module<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let mut items: Vec<Item> = vec![];
        items.extend(self.imports.iter().flat_map(|x| x.clean(cx)));
        items.extend(self.foreigns.iter().map(|x| x.clean(cx)));
        items.extend(self.mods.iter().map(|x| x.clean(cx)));
        items.extend(self.items.iter().map(|x| x.clean(cx)).flatten());
        items.extend(self.macros.iter().map(|x| x.clean(cx)));

        // determine if we should display the inner contents or
        // the outer `mod` item for the source code.
        let span = {
            let sm = cx.sess().source_map();
            let outer = sm.lookup_char_pos(self.where_outer.lo());
            let inner = sm.lookup_char_pos(self.where_inner.lo());
            if outer.file.start_pos == inner.file.start_pos {
                // mod foo { ... }
                self.where_outer
            } else {
                // mod foo; (and a separate SourceFile for the contents)
                self.where_inner
            }
        };

        let what_rustc_thinks = Item::from_hir_id_and_parts(
            self.id,
            self.name,
            ModuleItem(Module { is_crate: self.is_crate, items }),
            cx,
        );
        Item { source: span.clean(cx), ..what_rustc_thinks }
    }
}

impl Clean<Attributes> for [ast::Attribute] {
    fn clean(&self, cx: &DocContext<'_>) -> Attributes {
        Attributes::from_ast(cx.sess().diagnostic(), self, None)
    }
}

impl Clean<GenericBound> for hir::GenericBound<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> GenericBound {
        match *self {
            hir::GenericBound::Outlives(lt) => GenericBound::Outlives(lt.clean(cx)),
            hir::GenericBound::LangItemTrait(lang_item, span, _, generic_args) => {
                let def_id = cx.tcx.require_lang_item(lang_item, Some(span));

                let trait_ref = ty::TraitRef::identity(cx.tcx, def_id);

                let generic_args = generic_args.clean(cx);
                let bindings = match generic_args {
                    GenericArgs::AngleBracketed { bindings, .. } => bindings,
                    _ => bug!("clean: parenthesized `GenericBound::LangItemTrait`"),
                };

                GenericBound::TraitBound(
                    PolyTrait { trait_: (trait_ref, &*bindings).clean(cx), generic_params: vec![] },
                    hir::TraitBoundModifier::None,
                )
            }
            hir::GenericBound::Trait(ref t, modifier) => {
                GenericBound::TraitBound(t.clean(cx), modifier)
            }
        }
    }
}

impl Clean<Type> for (ty::TraitRef<'_>, &[TypeBinding]) {
    fn clean(&self, cx: &DocContext<'_>) -> Type {
        let (trait_ref, bounds) = *self;
        inline::record_extern_fqn(cx, trait_ref.def_id, TypeKind::Trait);
        let path = external_path(
            cx,
            cx.tcx.item_name(trait_ref.def_id),
            Some(trait_ref.def_id),
            true,
            bounds.to_vec(),
            trait_ref.substs,
        );

        debug!("ty::TraitRef\n  subst: {:?}\n", trait_ref.substs);

        ResolvedPath { path, param_names: None, did: trait_ref.def_id, is_generic: false }
    }
}

impl<'tcx> Clean<GenericBound> for ty::TraitRef<'tcx> {
    fn clean(&self, cx: &DocContext<'_>) -> GenericBound {
        GenericBound::TraitBound(
            PolyTrait { trait_: (*self, &[][..]).clean(cx), generic_params: vec![] },
            hir::TraitBoundModifier::None,
        )
    }
}

impl Clean<GenericBound> for (ty::PolyTraitRef<'_>, &[TypeBinding]) {
    fn clean(&self, cx: &DocContext<'_>) -> GenericBound {
        let (poly_trait_ref, bounds) = *self;
        let poly_trait_ref = poly_trait_ref.lift_to_tcx(cx.tcx).unwrap();

        // collect any late bound regions
        let late_bound_regions: Vec<_> = cx
            .tcx
            .collect_referenced_late_bound_regions(&poly_trait_ref)
            .into_iter()
            .filter_map(|br| match br {
                ty::BrNamed(_, name) => {
                    Some(GenericParamDef { name, kind: GenericParamDefKind::Lifetime })
                }
                _ => None,
            })
            .collect();

        GenericBound::TraitBound(
            PolyTrait {
                trait_: (poly_trait_ref.skip_binder(), bounds).clean(cx),
                generic_params: late_bound_regions,
            },
            hir::TraitBoundModifier::None,
        )
    }
}

impl<'tcx> Clean<GenericBound> for ty::PolyTraitRef<'tcx> {
    fn clean(&self, cx: &DocContext<'_>) -> GenericBound {
        (*self, &[][..]).clean(cx)
    }
}

impl<'tcx> Clean<Option<Vec<GenericBound>>> for InternalSubsts<'tcx> {
    fn clean(&self, cx: &DocContext<'_>) -> Option<Vec<GenericBound>> {
        let mut v = Vec::new();
        v.extend(self.regions().filter_map(|r| r.clean(cx)).map(GenericBound::Outlives));
        v.extend(self.types().map(|t| {
            GenericBound::TraitBound(
                PolyTrait { trait_: t.clean(cx), generic_params: Vec::new() },
                hir::TraitBoundModifier::None,
            )
        }));
        if !v.is_empty() { Some(v) } else { None }
    }
}

impl Clean<Lifetime> for hir::Lifetime {
    fn clean(&self, cx: &DocContext<'_>) -> Lifetime {
        let def = cx.tcx.named_region(self.hir_id);
        match def {
            Some(
                rl::Region::EarlyBound(_, node_id, _)
                | rl::Region::LateBound(_, node_id, _)
                | rl::Region::Free(_, node_id),
            ) => {
                if let Some(lt) = cx.lt_substs.borrow().get(&node_id).cloned() {
                    return lt;
                }
            }
            _ => {}
        }
        Lifetime(self.name.ident().name)
    }
}

impl Clean<Lifetime> for hir::GenericParam<'_> {
    fn clean(&self, _: &DocContext<'_>) -> Lifetime {
        match self.kind {
            hir::GenericParamKind::Lifetime { .. } => {
                if !self.bounds.is_empty() {
                    let mut bounds = self.bounds.iter().map(|bound| match bound {
                        hir::GenericBound::Outlives(lt) => lt,
                        _ => panic!(),
                    });
                    let name = bounds.next().expect("no more bounds").name.ident();
                    let mut s = format!("{}: {}", self.name.ident(), name);
                    for bound in bounds {
                        s.push_str(&format!(" + {}", bound.name.ident()));
                    }
                    Lifetime(Symbol::intern(&s))
                } else {
                    Lifetime(self.name.ident().name)
                }
            }
            _ => panic!(),
        }
    }
}

impl Clean<Constant> for hir::ConstArg {
    fn clean(&self, cx: &DocContext<'_>) -> Constant {
        Constant {
            type_: cx
                .tcx
                .type_of(cx.tcx.hir().body_owner_def_id(self.value.body).to_def_id())
                .clean(cx),
            expr: print_const_expr(cx, self.value.body),
            value: None,
            is_literal: is_literal_expr(cx, self.value.body.hir_id),
        }
    }
}

impl Clean<Lifetime> for ty::GenericParamDef {
    fn clean(&self, _cx: &DocContext<'_>) -> Lifetime {
        Lifetime(self.name)
    }
}

impl Clean<Option<Lifetime>> for ty::RegionKind {
    fn clean(&self, _cx: &DocContext<'_>) -> Option<Lifetime> {
        match *self {
            ty::ReStatic => Some(Lifetime::statik()),
            ty::ReLateBound(_, ty::BoundRegion { kind: ty::BrNamed(_, name) }) => {
                Some(Lifetime(name))
            }
            ty::ReEarlyBound(ref data) => Some(Lifetime(data.name)),

            ty::ReLateBound(..)
            | ty::ReFree(..)
            | ty::ReVar(..)
            | ty::RePlaceholder(..)
            | ty::ReEmpty(_)
            | ty::ReErased => {
                debug!("cannot clean region {:?}", self);
                None
            }
        }
    }
}

impl Clean<WherePredicate> for hir::WherePredicate<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> WherePredicate {
        match *self {
            hir::WherePredicate::BoundPredicate(ref wbp) => WherePredicate::BoundPredicate {
                ty: wbp.bounded_ty.clean(cx),
                bounds: wbp.bounds.clean(cx),
            },

            hir::WherePredicate::RegionPredicate(ref wrp) => WherePredicate::RegionPredicate {
                lifetime: wrp.lifetime.clean(cx),
                bounds: wrp.bounds.clean(cx),
            },

            hir::WherePredicate::EqPredicate(ref wrp) => {
                WherePredicate::EqPredicate { lhs: wrp.lhs_ty.clean(cx), rhs: wrp.rhs_ty.clean(cx) }
            }
        }
    }
}

impl<'a> Clean<Option<WherePredicate>> for ty::Predicate<'a> {
    fn clean(&self, cx: &DocContext<'_>) -> Option<WherePredicate> {
        let bound_predicate = self.bound_atom();
        match bound_predicate.skip_binder() {
            ty::PredicateAtom::Trait(pred, _) => Some(bound_predicate.rebind(pred).clean(cx)),
            ty::PredicateAtom::RegionOutlives(pred) => pred.clean(cx),
            ty::PredicateAtom::TypeOutlives(pred) => pred.clean(cx),
            ty::PredicateAtom::Projection(pred) => Some(pred.clean(cx)),

            ty::PredicateAtom::Subtype(..)
            | ty::PredicateAtom::WellFormed(..)
            | ty::PredicateAtom::ObjectSafe(..)
            | ty::PredicateAtom::ClosureKind(..)
            | ty::PredicateAtom::ConstEvaluatable(..)
            | ty::PredicateAtom::ConstEquate(..)
            | ty::PredicateAtom::TypeWellFormedFromEnv(..) => panic!("not user writable"),
        }
    }
}

impl<'a> Clean<WherePredicate> for ty::PolyTraitPredicate<'a> {
    fn clean(&self, cx: &DocContext<'_>) -> WherePredicate {
        let poly_trait_ref = self.map_bound(|pred| pred.trait_ref);
        WherePredicate::BoundPredicate {
            ty: poly_trait_ref.skip_binder().self_ty().clean(cx),
            bounds: vec![poly_trait_ref.clean(cx)],
        }
    }
}

impl<'tcx> Clean<Option<WherePredicate>>
    for ty::OutlivesPredicate<ty::Region<'tcx>, ty::Region<'tcx>>
{
    fn clean(&self, cx: &DocContext<'_>) -> Option<WherePredicate> {
        let ty::OutlivesPredicate(a, b) = self;

        if let (ty::ReEmpty(_), ty::ReEmpty(_)) = (a, b) {
            return None;
        }

        Some(WherePredicate::RegionPredicate {
            lifetime: a.clean(cx).expect("failed to clean lifetime"),
            bounds: vec![GenericBound::Outlives(b.clean(cx).expect("failed to clean bounds"))],
        })
    }
}

impl<'tcx> Clean<Option<WherePredicate>> for ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>> {
    fn clean(&self, cx: &DocContext<'_>) -> Option<WherePredicate> {
        let ty::OutlivesPredicate(ty, lt) = self;

        if let ty::ReEmpty(_) = lt {
            return None;
        }

        Some(WherePredicate::BoundPredicate {
            ty: ty.clean(cx),
            bounds: vec![GenericBound::Outlives(lt.clean(cx).expect("failed to clean lifetimes"))],
        })
    }
}

impl<'tcx> Clean<WherePredicate> for ty::ProjectionPredicate<'tcx> {
    fn clean(&self, cx: &DocContext<'_>) -> WherePredicate {
        let ty::ProjectionPredicate { projection_ty, ty } = self;
        WherePredicate::EqPredicate { lhs: projection_ty.clean(cx), rhs: ty.clean(cx) }
    }
}

impl<'tcx> Clean<Type> for ty::ProjectionTy<'tcx> {
    fn clean(&self, cx: &DocContext<'_>) -> Type {
        let lifted = self.lift_to_tcx(cx.tcx).unwrap();
        let trait_ = match lifted.trait_ref(cx.tcx).clean(cx) {
            GenericBound::TraitBound(t, _) => t.trait_,
            GenericBound::Outlives(_) => panic!("cleaning a trait got a lifetime"),
        };
        Type::QPath {
            name: cx.tcx.associated_item(self.item_def_id).ident.name,
            self_type: box self.self_ty().clean(cx),
            trait_: box trait_,
        }
    }
}

impl Clean<GenericParamDef> for ty::GenericParamDef {
    fn clean(&self, cx: &DocContext<'_>) -> GenericParamDef {
        let (name, kind) = match self.kind {
            ty::GenericParamDefKind::Lifetime => (self.name, GenericParamDefKind::Lifetime),
            ty::GenericParamDefKind::Type { has_default, synthetic, .. } => {
                let default =
                    if has_default { Some(cx.tcx.type_of(self.def_id).clean(cx)) } else { None };
                (
                    self.name,
                    GenericParamDefKind::Type {
                        did: self.def_id,
                        bounds: vec![], // These are filled in from the where-clauses.
                        default,
                        synthetic,
                    },
                )
            }
            ty::GenericParamDefKind::Const { .. } => (
                self.name,
                GenericParamDefKind::Const {
                    did: self.def_id,
                    ty: cx.tcx.type_of(self.def_id).clean(cx),
                },
            ),
        };

        GenericParamDef { name, kind }
    }
}

impl Clean<GenericParamDef> for hir::GenericParam<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> GenericParamDef {
        let (name, kind) = match self.kind {
            hir::GenericParamKind::Lifetime { .. } => {
                let name = if !self.bounds.is_empty() {
                    let mut bounds = self.bounds.iter().map(|bound| match bound {
                        hir::GenericBound::Outlives(lt) => lt,
                        _ => panic!(),
                    });
                    let name = bounds.next().expect("no more bounds").name.ident();
                    let mut s = format!("{}: {}", self.name.ident(), name);
                    for bound in bounds {
                        s.push_str(&format!(" + {}", bound.name.ident()));
                    }
                    Symbol::intern(&s)
                } else {
                    self.name.ident().name
                };
                (name, GenericParamDefKind::Lifetime)
            }
            hir::GenericParamKind::Type { ref default, synthetic } => (
                self.name.ident().name,
                GenericParamDefKind::Type {
                    did: cx.tcx.hir().local_def_id(self.hir_id).to_def_id(),
                    bounds: self.bounds.clean(cx),
                    default: default.clean(cx),
                    synthetic,
                },
            ),
            hir::GenericParamKind::Const { ref ty, default: _ } => (
                self.name.ident().name,
                GenericParamDefKind::Const {
                    did: cx.tcx.hir().local_def_id(self.hir_id).to_def_id(),
                    ty: ty.clean(cx),
                    // FIXME(const_generics_defaults): add `default` field here for docs
                },
            ),
        };

        GenericParamDef { name, kind }
    }
}

impl Clean<Generics> for hir::Generics<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Generics {
        // Synthetic type-parameters are inserted after normal ones.
        // In order for normal parameters to be able to refer to synthetic ones,
        // scans them first.
        fn is_impl_trait(param: &hir::GenericParam<'_>) -> bool {
            match param.kind {
                hir::GenericParamKind::Type { synthetic, .. } => {
                    synthetic == Some(hir::SyntheticTyParamKind::ImplTrait)
                }
                _ => false,
            }
        }
        /// This can happen for `async fn`, e.g. `async fn f<'_>(&'_ self)`.
        ///
        /// See [`lifetime_to_generic_param`] in [`rustc_ast_lowering`] for more information.
        ///
        /// [`lifetime_to_generic_param`]: rustc_ast_lowering::LoweringContext::lifetime_to_generic_param
        fn is_elided_lifetime(param: &hir::GenericParam<'_>) -> bool {
            match param.kind {
                hir::GenericParamKind::Lifetime { kind: hir::LifetimeParamKind::Elided } => true,
                _ => false,
            }
        }

        let impl_trait_params = self
            .params
            .iter()
            .filter(|param| is_impl_trait(param))
            .map(|param| {
                let param: GenericParamDef = param.clean(cx);
                match param.kind {
                    GenericParamDefKind::Lifetime => unreachable!(),
                    GenericParamDefKind::Type { did, ref bounds, .. } => {
                        cx.impl_trait_bounds.borrow_mut().insert(did.into(), bounds.clone());
                    }
                    GenericParamDefKind::Const { .. } => unreachable!(),
                }
                param
            })
            .collect::<Vec<_>>();

        let mut params = Vec::with_capacity(self.params.len());
        for p in self.params.iter().filter(|p| !is_impl_trait(p) && !is_elided_lifetime(p)) {
            let p = p.clean(cx);
            params.push(p);
        }
        params.extend(impl_trait_params);

        let mut generics =
            Generics { params, where_predicates: self.where_clause.predicates.clean(cx) };

        // Some duplicates are generated for ?Sized bounds between type params and where
        // predicates. The point in here is to move the bounds definitions from type params
        // to where predicates when such cases occur.
        for where_pred in &mut generics.where_predicates {
            match *where_pred {
                WherePredicate::BoundPredicate { ty: Generic(ref name), ref mut bounds } => {
                    if bounds.is_empty() {
                        for param in &mut generics.params {
                            match param.kind {
                                GenericParamDefKind::Lifetime => {}
                                GenericParamDefKind::Type { bounds: ref mut ty_bounds, .. } => {
                                    if &param.name == name {
                                        mem::swap(bounds, ty_bounds);
                                        break;
                                    }
                                }
                                GenericParamDefKind::Const { .. } => {}
                            }
                        }
                    }
                }
                _ => continue,
            }
        }
        generics
    }
}

impl<'a, 'tcx> Clean<Generics> for (&'a ty::Generics, ty::GenericPredicates<'tcx>) {
    fn clean(&self, cx: &DocContext<'_>) -> Generics {
        use self::WherePredicate as WP;
        use std::collections::BTreeMap;

        let (gens, preds) = *self;

        // Don't populate `cx.impl_trait_bounds` before `clean`ning `where` clauses,
        // since `Clean for ty::Predicate` would consume them.
        let mut impl_trait = BTreeMap::<ImplTraitParam, Vec<GenericBound>>::default();

        // Bounds in the type_params and lifetimes fields are repeated in the
        // predicates field (see rustc_typeck::collect::ty_generics), so remove
        // them.
        let stripped_params = gens
            .params
            .iter()
            .filter_map(|param| match param.kind {
                ty::GenericParamDefKind::Lifetime => Some(param.clean(cx)),
                ty::GenericParamDefKind::Type { synthetic, .. } => {
                    if param.name == kw::SelfUpper {
                        assert_eq!(param.index, 0);
                        return None;
                    }
                    if synthetic == Some(hir::SyntheticTyParamKind::ImplTrait) {
                        impl_trait.insert(param.index.into(), vec![]);
                        return None;
                    }
                    Some(param.clean(cx))
                }
                ty::GenericParamDefKind::Const { .. } => Some(param.clean(cx)),
            })
            .collect::<Vec<GenericParamDef>>();

        // param index -> [(DefId of trait, associated type name, type)]
        let mut impl_trait_proj = FxHashMap::<u32, Vec<(DefId, Symbol, Ty<'tcx>)>>::default();

        let where_predicates = preds
            .predicates
            .iter()
            .flat_map(|(p, _)| {
                let mut projection = None;
                let param_idx = (|| {
                    let bound_p = p.bound_atom();
                    match bound_p.skip_binder() {
                        ty::PredicateAtom::Trait(pred, _constness) => {
                            if let ty::Param(param) = pred.self_ty().kind() {
                                return Some(param.index);
                            }
                        }
                        ty::PredicateAtom::TypeOutlives(ty::OutlivesPredicate(ty, _reg)) => {
                            if let ty::Param(param) = ty.kind() {
                                return Some(param.index);
                            }
                        }
                        ty::PredicateAtom::Projection(p) => {
                            if let ty::Param(param) = p.projection_ty.self_ty().kind() {
                                projection = Some(bound_p.rebind(p));
                                return Some(param.index);
                            }
                        }
                        _ => (),
                    }

                    None
                })();

                if let Some(param_idx) = param_idx {
                    if let Some(b) = impl_trait.get_mut(&param_idx.into()) {
                        let p = p.clean(cx)?;

                        b.extend(
                            p.get_bounds()
                                .into_iter()
                                .flatten()
                                .cloned()
                                .filter(|b| !b.is_sized_bound(cx)),
                        );

                        let proj = projection
                            .map(|p| (p.skip_binder().projection_ty.clean(cx), p.skip_binder().ty));
                        if let Some(((_, trait_did, name), rhs)) =
                            proj.as_ref().and_then(|(lhs, rhs)| Some((lhs.projection()?, rhs)))
                        {
                            impl_trait_proj
                                .entry(param_idx)
                                .or_default()
                                .push((trait_did, name, rhs));
                        }

                        return None;
                    }
                }

                Some(p)
            })
            .collect::<Vec<_>>();

        for (param, mut bounds) in impl_trait {
            // Move trait bounds to the front.
            bounds.sort_by_key(|b| if let GenericBound::TraitBound(..) = b { false } else { true });

            if let crate::core::ImplTraitParam::ParamIndex(idx) = param {
                if let Some(proj) = impl_trait_proj.remove(&idx) {
                    for (trait_did, name, rhs) in proj {
                        simplify::merge_bounds(cx, &mut bounds, trait_did, name, &rhs.clean(cx));
                    }
                }
            } else {
                unreachable!();
            }

            cx.impl_trait_bounds.borrow_mut().insert(param, bounds);
        }

        // Now that `cx.impl_trait_bounds` is populated, we can process
        // remaining predicates which could contain `impl Trait`.
        let mut where_predicates =
            where_predicates.into_iter().flat_map(|p| p.clean(cx)).collect::<Vec<_>>();

        // Type parameters have a Sized bound by default unless removed with
        // ?Sized. Scan through the predicates and mark any type parameter with
        // a Sized bound, removing the bounds as we find them.
        //
        // Note that associated types also have a sized bound by default, but we
        // don't actually know the set of associated types right here so that's
        // handled in cleaning associated types
        let mut sized_params = FxHashSet::default();
        where_predicates.retain(|pred| match *pred {
            WP::BoundPredicate { ty: Generic(ref g), ref bounds } => {
                if bounds.iter().any(|b| b.is_sized_bound(cx)) {
                    sized_params.insert(*g);
                    false
                } else {
                    true
                }
            }
            _ => true,
        });

        // Run through the type parameters again and insert a ?Sized
        // unbound for any we didn't find to be Sized.
        for tp in &stripped_params {
            if matches!(tp.kind, types::GenericParamDefKind::Type { .. })
                && !sized_params.contains(&tp.name)
            {
                where_predicates.push(WP::BoundPredicate {
                    ty: Type::Generic(tp.name),
                    bounds: vec![GenericBound::maybe_sized(cx)],
                })
            }
        }

        // It would be nice to collect all of the bounds on a type and recombine
        // them if possible, to avoid e.g., `where T: Foo, T: Bar, T: Sized, T: 'a`
        // and instead see `where T: Foo + Bar + Sized + 'a`

        Generics {
            params: stripped_params,
            where_predicates: simplify::where_clauses(cx, where_predicates),
        }
    }
}

fn clean_fn_or_proc_macro(
    item: &hir::Item<'_>,
    sig: &'a hir::FnSig<'a>,
    generics: &'a hir::Generics<'a>,
    body_id: hir::BodyId,
    name: &mut Symbol,
    cx: &DocContext<'_>,
) -> ItemKind {
    let macro_kind = item.attrs.iter().find_map(|a| {
        if a.has_name(sym::proc_macro) {
            Some(MacroKind::Bang)
        } else if a.has_name(sym::proc_macro_derive) {
            Some(MacroKind::Derive)
        } else if a.has_name(sym::proc_macro_attribute) {
            Some(MacroKind::Attr)
        } else {
            None
        }
    });
    match macro_kind {
        Some(kind) => {
            if kind == MacroKind::Derive {
                *name = item
                    .attrs
                    .lists(sym::proc_macro_derive)
                    .find_map(|mi| mi.ident())
                    .expect("proc-macro derives require a name")
                    .name;
            }

            let mut helpers = Vec::new();
            for mi in item.attrs.lists(sym::proc_macro_derive) {
                if !mi.has_name(sym::attributes) {
                    continue;
                }

                if let Some(list) = mi.meta_item_list() {
                    for inner_mi in list {
                        if let Some(ident) = inner_mi.ident() {
                            helpers.push(ident.name);
                        }
                    }
                }
            }
            ProcMacroItem(ProcMacro { kind, helpers })
        }
        None => {
            let mut func = (sig, generics, body_id).clean(cx);
            let def_id = cx.tcx.hir().local_def_id(item.hir_id).to_def_id();
            func.header.constness =
                if is_const_fn(cx.tcx, def_id) && is_unstable_const_fn(cx.tcx, def_id).is_none() {
                    hir::Constness::Const
                } else {
                    hir::Constness::NotConst
                };
            FunctionItem(func)
        }
    }
}

impl<'a> Clean<Function> for (&'a hir::FnSig<'a>, &'a hir::Generics<'a>, hir::BodyId) {
    fn clean(&self, cx: &DocContext<'_>) -> Function {
        let (generics, decl) =
            enter_impl_trait(cx, || (self.1.clean(cx), (&*self.0.decl, self.2).clean(cx)));
        let (all_types, ret_types) = get_all_types(&generics, &decl, cx);
        Function { decl, generics, header: self.0.header, all_types, ret_types }
    }
}

impl<'a> Clean<Arguments> for (&'a [hir::Ty<'a>], &'a [Ident]) {
    fn clean(&self, cx: &DocContext<'_>) -> Arguments {
        Arguments {
            values: self
                .0
                .iter()
                .enumerate()
                .map(|(i, ty)| {
                    let mut name = self.1.get(i).map(|ident| ident.name).unwrap_or(kw::Empty);
                    if name.is_empty() {
                        name = kw::Underscore;
                    }
                    Argument { name, type_: ty.clean(cx) }
                })
                .collect(),
        }
    }
}

impl<'a> Clean<Arguments> for (&'a [hir::Ty<'a>], hir::BodyId) {
    fn clean(&self, cx: &DocContext<'_>) -> Arguments {
        let body = cx.tcx.hir().body(self.1);

        Arguments {
            values: self
                .0
                .iter()
                .enumerate()
                .map(|(i, ty)| Argument {
                    name: name_from_pat(&body.params[i].pat),
                    type_: ty.clean(cx),
                })
                .collect(),
        }
    }
}

impl<'a, A: Copy> Clean<FnDecl> for (&'a hir::FnDecl<'a>, A)
where
    (&'a [hir::Ty<'a>], A): Clean<Arguments>,
{
    fn clean(&self, cx: &DocContext<'_>) -> FnDecl {
        FnDecl {
            inputs: (&self.0.inputs[..], self.1).clean(cx),
            output: self.0.output.clean(cx),
            c_variadic: self.0.c_variadic,
            attrs: Attributes::default(),
        }
    }
}

impl<'tcx> Clean<FnDecl> for (DefId, ty::PolyFnSig<'tcx>) {
    fn clean(&self, cx: &DocContext<'_>) -> FnDecl {
        let (did, sig) = *self;
        let mut names = if did.is_local() { &[] } else { cx.tcx.fn_arg_names(did) }.iter();

        FnDecl {
            output: Return(sig.skip_binder().output().clean(cx)),
            attrs: Attributes::default(),
            c_variadic: sig.skip_binder().c_variadic,
            inputs: Arguments {
                values: sig
                    .skip_binder()
                    .inputs()
                    .iter()
                    .map(|t| Argument {
                        type_: t.clean(cx),
                        name: names.next().map(|i| i.name).unwrap_or(kw::Empty),
                    })
                    .collect(),
            },
        }
    }
}

impl Clean<FnRetTy> for hir::FnRetTy<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> FnRetTy {
        match *self {
            Self::Return(ref typ) => Return(typ.clean(cx)),
            Self::DefaultReturn(..) => DefaultReturn,
        }
    }
}

impl Clean<bool> for hir::IsAuto {
    fn clean(&self, _: &DocContext<'_>) -> bool {
        match *self {
            hir::IsAuto::Yes => true,
            hir::IsAuto::No => false,
        }
    }
}

impl Clean<Type> for hir::TraitRef<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Type {
        resolve_type(cx, self.path.clean(cx), self.hir_ref_id)
    }
}

impl Clean<PolyTrait> for hir::PolyTraitRef<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> PolyTrait {
        PolyTrait {
            trait_: self.trait_ref.clean(cx),
            generic_params: self.bound_generic_params.clean(cx),
        }
    }
}

impl Clean<TypeKind> for hir::def::DefKind {
    fn clean(&self, _: &DocContext<'_>) -> TypeKind {
        match *self {
            hir::def::DefKind::Mod => TypeKind::Module,
            hir::def::DefKind::Struct => TypeKind::Struct,
            hir::def::DefKind::Union => TypeKind::Union,
            hir::def::DefKind::Enum => TypeKind::Enum,
            hir::def::DefKind::Trait => TypeKind::Trait,
            hir::def::DefKind::TyAlias => TypeKind::Typedef,
            hir::def::DefKind::ForeignTy => TypeKind::Foreign,
            hir::def::DefKind::TraitAlias => TypeKind::TraitAlias,
            hir::def::DefKind::Fn => TypeKind::Function,
            hir::def::DefKind::Const => TypeKind::Const,
            hir::def::DefKind::Static => TypeKind::Static,
            hir::def::DefKind::Macro(_) => TypeKind::Macro,
            _ => TypeKind::Foreign,
        }
    }
}

impl Clean<Item> for hir::TraitItem<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let local_did = cx.tcx.hir().local_def_id(self.hir_id).to_def_id();
        cx.with_param_env(local_did, || {
            let inner = match self.kind {
                hir::TraitItemKind::Const(ref ty, default) => {
                    AssocConstItem(ty.clean(cx), default.map(|e| print_const_expr(cx, e)))
                }
                hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Provided(body)) => {
                    let mut m = (sig, &self.generics, body).clean(cx);
                    if m.header.constness == hir::Constness::Const
                        && is_unstable_const_fn(cx.tcx, local_did).is_some()
                    {
                        m.header.constness = hir::Constness::NotConst;
                    }
                    MethodItem(m, None)
                }
                hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Required(ref names)) => {
                    let (generics, decl) = enter_impl_trait(cx, || {
                        (self.generics.clean(cx), (&*sig.decl, &names[..]).clean(cx))
                    });
                    let (all_types, ret_types) = get_all_types(&generics, &decl, cx);
                    let mut t =
                        Function { header: sig.header, decl, generics, all_types, ret_types };
                    if t.header.constness == hir::Constness::Const
                        && is_unstable_const_fn(cx.tcx, local_did).is_some()
                    {
                        t.header.constness = hir::Constness::NotConst;
                    }
                    TyMethodItem(t)
                }
                hir::TraitItemKind::Type(ref bounds, ref default) => {
                    AssocTypeItem(bounds.clean(cx), default.clean(cx))
                }
            };
            Item::from_def_id_and_parts(local_did, Some(self.ident.name), inner, cx)
        })
    }
}

impl Clean<Item> for hir::ImplItem<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let local_did = cx.tcx.hir().local_def_id(self.hir_id).to_def_id();
        cx.with_param_env(local_did, || {
            let inner = match self.kind {
                hir::ImplItemKind::Const(ref ty, expr) => {
                    AssocConstItem(ty.clean(cx), Some(print_const_expr(cx, expr)))
                }
                hir::ImplItemKind::Fn(ref sig, body) => {
                    let mut m = (sig, &self.generics, body).clean(cx);
                    if m.header.constness == hir::Constness::Const
                        && is_unstable_const_fn(cx.tcx, local_did).is_some()
                    {
                        m.header.constness = hir::Constness::NotConst;
                    }
                    MethodItem(m, Some(self.defaultness))
                }
                hir::ImplItemKind::TyAlias(ref ty) => {
                    let type_ = ty.clean(cx);
                    let item_type = type_.def_id().and_then(|did| inline::build_ty(cx, did));
                    TypedefItem(Typedef { type_, generics: Generics::default(), item_type }, true)
                }
            };
            Item::from_def_id_and_parts(local_did, Some(self.ident.name), inner, cx)
        })
    }
}

impl Clean<Item> for ty::AssocItem {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let kind = match self.kind {
            ty::AssocKind::Const => {
                let ty = cx.tcx.type_of(self.def_id);
                let default = if self.defaultness.has_value() {
                    Some(inline::print_inlined_const(cx, self.def_id))
                } else {
                    None
                };
                AssocConstItem(ty.clean(cx), default)
            }
            ty::AssocKind::Fn => {
                let generics =
                    (cx.tcx.generics_of(self.def_id), cx.tcx.explicit_predicates_of(self.def_id))
                        .clean(cx);
                let sig = cx.tcx.fn_sig(self.def_id);
                let mut decl = (self.def_id, sig).clean(cx);

                if self.fn_has_self_parameter {
                    let self_ty = match self.container {
                        ty::ImplContainer(def_id) => cx.tcx.type_of(def_id),
                        ty::TraitContainer(_) => cx.tcx.types.self_param,
                    };
                    let self_arg_ty = sig.input(0).skip_binder();
                    if self_arg_ty == self_ty {
                        decl.inputs.values[0].type_ = Generic(kw::SelfUpper);
                    } else if let ty::Ref(_, ty, _) = *self_arg_ty.kind() {
                        if ty == self_ty {
                            match decl.inputs.values[0].type_ {
                                BorrowedRef { ref mut type_, .. } => {
                                    **type_ = Generic(kw::SelfUpper)
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                }

                let provided = match self.container {
                    ty::ImplContainer(_) => true,
                    ty::TraitContainer(_) => self.defaultness.has_value(),
                };
                let (all_types, ret_types) = get_all_types(&generics, &decl, cx);
                if provided {
                    let constness = if is_min_const_fn(cx.tcx, self.def_id) {
                        hir::Constness::Const
                    } else {
                        hir::Constness::NotConst
                    };
                    let asyncness = cx.tcx.asyncness(self.def_id);
                    let defaultness = match self.container {
                        ty::ImplContainer(_) => Some(self.defaultness),
                        ty::TraitContainer(_) => None,
                    };
                    MethodItem(
                        Function {
                            generics,
                            decl,
                            header: hir::FnHeader {
                                unsafety: sig.unsafety(),
                                abi: sig.abi(),
                                constness,
                                asyncness,
                            },
                            all_types,
                            ret_types,
                        },
                        defaultness,
                    )
                } else {
                    TyMethodItem(Function {
                        generics,
                        decl,
                        header: hir::FnHeader {
                            unsafety: sig.unsafety(),
                            abi: sig.abi(),
                            constness: hir::Constness::NotConst,
                            asyncness: hir::IsAsync::NotAsync,
                        },
                        all_types,
                        ret_types,
                    })
                }
            }
            ty::AssocKind::Type => {
                let my_name = self.ident.name;

                if let ty::TraitContainer(_) = self.container {
                    let bounds = cx.tcx.explicit_item_bounds(self.def_id);
                    let predicates = ty::GenericPredicates { parent: None, predicates: bounds };
                    let generics = (cx.tcx.generics_of(self.def_id), predicates).clean(cx);
                    let mut bounds = generics
                        .where_predicates
                        .iter()
                        .filter_map(|pred| {
                            let (name, self_type, trait_, bounds) = match *pred {
                                WherePredicate::BoundPredicate {
                                    ty: QPath { ref name, ref self_type, ref trait_ },
                                    ref bounds,
                                } => (name, self_type, trait_, bounds),
                                _ => return None,
                            };
                            if *name != my_name {
                                return None;
                            }
                            match **trait_ {
                                ResolvedPath { did, .. } if did == self.container.id() => {}
                                _ => return None,
                            }
                            match **self_type {
                                Generic(ref s) if *s == kw::SelfUpper => {}
                                _ => return None,
                            }
                            Some(bounds)
                        })
                        .flat_map(|i| i.iter().cloned())
                        .collect::<Vec<_>>();
                    // Our Sized/?Sized bound didn't get handled when creating the generics
                    // because we didn't actually get our whole set of bounds until just now
                    // (some of them may have come from the trait). If we do have a sized
                    // bound, we remove it, and if we don't then we add the `?Sized` bound
                    // at the end.
                    match bounds.iter().position(|b| b.is_sized_bound(cx)) {
                        Some(i) => {
                            bounds.remove(i);
                        }
                        None => bounds.push(GenericBound::maybe_sized(cx)),
                    }

                    let ty = if self.defaultness.has_value() {
                        Some(cx.tcx.type_of(self.def_id))
                    } else {
                        None
                    };

                    AssocTypeItem(bounds, ty.clean(cx))
                } else {
                    let type_ = cx.tcx.type_of(self.def_id).clean(cx);
                    let item_type = type_.def_id().and_then(|did| inline::build_ty(cx, did));
                    TypedefItem(
                        Typedef {
                            type_,
                            generics: Generics { params: Vec::new(), where_predicates: Vec::new() },
                            item_type,
                        },
                        true,
                    )
                }
            }
        };

        Item::from_def_id_and_parts(self.def_id, Some(self.ident.name), kind, cx)
    }
}

fn clean_qpath(hir_ty: &hir::Ty<'_>, cx: &DocContext<'_>) -> Type {
    use rustc_hir::GenericParamCount;
    let hir::Ty { hir_id, span, ref kind } = *hir_ty;
    let qpath = match kind {
        hir::TyKind::Path(qpath) => qpath,
        _ => unreachable!(),
    };

    match qpath {
        hir::QPath::Resolved(None, ref path) => {
            if let Res::Def(DefKind::TyParam, did) = path.res {
                if let Some(new_ty) = cx.ty_substs.borrow().get(&did).cloned() {
                    return new_ty;
                }
                if let Some(bounds) = cx.impl_trait_bounds.borrow_mut().remove(&did.into()) {
                    return ImplTrait(bounds);
                }
            }

            let mut alias = None;
            if let Res::Def(DefKind::TyAlias, def_id) = path.res {
                // Substitute private type aliases
                if let Some(def_id) = def_id.as_local() {
                    let hir_id = cx.tcx.hir().local_def_id_to_hir_id(def_id);
                    if !cx.renderinfo.borrow().access_levels.is_exported(def_id.to_def_id()) {
                        alias = Some(&cx.tcx.hir().expect_item(hir_id).kind);
                    }
                }
            };

            if let Some(&hir::ItemKind::TyAlias(ref ty, ref generics)) = alias {
                let provided_params = &path.segments.last().expect("segments were empty");
                let mut ty_substs = FxHashMap::default();
                let mut lt_substs = FxHashMap::default();
                let mut ct_substs = FxHashMap::default();
                let generic_args = provided_params.generic_args();
                {
                    let mut indices: GenericParamCount = Default::default();
                    for param in generics.params.iter() {
                        match param.kind {
                            hir::GenericParamKind::Lifetime { .. } => {
                                let mut j = 0;
                                let lifetime = generic_args.args.iter().find_map(|arg| match arg {
                                    hir::GenericArg::Lifetime(lt) => {
                                        if indices.lifetimes == j {
                                            return Some(lt);
                                        }
                                        j += 1;
                                        None
                                    }
                                    _ => None,
                                });
                                if let Some(lt) = lifetime.cloned() {
                                    let lt_def_id = cx.tcx.hir().local_def_id(param.hir_id);
                                    let cleaned = if !lt.is_elided() {
                                        lt.clean(cx)
                                    } else {
                                        self::types::Lifetime::elided()
                                    };
                                    lt_substs.insert(lt_def_id.to_def_id(), cleaned);
                                }
                                indices.lifetimes += 1;
                            }
                            hir::GenericParamKind::Type { ref default, .. } => {
                                let ty_param_def_id = cx.tcx.hir().local_def_id(param.hir_id);
                                let mut j = 0;
                                let type_ = generic_args.args.iter().find_map(|arg| match arg {
                                    hir::GenericArg::Type(ty) => {
                                        if indices.types == j {
                                            return Some(ty);
                                        }
                                        j += 1;
                                        None
                                    }
                                    _ => None,
                                });
                                if let Some(ty) = type_ {
                                    ty_substs.insert(ty_param_def_id.to_def_id(), ty.clean(cx));
                                } else if let Some(default) = *default {
                                    ty_substs
                                        .insert(ty_param_def_id.to_def_id(), default.clean(cx));
                                }
                                indices.types += 1;
                            }
                            hir::GenericParamKind::Const { .. } => {
                                let const_param_def_id = cx.tcx.hir().local_def_id(param.hir_id);
                                let mut j = 0;
                                let const_ = generic_args.args.iter().find_map(|arg| match arg {
                                    hir::GenericArg::Const(ct) => {
                                        if indices.consts == j {
                                            return Some(ct);
                                        }
                                        j += 1;
                                        None
                                    }
                                    _ => None,
                                });
                                if let Some(ct) = const_ {
                                    ct_substs.insert(const_param_def_id.to_def_id(), ct.clean(cx));
                                }
                                // FIXME(const_generics_defaults)
                                indices.consts += 1;
                            }
                        }
                    }
                }
                return cx.enter_alias(ty_substs, lt_substs, ct_substs, || ty.clean(cx));
            }
            resolve_type(cx, path.clean(cx), hir_id)
        }
        hir::QPath::Resolved(Some(ref qself), ref p) => {
            // Try to normalize `<X as Y>::T` to a type
            let ty = hir_ty_to_ty(cx.tcx, hir_ty);
            if let Some(normalized_value) = normalize(cx, ty) {
                return normalized_value.clean(cx);
            }

            let segments = if p.is_global() { &p.segments[1..] } else { &p.segments };
            let trait_segments = &segments[..segments.len() - 1];
            let trait_path = self::Path {
                global: p.is_global(),
                res: Res::Def(
                    DefKind::Trait,
                    cx.tcx.associated_item(p.res.def_id()).container.id(),
                ),
                segments: trait_segments.clean(cx),
            };
            Type::QPath {
                name: p.segments.last().expect("segments were empty").ident.name,
                self_type: box qself.clean(cx),
                trait_: box resolve_type(cx, trait_path, hir_id),
            }
        }
        hir::QPath::TypeRelative(ref qself, ref segment) => {
            let ty = hir_ty_to_ty(cx.tcx, hir_ty);
            let res = if let ty::Projection(proj) = ty.kind() {
                Res::Def(DefKind::Trait, proj.trait_ref(cx.tcx).def_id)
            } else {
                Res::Err
            };
            let trait_path = hir::Path { span, res, segments: &[] };
            Type::QPath {
                name: segment.ident.name,
                self_type: box qself.clean(cx),
                trait_: box resolve_type(cx, trait_path.clean(cx), hir_id),
            }
        }
        hir::QPath::LangItem(..) => bug!("clean: requiring documentation of lang item"),
    }
}

impl Clean<Type> for hir::Ty<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Type {
        use rustc_hir::*;

        match self.kind {
            TyKind::Never => Never,
            TyKind::Ptr(ref m) => RawPointer(m.mutbl, box m.ty.clean(cx)),
            TyKind::Rptr(ref l, ref m) => {
                // There are two times a `Fresh` lifetime can be created:
                // 1. For `&'_ x`, written by the user. This corresponds to `lower_lifetime` in `rustc_ast_lowering`.
                // 2. For `&x` as a parameter to an `async fn`. This corresponds to `elided_ref_lifetime in `rustc_ast_lowering`.
                //    See #59286 for more information.
                // Ideally we would only hide the `'_` for case 2., but I don't know a way to distinguish it.
                // Turning `fn f(&'_ self)` into `fn f(&self)` isn't the worst thing in the world, though;
                // there's no case where it could cause the function to fail to compile.
                let elided =
                    l.is_elided() || matches!(l.name, LifetimeName::Param(ParamName::Fresh(_)));
                let lifetime = if elided { None } else { Some(l.clean(cx)) };
                BorrowedRef { lifetime, mutability: m.mutbl, type_: box m.ty.clean(cx) }
            }
            TyKind::Slice(ref ty) => Slice(box ty.clean(cx)),
            TyKind::Array(ref ty, ref length) => {
                let def_id = cx.tcx.hir().local_def_id(length.hir_id);
                // NOTE(min_const_generics): We can't use `const_eval_poly` for constants
                // as we currently do not supply the parent generics to anonymous constants
                // but do allow `ConstKind::Param`.
                //
                // `const_eval_poly` tries to to first substitute generic parameters which
                // results in an ICE while manually constructing the constant and using `eval`
                // does nothing for `ConstKind::Param`.
                let ct = ty::Const::from_anon_const(cx.tcx, def_id);
                let param_env = cx.tcx.param_env(def_id);
                let length = print_const(cx, ct.eval(cx.tcx, param_env));
                Array(box ty.clean(cx), length)
            }
            TyKind::Tup(ref tys) => Tuple(tys.clean(cx)),
            TyKind::OpaqueDef(item_id, _) => {
                let item = cx.tcx.hir().expect_item(item_id.id);
                if let hir::ItemKind::OpaqueTy(ref ty) = item.kind {
                    ImplTrait(ty.bounds.clean(cx))
                } else {
                    unreachable!()
                }
            }
            TyKind::Path(_) => clean_qpath(&self, cx),
            TyKind::TraitObject(ref bounds, ref lifetime) => {
                match bounds[0].clean(cx).trait_ {
                    ResolvedPath { path, param_names: None, did, is_generic } => {
                        let mut bounds: Vec<self::GenericBound> = bounds[1..]
                            .iter()
                            .map(|bound| {
                                self::GenericBound::TraitBound(
                                    bound.clean(cx),
                                    hir::TraitBoundModifier::None,
                                )
                            })
                            .collect();
                        if !lifetime.is_elided() {
                            bounds.push(self::GenericBound::Outlives(lifetime.clean(cx)));
                        }
                        ResolvedPath { path, param_names: Some(bounds), did, is_generic }
                    }
                    _ => Infer, // shouldn't happen
                }
            }
            TyKind::BareFn(ref barefn) => BareFunction(box barefn.clean(cx)),
            TyKind::Infer | TyKind::Err => Infer,
            TyKind::Typeof(..) => panic!("unimplemented type {:?}", self.kind),
        }
    }
}

/// Returns `None` if the type could not be normalized
fn normalize(cx: &DocContext<'tcx>, ty: Ty<'_>) -> Option<Ty<'tcx>> {
    // HACK: low-churn fix for #79459 while we wait for a trait normalization fix
    if !cx.tcx.sess.opts.debugging_opts.normalize_docs {
        return None;
    }

    use crate::rustc_trait_selection::infer::TyCtxtInferExt;
    use crate::rustc_trait_selection::traits::query::normalize::AtExt;
    use rustc_middle::traits::ObligationCause;

    // Try to normalize `<X as Y>::T` to a type
    let lifted = ty.lift_to_tcx(cx.tcx).unwrap();
    let normalized = cx.tcx.infer_ctxt().enter(|infcx| {
        infcx
            .at(&ObligationCause::dummy(), cx.param_env.get())
            .normalize(lifted)
            .map(|resolved| infcx.resolve_vars_if_possible(resolved.value))
    });
    match normalized {
        Ok(normalized_value) => {
            debug!("normalized {:?} to {:?}", ty, normalized_value);
            Some(normalized_value)
        }
        Err(err) => {
            debug!("failed to normalize {:?}: {:?}", ty, err);
            None
        }
    }
}

impl<'tcx> Clean<Type> for Ty<'tcx> {
    fn clean(&self, cx: &DocContext<'_>) -> Type {
        debug!("cleaning type: {:?}", self);
        let ty = normalize(cx, self).unwrap_or(self);
        match *ty.kind() {
            ty::Never => Never,
            ty::Bool => Primitive(PrimitiveType::Bool),
            ty::Char => Primitive(PrimitiveType::Char),
            ty::Int(int_ty) => Primitive(int_ty.into()),
            ty::Uint(uint_ty) => Primitive(uint_ty.into()),
            ty::Float(float_ty) => Primitive(float_ty.into()),
            ty::Str => Primitive(PrimitiveType::Str),
            ty::Slice(ty) => Slice(box ty.clean(cx)),
            ty::Array(ty, n) => {
                let mut n = cx.tcx.lift(n).expect("array lift failed");
                n = n.eval(cx.tcx, ty::ParamEnv::reveal_all());
                let n = print_const(cx, n);
                Array(box ty.clean(cx), n)
            }
            ty::RawPtr(mt) => RawPointer(mt.mutbl, box mt.ty.clean(cx)),
            ty::Ref(r, ty, mutbl) => {
                BorrowedRef { lifetime: r.clean(cx), mutability: mutbl, type_: box ty.clean(cx) }
            }
            ty::FnDef(..) | ty::FnPtr(_) => {
                let ty = cx.tcx.lift(*self).expect("FnPtr lift failed");
                let sig = ty.fn_sig(cx.tcx);
                let def_id = DefId::local(CRATE_DEF_INDEX);
                BareFunction(box BareFunctionDecl {
                    unsafety: sig.unsafety(),
                    generic_params: Vec::new(),
                    decl: (def_id, sig).clean(cx),
                    abi: sig.abi(),
                })
            }
            ty::Adt(def, substs) => {
                let did = def.did;
                let kind = match def.adt_kind() {
                    AdtKind::Struct => TypeKind::Struct,
                    AdtKind::Union => TypeKind::Union,
                    AdtKind::Enum => TypeKind::Enum,
                };
                inline::record_extern_fqn(cx, did, kind);
                let path = external_path(cx, cx.tcx.item_name(did), None, false, vec![], substs);
                ResolvedPath { path, param_names: None, did, is_generic: false }
            }
            ty::Foreign(did) => {
                inline::record_extern_fqn(cx, did, TypeKind::Foreign);
                let path = external_path(
                    cx,
                    cx.tcx.item_name(did),
                    None,
                    false,
                    vec![],
                    InternalSubsts::empty(),
                );
                ResolvedPath { path, param_names: None, did, is_generic: false }
            }
            ty::Dynamic(ref obj, ref reg) => {
                // HACK: pick the first `did` as the `did` of the trait object. Someone
                // might want to implement "native" support for marker-trait-only
                // trait objects.
                let mut dids = obj.principal_def_id().into_iter().chain(obj.auto_traits());
                let did = dids
                    .next()
                    .unwrap_or_else(|| panic!("found trait object `{:?}` with no traits?", self));
                let substs = match obj.principal() {
                    Some(principal) => principal.skip_binder().substs,
                    // marker traits have no substs.
                    _ => cx.tcx.intern_substs(&[]),
                };

                inline::record_extern_fqn(cx, did, TypeKind::Trait);

                let mut param_names = vec![];
                if let Some(b) = reg.clean(cx) {
                    param_names.push(GenericBound::Outlives(b));
                }
                for did in dids {
                    let empty = cx.tcx.intern_substs(&[]);
                    let path =
                        external_path(cx, cx.tcx.item_name(did), Some(did), false, vec![], empty);
                    inline::record_extern_fqn(cx, did, TypeKind::Trait);
                    let bound = GenericBound::TraitBound(
                        PolyTrait {
                            trait_: ResolvedPath {
                                path,
                                param_names: None,
                                did,
                                is_generic: false,
                            },
                            generic_params: Vec::new(),
                        },
                        hir::TraitBoundModifier::None,
                    );
                    param_names.push(bound);
                }

                let mut bindings = vec![];
                for pb in obj.projection_bounds() {
                    bindings.push(TypeBinding {
                        name: cx.tcx.associated_item(pb.item_def_id()).ident.name,
                        kind: TypeBindingKind::Equality { ty: pb.skip_binder().ty.clean(cx) },
                    });
                }

                let path =
                    external_path(cx, cx.tcx.item_name(did), Some(did), false, bindings, substs);
                ResolvedPath { path, param_names: Some(param_names), did, is_generic: false }
            }
            ty::Tuple(ref t) => {
                Tuple(t.iter().map(|t| t.expect_ty()).collect::<Vec<_>>().clean(cx))
            }

            ty::Projection(ref data) => data.clean(cx),

            ty::Param(ref p) => {
                if let Some(bounds) = cx.impl_trait_bounds.borrow_mut().remove(&p.index.into()) {
                    ImplTrait(bounds)
                } else {
                    Generic(p.name)
                }
            }

            ty::Opaque(def_id, substs) => {
                // Grab the "TraitA + TraitB" from `impl TraitA + TraitB`,
                // by looking up the bounds associated with the def_id.
                let substs = cx.tcx.lift(substs).expect("Opaque lift failed");
                let bounds = cx
                    .tcx
                    .explicit_item_bounds(def_id)
                    .iter()
                    .map(|(bound, _)| bound.subst(cx.tcx, substs))
                    .collect::<Vec<_>>();
                let mut regions = vec![];
                let mut has_sized = false;
                let mut bounds = bounds
                    .iter()
                    .filter_map(|bound| {
                        // Note: The substs of opaque types can contain unbound variables,
                        // meaning that we have to use `ignore_quantifiers_with_unbound_vars` here.
                        let bound_predicate = bound.bound_atom_with_opt_escaping(cx.tcx);
                        let trait_ref = match bound_predicate.skip_binder() {
                            ty::PredicateAtom::Trait(tr, _constness) => {
                                bound_predicate.rebind(tr.trait_ref)
                            }
                            ty::PredicateAtom::TypeOutlives(ty::OutlivesPredicate(_ty, reg)) => {
                                if let Some(r) = reg.clean(cx) {
                                    regions.push(GenericBound::Outlives(r));
                                }
                                return None;
                            }
                            _ => return None,
                        };

                        if let Some(sized) = cx.tcx.lang_items().sized_trait() {
                            if trait_ref.def_id() == sized {
                                has_sized = true;
                                return None;
                            }
                        }

                        let bounds: Vec<_> = bounds
                            .iter()
                            .filter_map(|bound| {
                                if let ty::PredicateAtom::Projection(proj) =
                                    bound.bound_atom_with_opt_escaping(cx.tcx).skip_binder()
                                {
                                    if proj.projection_ty.trait_ref(cx.tcx)
                                        == trait_ref.skip_binder()
                                    {
                                        Some(TypeBinding {
                                            name: cx
                                                .tcx
                                                .associated_item(proj.projection_ty.item_def_id)
                                                .ident
                                                .name,
                                            kind: TypeBindingKind::Equality {
                                                ty: proj.ty.clean(cx),
                                            },
                                        })
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect();

                        Some((trait_ref, &bounds[..]).clean(cx))
                    })
                    .collect::<Vec<_>>();
                bounds.extend(regions);
                if !has_sized && !bounds.is_empty() {
                    bounds.insert(0, GenericBound::maybe_sized(cx));
                }
                ImplTrait(bounds)
            }

            ty::Closure(..) | ty::Generator(..) => Tuple(vec![]), // FIXME(pcwalton)

            ty::Bound(..) => panic!("Bound"),
            ty::Placeholder(..) => panic!("Placeholder"),
            ty::GeneratorWitness(..) => panic!("GeneratorWitness"),
            ty::Infer(..) => panic!("Infer"),
            ty::Error(_) => panic!("Error"),
        }
    }
}

impl<'tcx> Clean<Constant> for ty::Const<'tcx> {
    fn clean(&self, cx: &DocContext<'_>) -> Constant {
        Constant {
            type_: self.ty.clean(cx),
            expr: format!("{}", self),
            value: None,
            is_literal: false,
        }
    }
}

impl Clean<Item> for hir::StructField<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let what_rustc_thinks = Item::from_hir_id_and_parts(
            self.hir_id,
            Some(self.ident.name),
            StructFieldItem(self.ty.clean(cx)),
            cx,
        );
        // Don't show `pub` for fields on enum variants; they are always public
        Item { visibility: self.vis.clean(cx), ..what_rustc_thinks }
    }
}

impl Clean<Item> for ty::FieldDef {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let what_rustc_thinks = Item::from_def_id_and_parts(
            self.did,
            Some(self.ident.name),
            StructFieldItem(cx.tcx.type_of(self.did).clean(cx)),
            cx,
        );
        // Don't show `pub` for fields on enum variants; they are always public
        Item { visibility: self.vis.clean(cx), ..what_rustc_thinks }
    }
}

impl Clean<Visibility> for hir::Visibility<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Visibility {
        match self.node {
            hir::VisibilityKind::Public => Visibility::Public,
            hir::VisibilityKind::Inherited => Visibility::Inherited,
            hir::VisibilityKind::Crate(_) => {
                let krate = DefId::local(CRATE_DEF_INDEX);
                Visibility::Restricted(krate)
            }
            hir::VisibilityKind::Restricted { ref path, .. } => {
                let path = path.clean(cx);
                let did = register_res(cx, path.res);
                Visibility::Restricted(did)
            }
        }
    }
}

impl Clean<Visibility> for ty::Visibility {
    fn clean(&self, _cx: &DocContext<'_>) -> Visibility {
        match *self {
            ty::Visibility::Public => Visibility::Public,
            // NOTE: this is not quite right: `ty` uses `Invisible` to mean 'private',
            // while rustdoc really does mean inherited. That means that for enum variants, such as
            // `pub enum E { V }`, `V` will be marked as `Public` by `ty`, but as `Inherited` by rustdoc.
            // This is the main reason `impl Clean for hir::Visibility` still exists; various parts of clean
            // override `tcx.visibility` explicitly to make sure this distinction is captured.
            ty::Visibility::Invisible => Visibility::Inherited,
            ty::Visibility::Restricted(module) => Visibility::Restricted(module),
        }
    }
}

impl Clean<VariantStruct> for rustc_hir::VariantData<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> VariantStruct {
        VariantStruct {
            struct_type: doctree::struct_type_from_def(self),
            fields: self.fields().iter().map(|x| x.clean(cx)).collect(),
            fields_stripped: false,
        }
    }
}

impl Clean<Item> for doctree::Variant<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let what_rustc_thinks = Item::from_hir_id_and_parts(
            self.id,
            Some(self.name),
            VariantItem(Variant { kind: self.def.clean(cx) }),
            cx,
        );
        // don't show `pub` for variants, which are always public
        Item { visibility: Inherited, ..what_rustc_thinks }
    }
}

impl Clean<Item> for ty::VariantDef {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let kind = match self.ctor_kind {
            CtorKind::Const => VariantKind::CLike,
            CtorKind::Fn => VariantKind::Tuple(
                self.fields.iter().map(|f| cx.tcx.type_of(f.did).clean(cx)).collect(),
            ),
            CtorKind::Fictive => VariantKind::Struct(VariantStruct {
                struct_type: doctree::Plain,
                fields_stripped: false,
                fields: self
                    .fields
                    .iter()
                    .map(|field| {
                        let name = Some(field.ident.name);
                        let kind = StructFieldItem(cx.tcx.type_of(field.did).clean(cx));
                        let what_rustc_thinks =
                            Item::from_def_id_and_parts(field.did, name, kind, cx);
                        // don't show `pub` for fields, which are always public
                        Item { visibility: Visibility::Inherited, ..what_rustc_thinks }
                    })
                    .collect(),
            }),
        };
        let what_rustc_thinks = Item::from_def_id_and_parts(
            self.def_id,
            Some(self.ident.name),
            VariantItem(Variant { kind }),
            cx,
        );
        // don't show `pub` for fields, which are always public
        Item { visibility: Inherited, ..what_rustc_thinks }
    }
}

impl Clean<VariantKind> for hir::VariantData<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> VariantKind {
        match self {
            hir::VariantData::Struct(..) => VariantKind::Struct(self.clean(cx)),
            hir::VariantData::Tuple(..) => {
                VariantKind::Tuple(self.fields().iter().map(|x| x.ty.clean(cx)).collect())
            }
            hir::VariantData::Unit(..) => VariantKind::CLike,
        }
    }
}

impl Clean<Span> for rustc_span::Span {
    fn clean(&self, _cx: &DocContext<'_>) -> Span {
        Span::from_rustc_span(*self)
    }
}

impl Clean<Path> for hir::Path<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Path {
        Path {
            global: self.is_global(),
            res: self.res,
            segments: if self.is_global() { &self.segments[1..] } else { &self.segments }.clean(cx),
        }
    }
}

impl Clean<GenericArgs> for hir::GenericArgs<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> GenericArgs {
        if self.parenthesized {
            let output = self.bindings[0].ty().clean(cx);
            GenericArgs::Parenthesized {
                inputs: self.inputs().clean(cx),
                output: if output != Type::Tuple(Vec::new()) { Some(output) } else { None },
            }
        } else {
            GenericArgs::AngleBracketed {
                args: self
                    .args
                    .iter()
                    .map(|arg| match arg {
                        hir::GenericArg::Lifetime(lt) if !lt.is_elided() => {
                            GenericArg::Lifetime(lt.clean(cx))
                        }
                        hir::GenericArg::Lifetime(_) => GenericArg::Lifetime(Lifetime::elided()),
                        hir::GenericArg::Type(ty) => GenericArg::Type(ty.clean(cx)),
                        hir::GenericArg::Const(ct) => GenericArg::Const(ct.clean(cx)),
                    })
                    .collect(),
                bindings: self.bindings.clean(cx),
            }
        }
    }
}

impl Clean<PathSegment> for hir::PathSegment<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> PathSegment {
        PathSegment { name: self.ident.name, args: self.generic_args().clean(cx) }
    }
}

impl Clean<String> for Ident {
    #[inline]
    fn clean(&self, cx: &DocContext<'_>) -> String {
        self.name.clean(cx)
    }
}

impl Clean<String> for Symbol {
    #[inline]
    fn clean(&self, _: &DocContext<'_>) -> String {
        self.to_string()
    }
}

impl Clean<BareFunctionDecl> for hir::BareFnTy<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> BareFunctionDecl {
        let (generic_params, decl) = enter_impl_trait(cx, || {
            (self.generic_params.clean(cx), (&*self.decl, &self.param_names[..]).clean(cx))
        });
        BareFunctionDecl { unsafety: self.unsafety, abi: self.abi, decl, generic_params }
    }
}

impl Clean<Vec<Item>> for (&hir::Item<'_>, Option<Symbol>) {
    fn clean(&self, cx: &DocContext<'_>) -> Vec<Item> {
        use hir::ItemKind;

        let (item, renamed) = self;
        let def_id = cx.tcx.hir().local_def_id(item.hir_id).to_def_id();
        let mut name = renamed.unwrap_or_else(|| cx.tcx.hir().name(item.hir_id));
        cx.with_param_env(def_id, || {
            let kind = match item.kind {
                ItemKind::Static(ty, mutability, body_id) => StaticItem(Static {
                    type_: ty.clean(cx),
                    mutability,
                    expr: print_const_expr(cx, body_id),
                }),
                ItemKind::Const(ty, body_id) => ConstantItem(Constant {
                    type_: ty.clean(cx),
                    expr: print_const_expr(cx, body_id),
                    value: print_evaluated_const(cx, def_id),
                    is_literal: is_literal_expr(cx, body_id.hir_id),
                }),
                ItemKind::OpaqueTy(ref ty) => OpaqueTyItem(OpaqueTy {
                    bounds: ty.bounds.clean(cx),
                    generics: ty.generics.clean(cx),
                }),
                ItemKind::TyAlias(ty, ref generics) => {
                    let rustdoc_ty = ty.clean(cx);
                    let item_type = rustdoc_ty.def_id().and_then(|did| inline::build_ty(cx, did));
                    TypedefItem(
                        Typedef { type_: rustdoc_ty, generics: generics.clean(cx), item_type },
                        false,
                    )
                }
                ItemKind::Enum(ref def, ref generics) => EnumItem(Enum {
                    variants: def.variants.iter().map(|v| v.clean(cx)).collect(),
                    generics: generics.clean(cx),
                    variants_stripped: false,
                }),
                ItemKind::TraitAlias(ref generics, bounds) => TraitAliasItem(TraitAlias {
                    generics: generics.clean(cx),
                    bounds: bounds.clean(cx),
                }),
                ItemKind::Union(ref variant_data, ref generics) => UnionItem(Union {
                    struct_type: doctree::struct_type_from_def(&variant_data),
                    generics: generics.clean(cx),
                    fields: variant_data.fields().clean(cx),
                    fields_stripped: false,
                }),
                ItemKind::Struct(ref variant_data, ref generics) => StructItem(Struct {
                    struct_type: doctree::struct_type_from_def(&variant_data),
                    generics: generics.clean(cx),
                    fields: variant_data.fields().clean(cx),
                    fields_stripped: false,
                }),
                ItemKind::Impl { .. } => return clean_impl(item, cx),
                // proc macros can have a name set by attributes
                ItemKind::Fn(ref sig, ref generics, body_id) => {
                    clean_fn_or_proc_macro(item, sig, generics, body_id, &mut name, cx)
                }
                hir::ItemKind::Trait(is_auto, unsafety, ref generics, ref bounds, ref item_ids) => {
                    let items = item_ids
                        .iter()
                        .map(|ti| cx.tcx.hir().trait_item(ti.id).clean(cx))
                        .collect();
                    let attrs = item.attrs.clean(cx);
                    let is_spotlight = attrs.has_doc_flag(sym::spotlight);
                    TraitItem(Trait {
                        unsafety,
                        items,
                        generics: generics.clean(cx),
                        bounds: bounds.clean(cx),
                        is_spotlight,
                        is_auto: is_auto.clean(cx),
                    })
                }
                ItemKind::ExternCrate(orig_name) => {
                    return clean_extern_crate(item, name, orig_name, cx);
                }
                _ => unreachable!("not yet converted"),
            };

            vec![Item::from_def_id_and_parts(def_id, Some(name), kind, cx)]
        })
    }
}

impl Clean<Item> for hir::Variant<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let kind = VariantItem(Variant { kind: self.data.clean(cx) });
        let what_rustc_thinks =
            Item::from_hir_id_and_parts(self.id, Some(self.ident.name), kind, cx);
        // don't show `pub` for variants, which are always public
        Item { visibility: Inherited, ..what_rustc_thinks }
    }
}

impl Clean<ImplPolarity> for ty::ImplPolarity {
    fn clean(&self, _: &DocContext<'_>) -> ImplPolarity {
        match self {
            &ty::ImplPolarity::Positive |
            // FIXME: do we want to do something else here?
            &ty::ImplPolarity::Reservation => ImplPolarity::Positive,
            &ty::ImplPolarity::Negative => ImplPolarity::Negative,
        }
    }
}

fn clean_impl(impl_: &hir::Item<'_>, cx: &DocContext<'_>) -> Vec<Item> {
    let mut ret = Vec::new();
    let (trait_, items, for_, unsafety, generics) = match &impl_.kind {
        hir::ItemKind::Impl { of_trait, items, self_ty, unsafety, generics, .. } => {
            (of_trait, items, self_ty, *unsafety, generics)
        }
        _ => unreachable!(),
    };
    let trait_ = trait_.clean(cx);
    let items = items.iter().map(|ii| cx.tcx.hir().impl_item(ii.id).clean(cx)).collect::<Vec<_>>();
    let def_id = cx.tcx.hir().local_def_id(impl_.hir_id);

    // If this impl block is an implementation of the Deref trait, then we
    // need to try inlining the target's inherent impl blocks as well.
    if trait_.def_id() == cx.tcx.lang_items().deref_trait() {
        build_deref_target_impls(cx, &items, &mut ret);
    }

    let provided: FxHashSet<Symbol> = trait_
        .def_id()
        .map(|did| cx.tcx.provided_trait_methods(did).map(|meth| meth.ident.name).collect())
        .unwrap_or_default();

    let for_ = for_.clean(cx);
    let type_alias = for_.def_id().and_then(|did| match cx.tcx.def_kind(did) {
        DefKind::TyAlias => Some(cx.tcx.type_of(did).clean(cx)),
        _ => None,
    });
    let make_item = |trait_: Option<Type>, for_: Type, items: Vec<Item>| {
        let kind = ImplItem(Impl {
            unsafety,
            generics: generics.clean(cx),
            provided_trait_methods: provided.clone(),
            trait_,
            for_,
            items,
            polarity: Some(cx.tcx.impl_polarity(def_id).clean(cx)),
            synthetic: false,
            blanket_impl: None,
        });
        Item::from_hir_id_and_parts(impl_.hir_id, None, kind, cx)
    };
    if let Some(type_alias) = type_alias {
        ret.push(make_item(trait_.clone(), type_alias, items.clone()));
    }
    ret.push(make_item(trait_, for_, items));
    ret
}

fn clean_extern_crate(
    krate: &hir::Item<'_>,
    name: Symbol,
    orig_name: Option<Symbol>,
    cx: &DocContext<'_>,
) -> Vec<Item> {
    // this is the ID of the `extern crate` statement
    let def_id = cx.tcx.hir().local_def_id(krate.hir_id);
    let cnum = cx.tcx.extern_mod_stmt_cnum(def_id).unwrap_or(LOCAL_CRATE);
    // this is the ID of the crate itself
    let crate_def_id = DefId { krate: cnum, index: CRATE_DEF_INDEX };
    let please_inline = krate.vis.node.is_pub()
        && krate.attrs.iter().any(|a| {
            a.has_name(sym::doc)
                && match a.meta_item_list() {
                    Some(l) => attr::list_contains_name(&l, sym::inline),
                    None => false,
                }
        });

    if please_inline {
        let mut visited = FxHashSet::default();

        let res = Res::Def(DefKind::Mod, crate_def_id);

        if let Some(items) = inline::try_inline(
            cx,
            cx.tcx.parent_module(krate.hir_id).to_def_id(),
            res,
            name,
            Some(krate.attrs),
            &mut visited,
        ) {
            return items;
        }
    }
    // FIXME: using `from_def_id_and_kind` breaks `rustdoc/masked` for some reason
    vec![Item {
        name: None,
        attrs: krate.attrs.clean(cx),
        source: krate.span.clean(cx),
        def_id: crate_def_id,
        visibility: krate.vis.clean(cx),
        kind: box ExternCrateItem(name, orig_name),
    }]
}

impl Clean<Vec<Item>> for doctree::Import<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Vec<Item> {
        // We need this comparison because some imports (for std types for example)
        // are "inserted" as well but directly by the compiler and they should not be
        // taken into account.
        if self.span.ctxt().outer_expn_data().kind == ExpnKind::AstPass(AstPass::StdImports) {
            return Vec::new();
        }

        let (doc_meta_item, please_inline) = self.attrs.lists(sym::doc).get_word_attr(sym::inline);
        let pub_underscore = self.vis.node.is_pub() && self.name == kw::Underscore;

        if pub_underscore && please_inline {
            rustc_errors::struct_span_err!(
                cx.tcx.sess,
                doc_meta_item.unwrap().span(),
                E0780,
                "anonymous imports cannot be inlined"
            )
            .span_label(self.span, "anonymous import")
            .emit();
        }

        // We consider inlining the documentation of `pub use` statements, but we
        // forcefully don't inline if this is not public or if the
        // #[doc(no_inline)] attribute is present.
        // Don't inline doc(hidden) imports so they can be stripped at a later stage.
        let mut denied = !self.vis.node.is_pub()
            || pub_underscore
            || self.attrs.iter().any(|a| {
                a.has_name(sym::doc)
                    && match a.meta_item_list() {
                        Some(l) => {
                            attr::list_contains_name(&l, sym::no_inline)
                                || attr::list_contains_name(&l, sym::hidden)
                        }
                        None => false,
                    }
            });
        // Also check whether imports were asked to be inlined, in case we're trying to re-export a
        // crate in Rust 2018+
        let path = self.path.clean(cx);
        let inner = if self.glob {
            if !denied {
                let mut visited = FxHashSet::default();
                if let Some(items) = inline::try_inline_glob(cx, path.res, &mut visited) {
                    return items;
                }
            }
            Import::new_glob(resolve_use_source(cx, path), true)
        } else {
            let name = self.name;
            if !please_inline {
                if let Res::Def(DefKind::Mod, did) = path.res {
                    if !did.is_local() && did.index == CRATE_DEF_INDEX {
                        // if we're `pub use`ing an extern crate root, don't inline it unless we
                        // were specifically asked for it
                        denied = true;
                    }
                }
            }
            if !denied {
                let mut visited = FxHashSet::default();

                if let Some(mut items) = inline::try_inline(
                    cx,
                    cx.tcx.parent_module(self.id).to_def_id(),
                    path.res,
                    name,
                    Some(self.attrs),
                    &mut visited,
                ) {
                    items.push(Item {
                        name: None,
                        attrs: self.attrs.clean(cx),
                        source: self.span.clean(cx),
                        def_id: cx.tcx.hir().local_def_id(self.id).to_def_id(),
                        visibility: self.vis.clean(cx),
                        kind: box ImportItem(Import::new_simple(
                            self.name,
                            resolve_use_source(cx, path),
                            false,
                        )),
                    });
                    return items;
                }
            }
            Import::new_simple(name, resolve_use_source(cx, path), true)
        };

        vec![Item {
            name: None,
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id).to_def_id(),
            visibility: self.vis.clean(cx),
            kind: box ImportItem(inner),
        }]
    }
}

impl Clean<Item> for (&hir::ForeignItem<'_>, Option<Symbol>) {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let (item, renamed) = self;
        cx.with_param_env(cx.tcx.hir().local_def_id(item.hir_id).to_def_id(), || {
            let kind = match item.kind {
                hir::ForeignItemKind::Fn(ref decl, ref names, ref generics) => {
                    let abi = cx.tcx.hir().get_foreign_abi(item.hir_id);
                    let (generics, decl) = enter_impl_trait(cx, || {
                        (generics.clean(cx), (&**decl, &names[..]).clean(cx))
                    });
                    let (all_types, ret_types) = get_all_types(&generics, &decl, cx);
                    ForeignFunctionItem(Function {
                        decl,
                        generics,
                        header: hir::FnHeader {
                            unsafety: hir::Unsafety::Unsafe,
                            abi,
                            constness: hir::Constness::NotConst,
                            asyncness: hir::IsAsync::NotAsync,
                        },
                        all_types,
                        ret_types,
                    })
                }
                hir::ForeignItemKind::Static(ref ty, mutability) => ForeignStaticItem(Static {
                    type_: ty.clean(cx),
                    mutability,
                    expr: String::new(),
                }),
                hir::ForeignItemKind::Type => ForeignTypeItem,
            };

            Item::from_hir_id_and_parts(
                item.hir_id,
                Some(renamed.unwrap_or(item.ident.name)),
                kind,
                cx,
            )
        })
    }
}

impl Clean<Item> for (&hir::MacroDef<'_>, Option<Symbol>) {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let (item, renamed) = self;
        let name = renamed.unwrap_or(item.ident.name);
        let tts = item.ast.body.inner_tokens().trees().collect::<Vec<_>>();
        // Extract the spans of all matchers. They represent the "interface" of the macro.
        let matchers = tts.chunks(4).map(|arm| arm[0].span()).collect::<Vec<_>>();
        let source = if item.ast.macro_rules {
            format!(
                "macro_rules! {} {{\n{}}}",
                name,
                matchers
                    .iter()
                    .map(|span| { format!("    {} => {{ ... }};\n", span.to_src(cx)) })
                    .collect::<String>(),
            )
        } else {
            let vis = item.vis.clean(cx);
            let def_id = cx.tcx.hir().local_def_id(item.hir_id).to_def_id();

            if matchers.len() <= 1 {
                format!(
                    "{}macro {}{} {{\n    ...\n}}",
                    vis.print_with_space(cx.tcx, def_id),
                    name,
                    matchers.iter().map(|span| span.to_src(cx)).collect::<String>(),
                )
            } else {
                format!(
                    "{}macro {} {{\n{}}}",
                    vis.print_with_space(cx.tcx, def_id),
                    name,
                    matchers
                        .iter()
                        .map(|span| { format!("    {} => {{ ... }},\n", span.to_src(cx)) })
                        .collect::<String>(),
                )
            }
        };

        Item::from_hir_id_and_parts(
            item.hir_id,
            Some(name),
            MacroItem(Macro { source, imported_from: None }),
            cx,
        )
    }
}

impl Clean<TypeBinding> for hir::TypeBinding<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> TypeBinding {
        TypeBinding { name: self.ident.name, kind: self.kind.clean(cx) }
    }
}

impl Clean<TypeBindingKind> for hir::TypeBindingKind<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> TypeBindingKind {
        match *self {
            hir::TypeBindingKind::Equality { ref ty } => {
                TypeBindingKind::Equality { ty: ty.clean(cx) }
            }
            hir::TypeBindingKind::Constraint { ref bounds } => {
                TypeBindingKind::Constraint { bounds: bounds.iter().map(|b| b.clean(cx)).collect() }
            }
        }
    }
}

enum SimpleBound {
    TraitBound(Vec<PathSegment>, Vec<SimpleBound>, Vec<GenericParamDef>, hir::TraitBoundModifier),
    Outlives(Lifetime),
}

impl From<GenericBound> for SimpleBound {
    fn from(bound: GenericBound) -> Self {
        match bound.clone() {
            GenericBound::Outlives(l) => SimpleBound::Outlives(l),
            GenericBound::TraitBound(t, mod_) => match t.trait_ {
                Type::ResolvedPath { path, param_names, .. } => SimpleBound::TraitBound(
                    path.segments,
                    param_names.map_or_else(Vec::new, |v| {
                        v.iter().map(|p| SimpleBound::from(p.clone())).collect()
                    }),
                    t.generic_params,
                    mod_,
                ),
                _ => panic!("Unexpected bound {:?}", bound),
            },
        }
    }
}
