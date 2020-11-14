//! This module contains the "cleaned" pieces of the AST, and the functions
//! that clean them.

mod auto_trait;
mod blanket_impl;
pub mod cfg;
pub mod inline;
mod simplify;
pub mod types;
pub mod utils;

use rustc_ast as ast;
use rustc_attr as attr;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::{CrateNum, DefId, CRATE_DEF_INDEX};
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
use rustc_span::{self, ExpnKind, Pos};
use rustc_typeck::hir_ty_to_ty;

use std::collections::hash_map::Entry;
use std::default::Default;
use std::hash::Hash;
use std::rc::Rc;
use std::{mem, vec};

use crate::core::{self, DocContext, ImplTraitParam};
use crate::doctree;

use utils::*;

pub use utils::{get_auto_trait_and_blanket_impls, krate, register_res};

pub use self::types::FnRetTy::*;
pub use self::types::ItemKind::*;
pub use self::types::SelfTy::*;
pub use self::types::Type::*;
pub use self::types::Visibility::{Inherited, Public};
pub use self::types::*;

const FN_OUTPUT_NAME: &str = "Output";

pub trait Clean<T> {
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
                return prim.map(|p| (def_id, p, attrs));
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
                            as_primitive(path.res).map(|(_, prim, attrs)| {
                                // Pretend the primitive is local.
                                (cx.tcx.hir().local_def_id(id.id).to_def_id(), prim, attrs)
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
                    if let Some(v) = attr.value_str() {
                        if attr.has_name(sym::keyword) {
                            if v.is_doc_keyword() {
                                keyword = Some(v.to_string());
                                break;
                            }
                            // FIXME: should warn on unknown keywords?
                        }
                    }
                }
                return keyword.map(|p| (def_id, p, attrs));
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
                            as_keyword(path.res).map(|(_, prim, attrs)| {
                                (cx.tcx.hir().local_def_id(id.id).to_def_id(), prim, attrs)
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
            name: cx.tcx.crate_name(*self).to_string(),
            src: krate_src,
            attrs: cx.tcx.get_attrs(root).clean(cx),
            primitives,
            keywords,
        }
    }
}

impl Clean<Item> for doctree::Module<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let name = if self.name.is_some() {
            self.name.expect("No name provided").clean(cx)
        } else {
            String::new()
        };

        // maintain a stack of mod ids, for doc comment path resolution
        // but we also need to resolve the module's own docs based on whether its docs were written
        // inside or outside the module, so check for that
        let attrs = self.attrs.clean(cx);

        let mut items: Vec<Item> = vec![];
        items.extend(self.extern_crates.iter().flat_map(|x| x.clean(cx)));
        items.extend(self.imports.iter().flat_map(|x| x.clean(cx)));
        items.extend(self.structs.iter().map(|x| x.clean(cx)));
        items.extend(self.unions.iter().map(|x| x.clean(cx)));
        items.extend(self.enums.iter().map(|x| x.clean(cx)));
        items.extend(self.fns.iter().map(|x| x.clean(cx)));
        items.extend(self.foreigns.iter().map(|x| x.clean(cx)));
        items.extend(self.mods.iter().map(|x| x.clean(cx)));
        items.extend(self.typedefs.iter().map(|x| x.clean(cx)));
        items.extend(self.opaque_tys.iter().map(|x| x.clean(cx)));
        items.extend(self.statics.iter().map(|x| x.clean(cx)));
        items.extend(self.constants.iter().map(|x| x.clean(cx)));
        items.extend(self.traits.iter().map(|x| x.clean(cx)));
        items.extend(self.impls.iter().flat_map(|x| x.clean(cx)));
        items.extend(self.macros.iter().map(|x| x.clean(cx)));
        items.extend(self.proc_macros.iter().map(|x| x.clean(cx)));
        items.extend(self.trait_aliases.iter().map(|x| x.clean(cx)));

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

        Item {
            name: Some(name),
            attrs,
            source: span.clean(cx),
            visibility: self.vis.clean(cx),
            stability: cx.stability(self.id),
            deprecation: cx.deprecation(self.id).clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id).to_def_id(),
            kind: ModuleItem(Module { is_crate: self.is_crate, items }),
        }
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
                ty::BrNamed(_, name) => Some(GenericParamDef {
                    name: name.to_string(),
                    kind: GenericParamDefKind::Lifetime,
                }),
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
        Lifetime(self.name.ident().to_string())
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
                    Lifetime(s)
                } else {
                    Lifetime(self.name.ident().to_string())
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
        Lifetime(self.name.to_string())
    }
}

impl Clean<Option<Lifetime>> for ty::RegionKind {
    fn clean(&self, cx: &DocContext<'_>) -> Option<Lifetime> {
        match *self {
            ty::ReStatic => Some(Lifetime::statik()),
            ty::ReLateBound(_, ty::BrNamed(_, name)) => Some(Lifetime(name.to_string())),
            ty::ReEarlyBound(ref data) => Some(Lifetime(data.name.clean(cx))),

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
        match self.skip_binders() {
            ty::PredicateAtom::Trait(pred, _) => Some(ty::Binder::bind(pred).clean(cx)),
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
            name: cx.tcx.associated_item(self.item_def_id).ident.name.clean(cx),
            self_type: box self.self_ty().clean(cx),
            trait_: box trait_,
        }
    }
}

impl Clean<GenericParamDef> for ty::GenericParamDef {
    fn clean(&self, cx: &DocContext<'_>) -> GenericParamDef {
        let (name, kind) = match self.kind {
            ty::GenericParamDefKind::Lifetime => {
                (self.name.to_string(), GenericParamDefKind::Lifetime)
            }
            ty::GenericParamDefKind::Type { has_default, synthetic, .. } => {
                let default =
                    if has_default { Some(cx.tcx.type_of(self.def_id).clean(cx)) } else { None };
                (
                    self.name.clean(cx),
                    GenericParamDefKind::Type {
                        did: self.def_id,
                        bounds: vec![], // These are filled in from the where-clauses.
                        default,
                        synthetic,
                    },
                )
            }
            ty::GenericParamDefKind::Const { .. } => (
                self.name.clean(cx),
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
                    s
                } else {
                    self.name.ident().to_string()
                };
                (name, GenericParamDefKind::Lifetime)
            }
            hir::GenericParamKind::Type { ref default, synthetic } => (
                self.name.ident().name.clean(cx),
                GenericParamDefKind::Type {
                    did: cx.tcx.hir().local_def_id(self.hir_id).to_def_id(),
                    bounds: self.bounds.clean(cx),
                    default: default.clean(cx),
                    synthetic,
                },
            ),
            hir::GenericParamKind::Const { ref ty } => (
                self.name.ident().name.clean(cx),
                GenericParamDefKind::Const {
                    did: cx.tcx.hir().local_def_id(self.hir_id).to_def_id(),
                    ty: ty.clean(cx),
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
        for p in self.params.iter().filter(|p| !is_impl_trait(p)) {
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
        let mut impl_trait_proj = FxHashMap::<u32, Vec<(DefId, String, Ty<'tcx>)>>::default();

        let where_predicates = preds
            .predicates
            .iter()
            .flat_map(|(p, _)| {
                let mut projection = None;
                let param_idx = (|| {
                    match p.skip_binders() {
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
                                projection = Some(ty::Binder::bind(p));
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
                            impl_trait_proj.entry(param_idx).or_default().push((
                                trait_did,
                                name.to_string(),
                                rhs,
                            ));
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
                        simplify::merge_bounds(cx, &mut bounds, trait_did, &name, &rhs.clean(cx));
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
                    sized_params.insert(g.clone());
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
                    ty: Type::Generic(tp.name.clone()),
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

impl<'a> Clean<Method>
    for (&'a hir::FnSig<'a>, &'a hir::Generics<'a>, hir::BodyId, Option<hir::Defaultness>)
{
    fn clean(&self, cx: &DocContext<'_>) -> Method {
        let (generics, decl) =
            enter_impl_trait(cx, || (self.1.clean(cx), (&*self.0.decl, self.2).clean(cx)));
        let (all_types, ret_types) = get_all_types(&generics, &decl, cx);
        Method { decl, generics, header: self.0.header, defaultness: self.3, all_types, ret_types }
    }
}

impl Clean<Item> for doctree::Function<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let (generics, decl) =
            enter_impl_trait(cx, || (self.generics.clean(cx), (self.decl, self.body).clean(cx)));

        let did = cx.tcx.hir().local_def_id(self.id);
        let constness = if is_const_fn(cx.tcx, did.to_def_id())
            && !is_unstable_const_fn(cx.tcx, did.to_def_id()).is_some()
        {
            hir::Constness::Const
        } else {
            hir::Constness::NotConst
        };
        let (all_types, ret_types) = get_all_types(&generics, &decl, cx);
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            visibility: self.vis.clean(cx),
            stability: cx.stability(self.id),
            deprecation: cx.deprecation(self.id).clean(cx),
            def_id: did.to_def_id(),
            kind: FunctionItem(Function {
                decl,
                generics,
                header: hir::FnHeader { constness, ..self.header },
                all_types,
                ret_types,
            }),
        }
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
                    let mut name = self.1.get(i).map(|ident| ident.to_string()).unwrap_or_default();
                    if name.is_empty() {
                        name = "_".to_string();
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
                        name: names.next().map_or(String::new(), |name| name.to_string()),
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

impl Clean<Item> for doctree::Trait<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let attrs = self.attrs.clean(cx);
        let is_spotlight = attrs.has_doc_flag(sym::spotlight);
        Item {
            name: Some(self.name.clean(cx)),
            attrs,
            source: self.span.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id).to_def_id(),
            visibility: self.vis.clean(cx),
            stability: cx.stability(self.id),
            deprecation: cx.deprecation(self.id).clean(cx),
            kind: TraitItem(Trait {
                auto: self.is_auto.clean(cx),
                unsafety: self.unsafety,
                items: self.items.iter().map(|ti| ti.clean(cx)).collect(),
                generics: self.generics.clean(cx),
                bounds: self.bounds.clean(cx),
                is_spotlight,
                is_auto: self.is_auto.clean(cx),
            }),
        }
    }
}

impl Clean<Item> for doctree::TraitAlias<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let attrs = self.attrs.clean(cx);
        Item {
            name: Some(self.name.clean(cx)),
            attrs,
            source: self.span.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id).to_def_id(),
            visibility: self.vis.clean(cx),
            stability: cx.stability(self.id),
            deprecation: cx.deprecation(self.id).clean(cx),
            kind: TraitAliasItem(TraitAlias {
                generics: self.generics.clean(cx),
                bounds: self.bounds.clean(cx),
            }),
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
        let local_did = cx.tcx.hir().local_def_id(self.hir_id);
        let kind = match self.kind {
            hir::TraitItemKind::Const(ref ty, default) => {
                AssocConstItem(ty.clean(cx), default.map(|e| print_const_expr(cx, e)))
            }
            hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Provided(body)) => {
                let mut m = (sig, &self.generics, body, None).clean(cx);
                if m.header.constness == hir::Constness::Const
                    && is_unstable_const_fn(cx.tcx, local_did.to_def_id()).is_some()
                {
                    m.header.constness = hir::Constness::NotConst;
                }
                MethodItem(m)
            }
            hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Required(ref names)) => {
                let (generics, decl) = enter_impl_trait(cx, || {
                    (self.generics.clean(cx), (&*sig.decl, &names[..]).clean(cx))
                });
                let (all_types, ret_types) = get_all_types(&generics, &decl, cx);
                let mut t = TyMethod { header: sig.header, decl, generics, all_types, ret_types };
                if t.header.constness == hir::Constness::Const
                    && is_unstable_const_fn(cx.tcx, local_did.to_def_id()).is_some()
                {
                    t.header.constness = hir::Constness::NotConst;
                }
                TyMethodItem(t)
            }
            hir::TraitItemKind::Type(ref bounds, ref default) => {
                AssocTypeItem(bounds.clean(cx), default.clean(cx))
            }
        };
        Item {
            name: Some(self.ident.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: local_did.to_def_id(),
            visibility: Visibility::Inherited,
            stability: get_stability(cx, local_did.to_def_id()),
            deprecation: get_deprecation(cx, local_did.to_def_id()),
            kind,
        }
    }
}

impl Clean<Item> for hir::ImplItem<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let local_did = cx.tcx.hir().local_def_id(self.hir_id);
        let kind = match self.kind {
            hir::ImplItemKind::Const(ref ty, expr) => {
                AssocConstItem(ty.clean(cx), Some(print_const_expr(cx, expr)))
            }
            hir::ImplItemKind::Fn(ref sig, body) => {
                let mut m = (sig, &self.generics, body, Some(self.defaultness)).clean(cx);
                if m.header.constness == hir::Constness::Const
                    && is_unstable_const_fn(cx.tcx, local_did.to_def_id()).is_some()
                {
                    m.header.constness = hir::Constness::NotConst;
                }
                MethodItem(m)
            }
            hir::ImplItemKind::TyAlias(ref ty) => {
                let type_ = ty.clean(cx);
                let item_type = type_.def_id().and_then(|did| inline::build_ty(cx, did));
                TypedefItem(Typedef { type_, generics: Generics::default(), item_type }, true)
            }
        };
        Item {
            name: Some(self.ident.name.clean(cx)),
            source: self.span.clean(cx),
            attrs: self.attrs.clean(cx),
            def_id: local_did.to_def_id(),
            visibility: self.vis.clean(cx),
            stability: get_stability(cx, local_did.to_def_id()),
            deprecation: get_deprecation(cx, local_did.to_def_id()),
            kind,
        }
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
                        decl.inputs.values[0].type_ = Generic(String::from("Self"));
                    } else if let ty::Ref(_, ty, _) = *self_arg_ty.kind() {
                        if ty == self_ty {
                            match decl.inputs.values[0].type_ {
                                BorrowedRef { ref mut type_, .. } => {
                                    **type_ = Generic(String::from("Self"))
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
                    MethodItem(Method {
                        generics,
                        decl,
                        header: hir::FnHeader {
                            unsafety: sig.unsafety(),
                            abi: sig.abi(),
                            constness,
                            asyncness,
                        },
                        defaultness,
                        all_types,
                        ret_types,
                    })
                } else {
                    TyMethodItem(TyMethod {
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
                let my_name = self.ident.name.clean(cx);

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
                                Generic(ref s) if *s == "Self" => {}
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

        let visibility = match self.container {
            ty::ImplContainer(_) => self.vis.clean(cx),
            ty::TraitContainer(_) => Inherited,
        };

        Item {
            name: Some(self.ident.name.clean(cx)),
            visibility,
            stability: get_stability(cx, self.def_id),
            deprecation: get_deprecation(cx, self.def_id),
            def_id: self.def_id,
            attrs: inline::load_attrs(cx, self.def_id).clean(cx),
            source: cx.tcx.def_span(self.def_id).clean(cx),
            kind,
        }
    }
}

impl Clean<Type> for hir::Ty<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Type {
        use rustc_hir::*;

        match self.kind {
            TyKind::Never => Never,
            TyKind::Ptr(ref m) => RawPointer(m.mutbl, box m.ty.clean(cx)),
            TyKind::Rptr(ref l, ref m) => {
                let lifetime = if l.is_elided() { None } else { Some(l.clean(cx)) };
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
            TyKind::Path(hir::QPath::Resolved(None, ref path)) => {
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
                                    let lifetime =
                                        generic_args.args.iter().find_map(|arg| match arg {
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
                                    let type_ =
                                        generic_args.args.iter().find_map(|arg| match arg {
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
                                    let const_param_def_id =
                                        cx.tcx.hir().local_def_id(param.hir_id);
                                    let mut j = 0;
                                    let const_ =
                                        generic_args.args.iter().find_map(|arg| match arg {
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
                                        ct_substs
                                            .insert(const_param_def_id.to_def_id(), ct.clean(cx));
                                    }
                                    // FIXME(const_generics:defaults)
                                    indices.consts += 1;
                                }
                            }
                        }
                    }
                    return cx.enter_alias(ty_substs, lt_substs, ct_substs, || ty.clean(cx));
                }
                resolve_type(cx, path.clean(cx), self.hir_id)
            }
            TyKind::Path(hir::QPath::Resolved(Some(ref qself), ref p)) => {
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
                    name: p.segments.last().expect("segments were empty").ident.name.clean(cx),
                    self_type: box qself.clean(cx),
                    trait_: box resolve_type(cx, trait_path, self.hir_id),
                }
            }
            TyKind::Path(hir::QPath::TypeRelative(ref qself, ref segment)) => {
                let mut res = Res::Err;
                let ty = hir_ty_to_ty(cx.tcx, self);
                if let ty::Projection(proj) = ty.kind() {
                    res = Res::Def(DefKind::Trait, proj.trait_ref(cx.tcx).def_id);
                }
                let trait_path = hir::Path { span: self.span, res, segments: &[] };
                Type::QPath {
                    name: segment.ident.name.clean(cx),
                    self_type: box qself.clean(cx),
                    trait_: box resolve_type(cx, trait_path.clean(cx), self.hir_id),
                }
            }
            TyKind::Path(hir::QPath::LangItem(..)) => {
                bug!("clean: requiring documentation of lang item")
            }
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

impl<'tcx> Clean<Type> for Ty<'tcx> {
    fn clean(&self, cx: &DocContext<'_>) -> Type {
        debug!("cleaning type: {:?}", self);
        match *self.kind() {
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
                        name: cx.tcx.associated_item(pb.item_def_id()).ident.name.clean(cx),
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
                    Generic(p.name.to_string())
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
                        let trait_ref = match bound
                            .bound_atom_with_opt_escaping(cx.tcx)
                            .skip_binder()
                        {
                            ty::PredicateAtom::Trait(tr, _constness) => {
                                ty::Binder::bind(tr.trait_ref)
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
                                                .name
                                                .clean(cx),
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
        let local_did = cx.tcx.hir().local_def_id(self.hir_id);

        Item {
            name: Some(self.ident.name).clean(cx),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            visibility: self.vis.clean(cx),
            stability: get_stability(cx, local_did.to_def_id()),
            deprecation: get_deprecation(cx, local_did.to_def_id()),
            def_id: local_did.to_def_id(),
            kind: StructFieldItem(self.ty.clean(cx)),
        }
    }
}

impl Clean<Item> for ty::FieldDef {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        Item {
            name: Some(self.ident.name).clean(cx),
            attrs: cx.tcx.get_attrs(self.did).clean(cx),
            source: cx.tcx.def_span(self.did).clean(cx),
            visibility: self.vis.clean(cx),
            stability: get_stability(cx, self.did),
            deprecation: get_deprecation(cx, self.did),
            def_id: self.did,
            kind: StructFieldItem(cx.tcx.type_of(self.did).clean(cx)),
        }
    }
}

impl Clean<Visibility> for hir::Visibility<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Visibility {
        match self.node {
            hir::VisibilityKind::Public => Visibility::Public,
            hir::VisibilityKind::Inherited => Visibility::Inherited,
            hir::VisibilityKind::Crate(_) => Visibility::Crate,
            hir::VisibilityKind::Restricted { ref path, .. } => {
                let path = path.clean(cx);
                let did = register_res(cx, path.res);
                Visibility::Restricted(did, path)
            }
        }
    }
}

impl Clean<Visibility> for ty::Visibility {
    fn clean(&self, _: &DocContext<'_>) -> Visibility {
        if *self == ty::Visibility::Public { Public } else { Inherited }
    }
}

impl Clean<Item> for doctree::Struct<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id).to_def_id(),
            visibility: self.vis.clean(cx),
            stability: cx.stability(self.id),
            deprecation: cx.deprecation(self.id).clean(cx),
            kind: StructItem(Struct {
                struct_type: self.struct_type,
                generics: self.generics.clean(cx),
                fields: self.fields.clean(cx),
                fields_stripped: false,
            }),
        }
    }
}

impl Clean<Item> for doctree::Union<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id).to_def_id(),
            visibility: self.vis.clean(cx),
            stability: cx.stability(self.id),
            deprecation: cx.deprecation(self.id).clean(cx),
            kind: UnionItem(Union {
                struct_type: self.struct_type,
                generics: self.generics.clean(cx),
                fields: self.fields.clean(cx),
                fields_stripped: false,
            }),
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

impl Clean<Item> for doctree::Enum<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id).to_def_id(),
            visibility: self.vis.clean(cx),
            stability: cx.stability(self.id),
            deprecation: cx.deprecation(self.id).clean(cx),
            kind: EnumItem(Enum {
                variants: self.variants.iter().map(|v| v.clean(cx)).collect(),
                generics: self.generics.clean(cx),
                variants_stripped: false,
            }),
        }
    }
}

impl Clean<Item> for doctree::Variant<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            visibility: Inherited,
            stability: cx.stability(self.id),
            deprecation: cx.deprecation(self.id).clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id).to_def_id(),
            kind: VariantItem(Variant { kind: self.def.clean(cx) }),
        }
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
                    .map(|field| Item {
                        source: cx.tcx.def_span(field.did).clean(cx),
                        name: Some(field.ident.name.clean(cx)),
                        attrs: cx.tcx.get_attrs(field.did).clean(cx),
                        visibility: field.vis.clean(cx),
                        def_id: field.did,
                        stability: get_stability(cx, field.did),
                        deprecation: get_deprecation(cx, field.did),
                        kind: StructFieldItem(cx.tcx.type_of(field.did).clean(cx)),
                    })
                    .collect(),
            }),
        };
        Item {
            name: Some(self.ident.clean(cx)),
            attrs: inline::load_attrs(cx, self.def_id).clean(cx),
            source: cx.tcx.def_span(self.def_id).clean(cx),
            visibility: Inherited,
            def_id: self.def_id,
            kind: VariantItem(Variant { kind }),
            stability: get_stability(cx, self.def_id),
            deprecation: get_deprecation(cx, self.def_id),
        }
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
    fn clean(&self, cx: &DocContext<'_>) -> Span {
        if self.is_dummy() {
            return Span::empty();
        }

        // Get the macro invocation instead of the definition,
        // in case the span is result of a macro expansion.
        // (See rust-lang/rust#39726)
        let span = self.source_callsite();

        let sm = cx.sess().source_map();
        let filename = sm.span_to_filename(span);
        let lo = sm.lookup_char_pos(span.lo());
        let hi = sm.lookup_char_pos(span.hi());
        Span {
            filename,
            cnum: lo.file.cnum,
            loline: lo.line,
            locol: lo.col.to_usize(),
            hiline: hi.line,
            hicol: hi.col.to_usize(),
            original: span,
        }
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
        PathSegment { name: self.ident.name.clean(cx), args: self.generic_args().clean(cx) }
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

impl Clean<Item> for doctree::Typedef<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let type_ = self.ty.clean(cx);
        let item_type = type_.def_id().and_then(|did| inline::build_ty(cx, did));
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id).to_def_id(),
            visibility: self.vis.clean(cx),
            stability: cx.stability(self.id),
            deprecation: cx.deprecation(self.id).clean(cx),
            kind: TypedefItem(Typedef { type_, generics: self.gen.clean(cx), item_type }, false),
        }
    }
}

impl Clean<Item> for doctree::OpaqueTy<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id).to_def_id(),
            visibility: self.vis.clean(cx),
            stability: cx.stability(self.id),
            deprecation: cx.deprecation(self.id).clean(cx),
            kind: OpaqueTyItem(OpaqueTy {
                bounds: self.opaque_ty.bounds.clean(cx),
                generics: self.opaque_ty.generics.clean(cx),
            }),
        }
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

impl Clean<Item> for doctree::Static<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        debug!("cleaning static {}: {:?}", self.name.clean(cx), self);
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id).to_def_id(),
            visibility: self.vis.clean(cx),
            stability: cx.stability(self.id),
            deprecation: cx.deprecation(self.id).clean(cx),
            kind: StaticItem(Static {
                type_: self.type_.clean(cx),
                mutability: self.mutability,
                expr: print_const_expr(cx, self.expr),
            }),
        }
    }
}

impl Clean<Item> for doctree::Constant<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let def_id = cx.tcx.hir().local_def_id(self.id);

        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: def_id.to_def_id(),
            visibility: self.vis.clean(cx),
            stability: cx.stability(self.id),
            deprecation: cx.deprecation(self.id).clean(cx),
            kind: ConstantItem(Constant {
                type_: self.type_.clean(cx),
                expr: print_const_expr(cx, self.expr),
                value: print_evaluated_const(cx, def_id.to_def_id()),
                is_literal: is_literal_expr(cx, self.expr.hir_id),
            }),
        }
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

impl Clean<Vec<Item>> for doctree::Impl<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Vec<Item> {
        let mut ret = Vec::new();
        let trait_ = self.trait_.clean(cx);
        let items = self.items.iter().map(|ii| ii.clean(cx)).collect::<Vec<_>>();
        let def_id = cx.tcx.hir().local_def_id(self.id);

        // If this impl block is an implementation of the Deref trait, then we
        // need to try inlining the target's inherent impl blocks as well.
        if trait_.def_id() == cx.tcx.lang_items().deref_trait() {
            build_deref_target_impls(cx, &items, &mut ret);
        }

        let provided: FxHashSet<String> = trait_
            .def_id()
            .map(|did| {
                cx.tcx.provided_trait_methods(did).map(|meth| meth.ident.to_string()).collect()
            })
            .unwrap_or_default();

        let for_ = self.for_.clean(cx);
        let type_alias = for_.def_id().and_then(|did| match cx.tcx.def_kind(did) {
            DefKind::TyAlias => Some(cx.tcx.type_of(did).clean(cx)),
            _ => None,
        });
        let make_item = |trait_: Option<Type>, for_: Type, items: Vec<Item>| Item {
            name: None,
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: def_id.to_def_id(),
            visibility: self.vis.clean(cx),
            stability: cx.stability(self.id),
            deprecation: cx.deprecation(self.id).clean(cx),
            kind: ImplItem(Impl {
                unsafety: self.unsafety,
                generics: self.generics.clean(cx),
                provided_trait_methods: provided.clone(),
                trait_,
                for_,
                items,
                polarity: Some(cx.tcx.impl_polarity(def_id).clean(cx)),
                synthetic: false,
                blanket_impl: None,
            }),
        };
        if let Some(type_alias) = type_alias {
            ret.push(make_item(trait_.clone(), type_alias, items.clone()));
        }
        ret.push(make_item(trait_, for_, items));
        ret
    }
}

impl Clean<Vec<Item>> for doctree::ExternCrate<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Vec<Item> {
        let please_inline = self.vis.node.is_pub()
            && self.attrs.iter().any(|a| {
                a.has_name(sym::doc)
                    && match a.meta_item_list() {
                        Some(l) => attr::list_contains_name(&l, sym::inline),
                        None => false,
                    }
            });

        if please_inline {
            let mut visited = FxHashSet::default();

            let res = Res::Def(DefKind::Mod, DefId { krate: self.cnum, index: CRATE_DEF_INDEX });

            if let Some(items) = inline::try_inline(
                cx,
                cx.tcx.parent_module(self.hir_id).to_def_id(),
                res,
                self.name,
                Some(self.attrs),
                &mut visited,
            ) {
                return items;
            }
        }

        vec![Item {
            name: None,
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: DefId { krate: self.cnum, index: CRATE_DEF_INDEX },
            visibility: self.vis.clean(cx),
            stability: None,
            deprecation: None,
            kind: ExternCrateItem(self.name.clean(cx), self.path.clone()),
        }]
    }
}

impl Clean<Vec<Item>> for doctree::Import<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Vec<Item> {
        // We need this comparison because some imports (for std types for example)
        // are "inserted" as well but directly by the compiler and they should not be
        // taken into account.
        if self.span.ctxt().outer_expn_data().kind == ExpnKind::AstPass(AstPass::StdImports) {
            return Vec::new();
        }

        // We consider inlining the documentation of `pub use` statements, but we
        // forcefully don't inline if this is not public or if the
        // #[doc(no_inline)] attribute is present.
        // Don't inline doc(hidden) imports so they can be stripped at a later stage.
        let mut denied = !self.vis.node.is_pub()
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
        let please_inline = self.attrs.lists(sym::doc).has_word(sym::inline);
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
                        stability: None,
                        deprecation: None,
                        kind: ImportItem(Import::new_simple(
                            self.name.clean(cx),
                            resolve_use_source(cx, path),
                            false,
                        )),
                    });
                    return items;
                }
            }
            Import::new_simple(name.clean(cx), resolve_use_source(cx, path), true)
        };

        vec![Item {
            name: None,
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: DefId::local(CRATE_DEF_INDEX),
            visibility: self.vis.clean(cx),
            stability: None,
            deprecation: None,
            kind: ImportItem(inner),
        }]
    }
}

impl Clean<Item> for doctree::ForeignItem<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let kind = match self.kind {
            hir::ForeignItemKind::Fn(ref decl, ref names, ref generics) => {
                let abi = cx.tcx.hir().get_foreign_abi(self.id);
                let (generics, decl) =
                    enter_impl_trait(cx, || (generics.clean(cx), (&**decl, &names[..]).clean(cx)));
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
            hir::ForeignItemKind::Static(ref ty, mutbl) => ForeignStaticItem(Static {
                type_: ty.clean(cx),
                mutability: *mutbl,
                expr: String::new(),
            }),
            hir::ForeignItemKind::Type => ForeignTypeItem,
        };

        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id).to_def_id(),
            visibility: self.vis.clean(cx),
            stability: cx.stability(self.id),
            deprecation: cx.deprecation(self.id).clean(cx),
            kind,
        }
    }
}

impl Clean<Item> for doctree::Macro<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        let name = self.name.clean(cx);
        Item {
            name: Some(name.clone()),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            visibility: Public,
            stability: cx.stability(self.hid),
            deprecation: cx.deprecation(self.hid).clean(cx),
            def_id: self.def_id,
            kind: MacroItem(Macro {
                source: format!(
                    "macro_rules! {} {{\n{}}}",
                    name,
                    self.matchers
                        .iter()
                        .map(|span| { format!("    {} => {{ ... }};\n", span.to_src(cx)) })
                        .collect::<String>()
                ),
                imported_from: self.imported_from.clean(cx),
            }),
        }
    }
}

impl Clean<Item> for doctree::ProcMacro<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            visibility: Public,
            stability: cx.stability(self.id),
            deprecation: cx.deprecation(self.id).clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id).to_def_id(),
            kind: ProcMacroItem(ProcMacro { kind: self.kind, helpers: self.helpers.clean(cx) }),
        }
    }
}

impl Clean<Deprecation> for attr::Deprecation {
    fn clean(&self, _: &DocContext<'_>) -> Deprecation {
        Deprecation {
            since: self.since.map(|s| s.to_string()).filter(|s| !s.is_empty()),
            note: self.note.map(|n| n.to_string()).filter(|n| !n.is_empty()),
            is_since_rustc_version: self.is_since_rustc_version,
        }
    }
}

impl Clean<TypeBinding> for hir::TypeBinding<'_> {
    fn clean(&self, cx: &DocContext<'_>) -> TypeBinding {
        TypeBinding { name: self.ident.name.clean(cx), kind: self.kind.clean(cx) }
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
