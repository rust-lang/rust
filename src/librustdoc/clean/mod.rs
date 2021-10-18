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
use rustc_const_eval::const_eval::is_unstable_const_fn;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::{DefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_index::vec::{Idx, IndexVec};
use rustc_infer::infer::region_constraints::{Constraint, RegionConstraintData};
use rustc_middle::middle::resolve_lifetime as rl;
use rustc_middle::ty::fold::TypeFolder;
use rustc_middle::ty::subst::{InternalSubsts, Subst};
use rustc_middle::ty::{self, AdtKind, DefIdTree, Lift, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_span::hygiene::{AstPass, MacroKind};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{self, ExpnKind};
use rustc_target::spec::abi::Abi;
use rustc_typeck::check::intrinsic::intrinsic_operation_unsafety;
use rustc_typeck::hir_ty_to_ty;

use std::assert_matches::assert_matches;
use std::collections::hash_map::Entry;
use std::default::Default;
use std::hash::Hash;
use std::rc::Rc;
use std::{mem, vec};

use crate::core::{self, DocContext, ImplTraitParam};
use crate::doctree;
use crate::formats::item_type::ItemType;

use utils::*;

crate use utils::{get_auto_trait_and_blanket_impls, krate, register_res};

crate use self::types::FnRetTy::*;
crate use self::types::ItemKind::*;
crate use self::types::SelfTy::*;
crate use self::types::Type::*;
crate use self::types::Visibility::{Inherited, Public};
crate use self::types::*;

crate trait Clean<T> {
    fn clean(&self, cx: &mut DocContext<'_>) -> T;
}

impl<T: Clean<U>, U> Clean<Vec<U>> for [T] {
    fn clean(&self, cx: &mut DocContext<'_>) -> Vec<U> {
        self.iter().map(|x| x.clean(cx)).collect()
    }
}

impl<T: Clean<U>, U, V: Idx> Clean<IndexVec<V, U>> for IndexVec<V, T> {
    fn clean(&self, cx: &mut DocContext<'_>) -> IndexVec<V, U> {
        self.iter().map(|x| x.clean(cx)).collect()
    }
}

impl<T: Clean<U>, U> Clean<U> for &T {
    fn clean(&self, cx: &mut DocContext<'_>) -> U {
        (**self).clean(cx)
    }
}

impl<T: Clean<U>, U> Clean<U> for Rc<T> {
    fn clean(&self, cx: &mut DocContext<'_>) -> U {
        (**self).clean(cx)
    }
}

impl<T: Clean<U>, U> Clean<Option<U>> for Option<T> {
    fn clean(&self, cx: &mut DocContext<'_>) -> Option<U> {
        self.as_ref().map(|v| v.clean(cx))
    }
}

impl Clean<Item> for doctree::Module<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> Item {
        let mut items: Vec<Item> = vec![];
        items.extend(self.foreigns.iter().map(|x| x.clean(cx)));
        items.extend(self.mods.iter().map(|x| x.clean(cx)));
        items.extend(self.items.iter().map(|x| x.clean(cx)).flatten());

        // determine if we should display the inner contents or
        // the outer `mod` item for the source code.

        let span = Span::new({
            let where_outer = self.where_outer(cx.tcx);
            let sm = cx.sess().source_map();
            let outer = sm.lookup_char_pos(where_outer.lo());
            let inner = sm.lookup_char_pos(self.where_inner.lo());
            if outer.file.start_pos == inner.file.start_pos {
                // mod foo { ... }
                where_outer
            } else {
                // mod foo; (and a separate SourceFile for the contents)
                self.where_inner
            }
        });

        Item::from_hir_id_and_parts(
            self.id,
            Some(self.name),
            ModuleItem(Module { items, span }),
            cx,
        )
    }
}

impl Clean<Attributes> for [ast::Attribute] {
    fn clean(&self, _cx: &mut DocContext<'_>) -> Attributes {
        Attributes::from_ast(self, None)
    }
}

impl Clean<GenericBound> for hir::GenericBound<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> GenericBound {
        match *self {
            hir::GenericBound::Outlives(lt) => GenericBound::Outlives(lt.clean(cx)),
            hir::GenericBound::LangItemTrait(lang_item, span, _, generic_args) => {
                let def_id = cx.tcx.require_lang_item(lang_item, Some(span));

                let trait_ref = ty::TraitRef::identity(cx.tcx, def_id).skip_binder();

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

impl Clean<Path> for (ty::TraitRef<'_>, &[TypeBinding]) {
    fn clean(&self, cx: &mut DocContext<'_>) -> Path {
        let (trait_ref, bounds) = *self;
        let kind = cx.tcx.def_kind(trait_ref.def_id).into();
        if !matches!(kind, ItemType::Trait | ItemType::TraitAlias) {
            span_bug!(
                cx.tcx.def_span(trait_ref.def_id),
                "`TraitRef` had unexpected kind {:?}",
                kind
            );
        }
        inline::record_extern_fqn(cx, trait_ref.def_id, kind);
        let path = external_path(cx, trait_ref.def_id, true, bounds.to_vec(), trait_ref.substs);

        debug!("ty::TraitRef\n  subst: {:?}\n", trait_ref.substs);

        path
    }
}

impl Clean<Path> for ty::TraitRef<'tcx> {
    fn clean(&self, cx: &mut DocContext<'_>) -> Path {
        (*self, &[][..]).clean(cx)
    }
}

impl Clean<GenericBound> for (ty::PolyTraitRef<'_>, &[TypeBinding]) {
    fn clean(&self, cx: &mut DocContext<'_>) -> GenericBound {
        let (poly_trait_ref, bounds) = *self;
        let poly_trait_ref = poly_trait_ref.lift_to_tcx(cx.tcx).unwrap();

        // collect any late bound regions
        let late_bound_regions: Vec<_> = cx
            .tcx
            .collect_referenced_late_bound_regions(&poly_trait_ref)
            .into_iter()
            .filter_map(|br| match br {
                ty::BrNamed(_, name) => Some(GenericParamDef {
                    name,
                    kind: GenericParamDefKind::Lifetime { outlives: vec![] },
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
    fn clean(&self, cx: &mut DocContext<'_>) -> GenericBound {
        (*self, &[][..]).clean(cx)
    }
}

impl Clean<Lifetime> for hir::Lifetime {
    fn clean(&self, cx: &mut DocContext<'_>) -> Lifetime {
        let def = cx.tcx.named_region(self.hir_id);
        match def {
            Some(
                rl::Region::EarlyBound(_, node_id, _)
                | rl::Region::LateBound(_, _, node_id, _)
                | rl::Region::Free(_, node_id),
            ) => {
                if let Some(lt) = cx.lt_substs.get(&node_id).cloned() {
                    return lt;
                }
            }
            _ => {}
        }
        Lifetime(self.name.ident().name)
    }
}

impl Clean<Constant> for hir::ConstArg {
    fn clean(&self, cx: &mut DocContext<'_>) -> Constant {
        Constant {
            type_: cx
                .tcx
                .type_of(cx.tcx.hir().body_owner_def_id(self.value.body).to_def_id())
                .clean(cx),
            kind: ConstantKind::Anonymous { body: self.value.body },
        }
    }
}

impl Clean<Option<Lifetime>> for ty::RegionKind {
    fn clean(&self, _cx: &mut DocContext<'_>) -> Option<Lifetime> {
        match *self {
            ty::ReStatic => Some(Lifetime::statik()),
            ty::ReLateBound(_, ty::BoundRegion { kind: ty::BrNamed(_, name), .. }) => {
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
    fn clean(&self, cx: &mut DocContext<'_>) -> WherePredicate {
        match *self {
            hir::WherePredicate::BoundPredicate(ref wbp) => {
                let bound_params = wbp
                    .bound_generic_params
                    .into_iter()
                    .map(|param| {
                        // Higher-ranked params must be lifetimes.
                        // Higher-ranked lifetimes can't have bounds.
                        assert_matches!(
                            param,
                            hir::GenericParam {
                                kind: hir::GenericParamKind::Lifetime { .. },
                                bounds: [],
                                ..
                            }
                        );
                        Lifetime(param.name.ident().name)
                    })
                    .collect();
                WherePredicate::BoundPredicate {
                    ty: wbp.bounded_ty.clean(cx),
                    bounds: wbp.bounds.clean(cx),
                    bound_params,
                }
            }

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
    fn clean(&self, cx: &mut DocContext<'_>) -> Option<WherePredicate> {
        let bound_predicate = self.kind();
        match bound_predicate.skip_binder() {
            ty::PredicateKind::Trait(pred) => Some(bound_predicate.rebind(pred).clean(cx)),
            ty::PredicateKind::RegionOutlives(pred) => pred.clean(cx),
            ty::PredicateKind::TypeOutlives(pred) => pred.clean(cx),
            ty::PredicateKind::Projection(pred) => Some(pred.clean(cx)),
            ty::PredicateKind::ConstEvaluatable(..) => None,

            ty::PredicateKind::Subtype(..)
            | ty::PredicateKind::Coerce(..)
            | ty::PredicateKind::WellFormed(..)
            | ty::PredicateKind::ObjectSafe(..)
            | ty::PredicateKind::ClosureKind(..)
            | ty::PredicateKind::ConstEquate(..)
            | ty::PredicateKind::TypeWellFormedFromEnv(..) => panic!("not user writable"),
        }
    }
}

impl<'a> Clean<WherePredicate> for ty::PolyTraitPredicate<'a> {
    fn clean(&self, cx: &mut DocContext<'_>) -> WherePredicate {
        let poly_trait_ref = self.map_bound(|pred| pred.trait_ref);
        WherePredicate::BoundPredicate {
            ty: poly_trait_ref.skip_binder().self_ty().clean(cx),
            bounds: vec![poly_trait_ref.clean(cx)],
            bound_params: Vec::new(),
        }
    }
}

impl<'tcx> Clean<Option<WherePredicate>>
    for ty::OutlivesPredicate<ty::Region<'tcx>, ty::Region<'tcx>>
{
    fn clean(&self, cx: &mut DocContext<'_>) -> Option<WherePredicate> {
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
    fn clean(&self, cx: &mut DocContext<'_>) -> Option<WherePredicate> {
        let ty::OutlivesPredicate(ty, lt) = self;

        if let ty::ReEmpty(_) = lt {
            return None;
        }

        Some(WherePredicate::BoundPredicate {
            ty: ty.clean(cx),
            bounds: vec![GenericBound::Outlives(lt.clean(cx).expect("failed to clean lifetimes"))],
            bound_params: Vec::new(),
        })
    }
}

impl<'tcx> Clean<WherePredicate> for ty::ProjectionPredicate<'tcx> {
    fn clean(&self, cx: &mut DocContext<'_>) -> WherePredicate {
        let ty::ProjectionPredicate { projection_ty, ty } = self;
        WherePredicate::EqPredicate { lhs: projection_ty.clean(cx), rhs: ty.clean(cx) }
    }
}

impl<'tcx> Clean<Type> for ty::ProjectionTy<'tcx> {
    fn clean(&self, cx: &mut DocContext<'_>) -> Type {
        let lifted = self.lift_to_tcx(cx.tcx).unwrap();
        let trait_ = lifted.trait_ref(cx.tcx).clean(cx);
        let self_type = self.self_ty().clean(cx);
        Type::QPath {
            name: cx.tcx.associated_item(self.item_def_id).ident.name,
            self_def_id: self_type.def_id(),
            self_type: box self_type,
            trait_,
        }
    }
}

impl Clean<GenericParamDef> for ty::GenericParamDef {
    fn clean(&self, cx: &mut DocContext<'_>) -> GenericParamDef {
        let (name, kind) = match self.kind {
            ty::GenericParamDefKind::Lifetime => {
                (self.name, GenericParamDefKind::Lifetime { outlives: vec![] })
            }
            ty::GenericParamDefKind::Type { has_default, synthetic, .. } => {
                let default = if has_default {
                    let mut default = cx.tcx.type_of(self.def_id).clean(cx);

                    // We need to reassign the `self_def_id`, if there's a parent (which is the
                    // `Self` type), so we can properly render `<Self as X>` casts, because the
                    // information about which type `Self` is, is only present here, but not in
                    // the cleaning process of the type itself. To resolve this and have the
                    // `self_def_id` set, we override it here.
                    // See https://github.com/rust-lang/rust/issues/85454
                    if let QPath { ref mut self_def_id, .. } = default {
                        *self_def_id = cx.tcx.parent(self.def_id);
                    }

                    Some(default)
                } else {
                    None
                };
                (
                    self.name,
                    GenericParamDefKind::Type {
                        did: self.def_id,
                        bounds: vec![], // These are filled in from the where-clauses.
                        default: default.map(Box::new),
                        synthetic,
                    },
                )
            }
            ty::GenericParamDefKind::Const { has_default, .. } => (
                self.name,
                GenericParamDefKind::Const {
                    did: self.def_id,
                    ty: Box::new(cx.tcx.type_of(self.def_id).clean(cx)),
                    default: match has_default {
                        true => Some(Box::new(cx.tcx.const_param_default(self.def_id).to_string())),
                        false => None,
                    },
                },
            ),
        };

        GenericParamDef { name, kind }
    }
}

impl Clean<GenericParamDef> for hir::GenericParam<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> GenericParamDef {
        let (name, kind) = match self.kind {
            hir::GenericParamKind::Lifetime { .. } => {
                let outlives = self
                    .bounds
                    .iter()
                    .map(|bound| match bound {
                        hir::GenericBound::Outlives(lt) => lt.clean(cx),
                        _ => panic!(),
                    })
                    .collect();
                (self.name.ident().name, GenericParamDefKind::Lifetime { outlives })
            }
            hir::GenericParamKind::Type { ref default, synthetic } => (
                self.name.ident().name,
                GenericParamDefKind::Type {
                    did: cx.tcx.hir().local_def_id(self.hir_id).to_def_id(),
                    bounds: self.bounds.clean(cx),
                    default: default.clean(cx).map(Box::new),
                    synthetic,
                },
            ),
            hir::GenericParamKind::Const { ref ty, default } => (
                self.name.ident().name,
                GenericParamDefKind::Const {
                    did: cx.tcx.hir().local_def_id(self.hir_id).to_def_id(),
                    ty: Box::new(ty.clean(cx)),
                    default: default.map(|ct| {
                        let def_id = cx.tcx.hir().local_def_id(ct.hir_id);
                        Box::new(ty::Const::from_anon_const(cx.tcx, def_id).to_string())
                    }),
                },
            ),
        };

        GenericParamDef { name, kind }
    }
}

impl Clean<Generics> for hir::Generics<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> Generics {
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
            matches!(
                param.kind,
                hir::GenericParamKind::Lifetime { kind: hir::LifetimeParamKind::Elided }
            )
        }

        let impl_trait_params = self
            .params
            .iter()
            .filter(|param| is_impl_trait(param))
            .map(|param| {
                let param: GenericParamDef = param.clean(cx);
                match param.kind {
                    GenericParamDefKind::Lifetime { .. } => unreachable!(),
                    GenericParamDefKind::Type { did, ref bounds, .. } => {
                        cx.impl_trait_bounds.insert(did.into(), bounds.clone());
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
                WherePredicate::BoundPredicate {
                    ty: Generic(ref name), ref mut bounds, ..
                } => {
                    if bounds.is_empty() {
                        for param in &mut generics.params {
                            match param.kind {
                                GenericParamDefKind::Lifetime { .. } => {}
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
    fn clean(&self, cx: &mut DocContext<'_>) -> Generics {
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
                    let bound_p = p.kind();
                    match bound_p.skip_binder() {
                        ty::PredicateKind::Trait(pred) => {
                            if let ty::Param(param) = pred.self_ty().kind() {
                                return Some(param.index);
                            }
                        }
                        ty::PredicateKind::TypeOutlives(ty::OutlivesPredicate(ty, _reg)) => {
                            if let ty::Param(param) = ty.kind() {
                                return Some(param.index);
                            }
                        }
                        ty::PredicateKind::Projection(p) => {
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
            bounds.sort_by_key(|b| !matches!(b, GenericBound::TraitBound(..)));

            if let crate::core::ImplTraitParam::ParamIndex(idx) = param {
                if let Some(proj) = impl_trait_proj.remove(&idx) {
                    for (trait_did, name, rhs) in proj {
                        let rhs = rhs.clean(cx);
                        simplify::merge_bounds(cx, &mut bounds, trait_did, name, &rhs);
                    }
                }
            } else {
                unreachable!();
            }

            cx.impl_trait_bounds.insert(param, bounds);
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
            WP::BoundPredicate { ty: Generic(ref g), ref bounds, .. } => {
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
                    bound_params: Vec::new(),
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
    cx: &mut DocContext<'_>,
) -> ItemKind {
    let attrs = cx.tcx.hir().attrs(item.hir_id());
    let macro_kind = attrs.iter().find_map(|a| {
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
                *name = attrs
                    .lists(sym::proc_macro_derive)
                    .find_map(|mi| mi.ident())
                    .expect("proc-macro derives require a name")
                    .name;
            }

            let mut helpers = Vec::new();
            for mi in attrs.lists(sym::proc_macro_derive) {
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
            let def_id = item.def_id.to_def_id();
            func.header.constness =
                if cx.tcx.is_const_fn(def_id) && is_unstable_const_fn(cx.tcx, def_id).is_none() {
                    hir::Constness::Const
                } else {
                    hir::Constness::NotConst
                };
            FunctionItem(func)
        }
    }
}

impl<'a> Clean<Function> for (&'a hir::FnSig<'a>, &'a hir::Generics<'a>, hir::BodyId) {
    fn clean(&self, cx: &mut DocContext<'_>) -> Function {
        let (generics, decl) =
            enter_impl_trait(cx, |cx| (self.1.clean(cx), (&*self.0.decl, self.2).clean(cx)));
        Function { decl, generics, header: self.0.header }
    }
}

impl<'a> Clean<Arguments> for (&'a [hir::Ty<'a>], &'a [Ident]) {
    fn clean(&self, cx: &mut DocContext<'_>) -> Arguments {
        Arguments {
            values: self
                .0
                .iter()
                .enumerate()
                .map(|(i, ty)| {
                    let mut name = self.1.get(i).map_or(kw::Empty, |ident| ident.name);
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
    fn clean(&self, cx: &mut DocContext<'_>) -> Arguments {
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
    fn clean(&self, cx: &mut DocContext<'_>) -> FnDecl {
        FnDecl {
            inputs: (self.0.inputs, self.1).clean(cx),
            output: self.0.output.clean(cx),
            c_variadic: self.0.c_variadic,
        }
    }
}

impl<'tcx> Clean<FnDecl> for (DefId, ty::PolyFnSig<'tcx>) {
    fn clean(&self, cx: &mut DocContext<'_>) -> FnDecl {
        let (did, sig) = *self;
        let mut names = if did.is_local() { &[] } else { cx.tcx.fn_arg_names(did) }.iter();

        FnDecl {
            output: Return(sig.skip_binder().output().clean(cx)),
            c_variadic: sig.skip_binder().c_variadic,
            inputs: Arguments {
                values: sig
                    .skip_binder()
                    .inputs()
                    .iter()
                    .map(|t| Argument {
                        type_: t.clean(cx),
                        name: names.next().map_or(kw::Empty, |i| i.name),
                    })
                    .collect(),
            },
        }
    }
}

impl Clean<FnRetTy> for hir::FnRetTy<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> FnRetTy {
        match *self {
            Self::Return(ref typ) => Return(typ.clean(cx)),
            Self::DefaultReturn(..) => DefaultReturn,
        }
    }
}

impl Clean<bool> for hir::IsAuto {
    fn clean(&self, _: &mut DocContext<'_>) -> bool {
        match *self {
            hir::IsAuto::Yes => true,
            hir::IsAuto::No => false,
        }
    }
}

impl Clean<Path> for hir::TraitRef<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> Path {
        let path = self.path.clean(cx);
        register_res(cx, path.res);
        path
    }
}

impl Clean<PolyTrait> for hir::PolyTraitRef<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> PolyTrait {
        PolyTrait {
            trait_: self.trait_ref.clean(cx),
            generic_params: self.bound_generic_params.clean(cx),
        }
    }
}

impl Clean<Item> for hir::TraitItem<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> Item {
        let local_did = self.def_id.to_def_id();
        cx.with_param_env(local_did, |cx| {
            let inner = match self.kind {
                hir::TraitItemKind::Const(ref ty, default) => {
                    AssocConstItem(ty.clean(cx), default.map(|e| print_const_expr(cx.tcx, e)))
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
                    let (generics, decl) = enter_impl_trait(cx, |cx| {
                        (self.generics.clean(cx), (&*sig.decl, &names[..]).clean(cx))
                    });
                    let mut t = Function { header: sig.header, decl, generics };
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
            let what_rustc_thinks =
                Item::from_def_id_and_parts(local_did, Some(self.ident.name), inner, cx);
            // Trait items always inherit the trait's visibility -- we don't want to show `pub`.
            Item { visibility: Inherited, ..what_rustc_thinks }
        })
    }
}

impl Clean<Item> for hir::ImplItem<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> Item {
        let local_did = self.def_id.to_def_id();
        cx.with_param_env(local_did, |cx| {
            let inner = match self.kind {
                hir::ImplItemKind::Const(ref ty, expr) => {
                    AssocConstItem(ty.clean(cx), Some(print_const_expr(cx.tcx, expr)))
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
                hir::ImplItemKind::TyAlias(ref hir_ty) => {
                    let type_ = hir_ty.clean(cx);
                    let item_type = hir_ty_to_ty(cx.tcx, hir_ty).clean(cx);
                    TypedefItem(
                        Typedef {
                            type_,
                            generics: Generics::default(),
                            item_type: Some(item_type),
                        },
                        true,
                    )
                }
            };

            let what_rustc_thinks =
                Item::from_def_id_and_parts(local_did, Some(self.ident.name), inner, cx);
            let parent_item = cx.tcx.hir().expect_item(cx.tcx.hir().get_parent_item(self.hir_id()));
            if let hir::ItemKind::Impl(impl_) = &parent_item.kind {
                if impl_.of_trait.is_some() {
                    // Trait impl items always inherit the impl's visibility --
                    // we don't want to show `pub`.
                    Item { visibility: Inherited, ..what_rustc_thinks }
                } else {
                    what_rustc_thinks
                }
            } else {
                panic!("found impl item with non-impl parent {:?}", parent_item);
            }
        })
    }
}

impl Clean<Item> for ty::AssocItem {
    fn clean(&self, cx: &mut DocContext<'_>) -> Item {
        let tcx = cx.tcx;
        let kind = match self.kind {
            ty::AssocKind::Const => {
                let ty = tcx.type_of(self.def_id);
                let default = if self.defaultness.has_value() {
                    Some(inline::print_inlined_const(tcx, self.def_id))
                } else {
                    None
                };
                AssocConstItem(ty.clean(cx), default)
            }
            ty::AssocKind::Fn => {
                let generics =
                    (tcx.generics_of(self.def_id), tcx.explicit_predicates_of(self.def_id))
                        .clean(cx);
                let sig = tcx.fn_sig(self.def_id);
                let mut decl = (self.def_id, sig).clean(cx);

                if self.fn_has_self_parameter {
                    let self_ty = match self.container {
                        ty::ImplContainer(def_id) => tcx.type_of(def_id),
                        ty::TraitContainer(_) => tcx.types.self_param,
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
                if provided {
                    let constness = if tcx.is_const_fn_raw(self.def_id) {
                        hir::Constness::Const
                    } else {
                        hir::Constness::NotConst
                    };
                    let asyncness = tcx.asyncness(self.def_id);
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
                    })
                }
            }
            ty::AssocKind::Type => {
                let my_name = self.ident.name;

                if let ty::TraitContainer(_) = self.container {
                    let bounds = tcx.explicit_item_bounds(self.def_id);
                    let predicates = ty::GenericPredicates { parent: None, predicates: bounds };
                    let generics = (tcx.generics_of(self.def_id), predicates).clean(cx);
                    let mut bounds = generics
                        .where_predicates
                        .iter()
                        .filter_map(|pred| {
                            let (name, self_type, trait_, bounds) = match *pred {
                                WherePredicate::BoundPredicate {
                                    ty: QPath { ref name, ref self_type, ref trait_, .. },
                                    ref bounds,
                                    ..
                                } => (name, self_type, trait_, bounds),
                                _ => return None,
                            };
                            if *name != my_name {
                                return None;
                            }
                            if trait_.def_id() != self.container.id() {
                                return None;
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
                        Some(tcx.type_of(self.def_id))
                    } else {
                        None
                    };

                    AssocTypeItem(bounds, ty.clean(cx))
                } else {
                    // FIXME: when could this happen? Associated items in inherent impls?
                    let type_ = tcx.type_of(self.def_id).clean(cx);
                    TypedefItem(
                        Typedef {
                            type_,
                            generics: Generics { params: Vec::new(), where_predicates: Vec::new() },
                            item_type: None,
                        },
                        true,
                    )
                }
            }
        };

        Item::from_def_id_and_parts(self.def_id, Some(self.ident.name), kind, cx)
    }
}

fn clean_qpath(hir_ty: &hir::Ty<'_>, cx: &mut DocContext<'_>) -> Type {
    use rustc_hir::GenericParamCount;
    let hir::Ty { hir_id: _, span, ref kind } = *hir_ty;
    let qpath = match kind {
        hir::TyKind::Path(qpath) => qpath,
        _ => unreachable!(),
    };

    match qpath {
        hir::QPath::Resolved(None, ref path) => {
            if let Res::Def(DefKind::TyParam, did) = path.res {
                if let Some(new_ty) = cx.ty_substs.get(&did).cloned() {
                    return new_ty;
                }
                if let Some(bounds) = cx.impl_trait_bounds.remove(&did.into()) {
                    return ImplTrait(bounds);
                }
            }

            let mut alias = None;
            if let Res::Def(DefKind::TyAlias, def_id) = path.res {
                // Substitute private type aliases
                if let Some(def_id) = def_id.as_local() {
                    let hir_id = cx.tcx.hir().local_def_id_to_hir_id(def_id);
                    if !cx.cache.access_levels.is_exported(def_id.to_def_id()) {
                        alias = Some(&cx.tcx.hir().expect_item(hir_id).kind);
                    }
                }
            };

            if let Some(&hir::ItemKind::TyAlias(ref ty, ref generics)) = alias {
                let provided_params = &path.segments.last().expect("segments were empty");
                let mut ty_substs = FxHashMap::default();
                let mut lt_substs = FxHashMap::default();
                let mut ct_substs = FxHashMap::default();
                let generic_args = provided_params.args();
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
                return cx.enter_alias(ty_substs, lt_substs, ct_substs, |cx| ty.clean(cx));
            }
            let path = path.clean(cx);
            resolve_type(cx, path)
        }
        hir::QPath::Resolved(Some(ref qself), ref p) => {
            // Try to normalize `<X as Y>::T` to a type
            let ty = hir_ty_to_ty(cx.tcx, hir_ty);
            if let Some(normalized_value) = normalize(cx, ty) {
                return normalized_value.clean(cx);
            }

            let trait_segments = &p.segments[..p.segments.len() - 1];
            let trait_def = cx.tcx.associated_item(p.res.def_id()).container.id();
            let trait_ = self::Path {
                res: Res::Def(DefKind::Trait, trait_def),
                segments: trait_segments.clean(cx),
            };
            register_res(cx, trait_.res);
            Type::QPath {
                name: p.segments.last().expect("segments were empty").ident.name,
                self_def_id: Some(DefId::local(qself.hir_id.owner.local_def_index)),
                self_type: box qself.clean(cx),
                trait_,
            }
        }
        hir::QPath::TypeRelative(ref qself, ref segment) => {
            let ty = hir_ty_to_ty(cx.tcx, hir_ty);
            let res = match ty.kind() {
                ty::Projection(proj) => Res::Def(DefKind::Trait, proj.trait_ref(cx.tcx).def_id),
                // Rustdoc handles `ty::Error`s by turning them into `Type::Infer`s.
                ty::Error(_) => return Type::Infer,
                _ => bug!("clean: expected associated type, found `{:?}`", ty),
            };
            let trait_ = hir::Path { span, res, segments: &[] }.clean(cx);
            register_res(cx, trait_.res);
            Type::QPath {
                name: segment.ident.name,
                self_def_id: res.opt_def_id(),
                self_type: box qself.clean(cx),
                trait_,
            }
        }
        hir::QPath::LangItem(..) => bug!("clean: requiring documentation of lang item"),
    }
}

impl Clean<Type> for hir::Ty<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> Type {
        use rustc_hir::*;

        match self.kind {
            TyKind::Never => Primitive(PrimitiveType::Never),
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
                let item = cx.tcx.hir().item(item_id);
                if let hir::ItemKind::OpaqueTy(ref ty) = item.kind {
                    ImplTrait(ty.bounds.clean(cx))
                } else {
                    unreachable!()
                }
            }
            TyKind::Path(_) => clean_qpath(&self, cx),
            TyKind::TraitObject(ref bounds, ref lifetime, _) => {
                let bounds = bounds.iter().map(|bound| bound.clean(cx)).collect();
                let lifetime = if !lifetime.is_elided() { Some(lifetime.clean(cx)) } else { None };
                DynTrait(bounds, lifetime)
            }
            TyKind::BareFn(ref barefn) => BareFunction(box barefn.clean(cx)),
            // Rustdoc handles `TyKind::Err`s by turning them into `Type::Infer`s.
            TyKind::Infer | TyKind::Err => Infer,
            TyKind::Typeof(..) => panic!("unimplemented type {:?}", self.kind),
        }
    }
}

/// Returns `None` if the type could not be normalized
fn normalize(cx: &mut DocContext<'tcx>, ty: Ty<'_>) -> Option<Ty<'tcx>> {
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
            .at(&ObligationCause::dummy(), cx.param_env)
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
    fn clean(&self, cx: &mut DocContext<'_>) -> Type {
        trace!("cleaning type: {:?}", self);
        let ty = normalize(cx, self).unwrap_or(self);
        match *ty.kind() {
            ty::Never => Primitive(PrimitiveType::Never),
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
                    AdtKind::Struct => ItemType::Struct,
                    AdtKind::Union => ItemType::Union,
                    AdtKind::Enum => ItemType::Enum,
                };
                inline::record_extern_fqn(cx, did, kind);
                let path = external_path(cx, did, false, vec![], substs);
                ResolvedPath { path, did }
            }
            ty::Foreign(did) => {
                inline::record_extern_fqn(cx, did, ItemType::ForeignType);
                let path = external_path(cx, did, false, vec![], InternalSubsts::empty());
                ResolvedPath { path, did }
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

                inline::record_extern_fqn(cx, did, ItemType::Trait);

                let lifetime = reg.clean(cx);
                let mut bounds = vec![];

                for did in dids {
                    let empty = cx.tcx.intern_substs(&[]);
                    let path = external_path(cx, did, false, vec![], empty);
                    inline::record_extern_fqn(cx, did, ItemType::Trait);
                    let bound = PolyTrait { trait_: path, generic_params: Vec::new() };
                    bounds.push(bound);
                }

                let mut bindings = vec![];
                for pb in obj.projection_bounds() {
                    bindings.push(TypeBinding {
                        name: cx.tcx.associated_item(pb.item_def_id()).ident.name,
                        kind: TypeBindingKind::Equality { ty: pb.skip_binder().ty.clean(cx) },
                    });
                }

                let path = external_path(cx, did, false, bindings, substs);
                bounds.insert(0, PolyTrait { trait_: path, generic_params: Vec::new() });

                DynTrait(bounds, lifetime)
            }
            ty::Tuple(ref t) => {
                Tuple(t.iter().map(|t| t.expect_ty()).collect::<Vec<_>>().clean(cx))
            }

            ty::Projection(ref data) => data.clean(cx),

            ty::Param(ref p) => {
                if let Some(bounds) = cx.impl_trait_bounds.remove(&p.index.into()) {
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
                        let bound_predicate = bound.kind();
                        let trait_ref = match bound_predicate.skip_binder() {
                            ty::PredicateKind::Trait(tr) => bound_predicate.rebind(tr.trait_ref),
                            ty::PredicateKind::TypeOutlives(ty::OutlivesPredicate(_ty, reg)) => {
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
                                if let ty::PredicateKind::Projection(proj) =
                                    bound.kind().skip_binder()
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
    fn clean(&self, cx: &mut DocContext<'_>) -> Constant {
        // FIXME: instead of storing the stringified expression, store `self` directly instead.
        Constant {
            type_: self.ty.clean(cx),
            kind: ConstantKind::TyConst { expr: self.to_string() },
        }
    }
}

impl Clean<Item> for hir::FieldDef<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> Item {
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
    fn clean(&self, cx: &mut DocContext<'_>) -> Item {
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
    fn clean(&self, cx: &mut DocContext<'_>) -> Visibility {
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
    fn clean(&self, _cx: &mut DocContext<'_>) -> Visibility {
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
    fn clean(&self, cx: &mut DocContext<'_>) -> VariantStruct {
        VariantStruct {
            struct_type: CtorKind::from_hir(self),
            fields: self.fields().iter().map(|x| x.clean(cx)).collect(),
            fields_stripped: false,
        }
    }
}

impl Clean<Vec<Item>> for hir::VariantData<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> Vec<Item> {
        self.fields().iter().map(|x| x.clean(cx)).collect()
    }
}

impl Clean<Item> for ty::VariantDef {
    fn clean(&self, cx: &mut DocContext<'_>) -> Item {
        let kind = match self.ctor_kind {
            CtorKind::Const => Variant::CLike,
            CtorKind::Fn => Variant::Tuple(
                self.fields
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
            ),
            CtorKind::Fictive => Variant::Struct(VariantStruct {
                struct_type: CtorKind::Fictive,
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
        let what_rustc_thinks =
            Item::from_def_id_and_parts(self.def_id, Some(self.ident.name), VariantItem(kind), cx);
        // don't show `pub` for fields, which are always public
        Item { visibility: Inherited, ..what_rustc_thinks }
    }
}

impl Clean<Variant> for hir::VariantData<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> Variant {
        match self {
            hir::VariantData::Struct(..) => Variant::Struct(self.clean(cx)),
            hir::VariantData::Tuple(..) => Variant::Tuple(self.clean(cx)),
            hir::VariantData::Unit(..) => Variant::CLike,
        }
    }
}

impl Clean<Path> for hir::Path<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> Path {
        Path { res: self.res, segments: self.segments.clean(cx) }
    }
}

impl Clean<GenericArgs> for hir::GenericArgs<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> GenericArgs {
        if self.parenthesized {
            let output = self.bindings[0].ty().clean(cx);
            let output =
                if output != Type::Tuple(Vec::new()) { Some(Box::new(output)) } else { None };
            GenericArgs::Parenthesized { inputs: self.inputs().clean(cx), output }
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
                        hir::GenericArg::Const(ct) => GenericArg::Const(Box::new(ct.clean(cx))),
                        hir::GenericArg::Infer(_inf) => GenericArg::Infer,
                    })
                    .collect(),
                bindings: self.bindings.clean(cx),
            }
        }
    }
}

impl Clean<PathSegment> for hir::PathSegment<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> PathSegment {
        PathSegment { name: self.ident.name, args: self.args().clean(cx) }
    }
}

impl Clean<BareFunctionDecl> for hir::BareFnTy<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> BareFunctionDecl {
        let (generic_params, decl) = enter_impl_trait(cx, |cx| {
            (self.generic_params.clean(cx), (&*self.decl, self.param_names).clean(cx))
        });
        BareFunctionDecl { unsafety: self.unsafety, abi: self.abi, decl, generic_params }
    }
}

impl Clean<Vec<Item>> for (&hir::Item<'_>, Option<Symbol>) {
    fn clean(&self, cx: &mut DocContext<'_>) -> Vec<Item> {
        use hir::ItemKind;

        let (item, renamed) = self;
        let def_id = item.def_id.to_def_id();
        let mut name = renamed.unwrap_or_else(|| cx.tcx.hir().name(item.hir_id()));
        cx.with_param_env(def_id, |cx| {
            let kind = match item.kind {
                ItemKind::Static(ty, mutability, body_id) => {
                    StaticItem(Static { type_: ty.clean(cx), mutability, expr: Some(body_id) })
                }
                ItemKind::Const(ty, body_id) => ConstantItem(Constant {
                    type_: ty.clean(cx),
                    kind: ConstantKind::Local { body: body_id, def_id },
                }),
                ItemKind::OpaqueTy(ref ty) => OpaqueTyItem(OpaqueTy {
                    bounds: ty.bounds.clean(cx),
                    generics: ty.generics.clean(cx),
                }),
                ItemKind::TyAlias(hir_ty, ref generics) => {
                    let rustdoc_ty = hir_ty.clean(cx);
                    let ty = hir_ty_to_ty(cx.tcx, hir_ty).clean(cx);
                    TypedefItem(
                        Typedef {
                            type_: rustdoc_ty,
                            generics: generics.clean(cx),
                            item_type: Some(ty),
                        },
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
                    generics: generics.clean(cx),
                    fields: variant_data.fields().clean(cx),
                    fields_stripped: false,
                }),
                ItemKind::Struct(ref variant_data, ref generics) => StructItem(Struct {
                    struct_type: CtorKind::from_hir(variant_data),
                    generics: generics.clean(cx),
                    fields: variant_data.fields().clean(cx),
                    fields_stripped: false,
                }),
                ItemKind::Impl(ref impl_) => return clean_impl(impl_, item.hir_id(), cx),
                // proc macros can have a name set by attributes
                ItemKind::Fn(ref sig, ref generics, body_id) => {
                    clean_fn_or_proc_macro(item, sig, generics, body_id, &mut name, cx)
                }
                ItemKind::Macro(ref macro_def) => MacroItem(Macro {
                    source: display_macro_source(cx, name, &macro_def, def_id, &item.vis),
                }),
                ItemKind::Trait(is_auto, unsafety, ref generics, ref bounds, ref item_ids) => {
                    let items = item_ids
                        .iter()
                        .map(|ti| cx.tcx.hir().trait_item(ti.id).clean(cx))
                        .collect();
                    TraitItem(Trait {
                        unsafety,
                        items,
                        generics: generics.clean(cx),
                        bounds: bounds.clean(cx),
                        is_auto: is_auto.clean(cx),
                    })
                }
                ItemKind::ExternCrate(orig_name) => {
                    return clean_extern_crate(item, name, orig_name, cx);
                }
                ItemKind::Use(path, kind) => {
                    return clean_use_statement(item, name, path, kind, cx);
                }
                _ => unreachable!("not yet converted"),
            };

            vec![Item::from_def_id_and_parts(def_id, Some(name), kind, cx)]
        })
    }
}

impl Clean<Item> for hir::Variant<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> Item {
        let kind = VariantItem(self.data.clean(cx));
        let what_rustc_thinks =
            Item::from_hir_id_and_parts(self.id, Some(self.ident.name), kind, cx);
        // don't show `pub` for variants, which are always public
        Item { visibility: Inherited, ..what_rustc_thinks }
    }
}

impl Clean<bool> for ty::ImplPolarity {
    /// Returns whether the impl has negative polarity.
    fn clean(&self, _: &mut DocContext<'_>) -> bool {
        match self {
            &ty::ImplPolarity::Positive |
            // FIXME: do we want to do something else here?
            &ty::ImplPolarity::Reservation => false,
            &ty::ImplPolarity::Negative => true,
        }
    }
}

fn clean_impl(impl_: &hir::Impl<'_>, hir_id: hir::HirId, cx: &mut DocContext<'_>) -> Vec<Item> {
    let tcx = cx.tcx;
    let mut ret = Vec::new();
    let trait_ = impl_.of_trait.clean(cx);
    let items =
        impl_.items.iter().map(|ii| tcx.hir().impl_item(ii.id).clean(cx)).collect::<Vec<_>>();
    let def_id = tcx.hir().local_def_id(hir_id);

    // If this impl block is an implementation of the Deref trait, then we
    // need to try inlining the target's inherent impl blocks as well.
    if trait_.as_ref().map(|t| t.def_id()) == tcx.lang_items().deref_trait() {
        build_deref_target_impls(cx, &items, &mut ret);
    }

    let for_ = impl_.self_ty.clean(cx);
    let type_alias = for_.def_id().and_then(|did| match tcx.def_kind(did) {
        DefKind::TyAlias => Some(tcx.type_of(did).clean(cx)),
        _ => None,
    });
    let mut make_item = |trait_: Option<Path>, for_: Type, items: Vec<Item>| {
        let kind = ImplItem(Impl {
            span: types::rustc_span(tcx.hir().local_def_id(hir_id).to_def_id(), tcx),
            unsafety: impl_.unsafety,
            generics: impl_.generics.clean(cx),
            trait_,
            for_,
            items,
            negative_polarity: tcx.impl_polarity(def_id).clean(cx),
            synthetic: false,
            blanket_impl: None,
        });
        Item::from_hir_id_and_parts(hir_id, None, kind, cx)
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
    cx: &mut DocContext<'_>,
) -> Vec<Item> {
    // this is the ID of the `extern crate` statement
    let cnum = cx.tcx.extern_mod_stmt_cnum(krate.def_id).unwrap_or(LOCAL_CRATE);
    // this is the ID of the crate itself
    let crate_def_id = DefId { krate: cnum, index: CRATE_DEF_INDEX };
    let attrs = cx.tcx.hir().attrs(krate.hir_id());
    let please_inline = krate.vis.node.is_pub()
        && attrs.iter().any(|a| {
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
            cx.tcx.parent_module(krate.hir_id()).to_def_id(),
            Some(krate.def_id.to_def_id()),
            res,
            name,
            Some(attrs),
            &mut visited,
        ) {
            return items;
        }
    }

    // FIXME: using `from_def_id_and_kind` breaks `rustdoc/masked` for some reason
    vec![Item {
        name: Some(name),
        attrs: box attrs.clean(cx),
        def_id: crate_def_id.into(),
        visibility: krate.vis.clean(cx),
        kind: box ExternCrateItem { src: orig_name },
        cfg: attrs.cfg(cx.tcx, &cx.cache.hidden_cfg),
    }]
}

fn clean_use_statement(
    import: &hir::Item<'_>,
    name: Symbol,
    path: &hir::Path<'_>,
    kind: hir::UseKind,
    cx: &mut DocContext<'_>,
) -> Vec<Item> {
    // We need this comparison because some imports (for std types for example)
    // are "inserted" as well but directly by the compiler and they should not be
    // taken into account.
    if import.span.ctxt().outer_expn_data().kind == ExpnKind::AstPass(AstPass::StdImports) {
        return Vec::new();
    }

    let attrs = cx.tcx.hir().attrs(import.hir_id());
    let inline_attr = attrs.lists(sym::doc).get_word_attr(sym::inline);
    let pub_underscore = import.vis.node.is_pub() && name == kw::Underscore;

    if pub_underscore {
        if let Some(ref inline) = inline_attr {
            rustc_errors::struct_span_err!(
                cx.tcx.sess,
                inline.span(),
                E0780,
                "anonymous imports cannot be inlined"
            )
            .span_label(import.span, "anonymous import")
            .emit();
        }
    }

    // We consider inlining the documentation of `pub use` statements, but we
    // forcefully don't inline if this is not public or if the
    // #[doc(no_inline)] attribute is present.
    // Don't inline doc(hidden) imports so they can be stripped at a later stage.
    let mut denied = !(import.vis.node.is_pub()
        || (cx.render_options.document_private && import.vis.node.is_pub_restricted()))
        || pub_underscore
        || attrs.iter().any(|a| {
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
    let path = path.clean(cx);
    let inner = if kind == hir::UseKind::Glob {
        if !denied {
            let mut visited = FxHashSet::default();
            if let Some(items) = inline::try_inline_glob(cx, path.res, &mut visited) {
                return items;
            }
        }
        Import::new_glob(resolve_use_source(cx, path), true)
    } else {
        if inline_attr.is_none() {
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
            let import_def_id = import.def_id.to_def_id();

            if let Some(mut items) = inline::try_inline(
                cx,
                cx.tcx.parent_module(import.hir_id()).to_def_id(),
                Some(import_def_id),
                path.res,
                name,
                Some(attrs),
                &mut visited,
            ) {
                items.push(Item::from_def_id_and_parts(
                    import_def_id,
                    None,
                    ImportItem(Import::new_simple(name, resolve_use_source(cx, path), false)),
                    cx,
                ));
                return items;
            }
        }
        Import::new_simple(name, resolve_use_source(cx, path), true)
    };

    vec![Item::from_def_id_and_parts(import.def_id.to_def_id(), None, ImportItem(inner), cx)]
}

impl Clean<Item> for (&hir::ForeignItem<'_>, Option<Symbol>) {
    fn clean(&self, cx: &mut DocContext<'_>) -> Item {
        let (item, renamed) = self;
        cx.with_param_env(item.def_id.to_def_id(), |cx| {
            let kind = match item.kind {
                hir::ForeignItemKind::Fn(ref decl, ref names, ref generics) => {
                    let abi = cx.tcx.hir().get_foreign_abi(item.hir_id());
                    let (generics, decl) = enter_impl_trait(cx, |cx| {
                        (generics.clean(cx), (&**decl, &names[..]).clean(cx))
                    });
                    ForeignFunctionItem(Function {
                        decl,
                        generics,
                        header: hir::FnHeader {
                            unsafety: if abi == Abi::RustIntrinsic {
                                intrinsic_operation_unsafety(item.ident.name)
                            } else {
                                hir::Unsafety::Unsafe
                            },
                            abi,
                            constness: hir::Constness::NotConst,
                            asyncness: hir::IsAsync::NotAsync,
                        },
                    })
                }
                hir::ForeignItemKind::Static(ref ty, mutability) => {
                    ForeignStaticItem(Static { type_: ty.clean(cx), mutability, expr: None })
                }
                hir::ForeignItemKind::Type => ForeignTypeItem,
            };

            Item::from_hir_id_and_parts(
                item.hir_id(),
                Some(renamed.unwrap_or(item.ident.name)),
                kind,
                cx,
            )
        })
    }
}

impl Clean<TypeBinding> for hir::TypeBinding<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> TypeBinding {
        TypeBinding { name: self.ident.name, kind: self.kind.clean(cx) }
    }
}

impl Clean<TypeBindingKind> for hir::TypeBindingKind<'_> {
    fn clean(&self, cx: &mut DocContext<'_>) -> TypeBindingKind {
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
