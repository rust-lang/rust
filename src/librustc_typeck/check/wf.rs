// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use astconv::AstConv;
use check::{FnCtxt, Inherited, blank_fn_ctxt, regionck, wfcheck};
use constrained_type_params::{identify_constrained_type_params, Parameter};
use CrateCtxt;
use middle::region;
use middle::subst::{self, TypeSpace, FnSpace, ParamSpace, SelfSpace};
use middle::traits;
use middle::ty::{self, Ty};
use middle::ty::fold::{TypeFolder, TypeFoldable, super_fold_ty};

use std::cell::RefCell;
use std::collections::HashSet;
use syntax::ast;
use syntax::codemap::{DUMMY_SP, Span};
use syntax::parse::token::special_idents;

use rustc_front::intravisit::{self, Visitor, FnKind};
use rustc_front::hir;

pub struct CheckTypeWellFormedVisitor<'ccx, 'tcx:'ccx> {
    ccx: &'ccx CrateCtxt<'ccx, 'tcx>,
    cache: HashSet<Ty<'tcx>>
}

impl<'ccx, 'tcx> CheckTypeWellFormedVisitor<'ccx, 'tcx> {
    pub fn new(ccx: &'ccx CrateCtxt<'ccx, 'tcx>) -> CheckTypeWellFormedVisitor<'ccx, 'tcx> {
        CheckTypeWellFormedVisitor { ccx: ccx, cache: HashSet::new() }
    }

    fn tcx(&self) -> &ty::ctxt<'tcx> {
        self.ccx.tcx
    }

    /// Checks that the field types (in a struct def'n) or argument types (in an enum def'n) are
    /// well-formed, meaning that they do not require any constraints not declared in the struct
    /// definition itself. For example, this definition would be illegal:
    ///
    ///     struct Ref<'a, T> { x: &'a T }
    ///
    /// because the type did not declare that `T:'a`.
    ///
    /// We do this check as a pre-pass before checking fn bodies because if these constraints are
    /// not included it frequently leads to confusing errors in fn bodies. So it's better to check
    /// the types first.
    fn check_item_well_formed(&mut self, item: &hir::Item) {
        let ccx = self.ccx;
        debug!("check_item_well_formed(it.id={}, it.name={})",
               item.id,
               ccx.tcx.item_path_str(ccx.tcx.map.local_def_id(item.id)));

        match item.node {
            /// Right now we check that every default trait implementation
            /// has an implementation of itself. Basically, a case like:
            ///
            /// `impl Trait for T {}`
            ///
            /// has a requirement of `T: Trait` which was required for default
            /// method implementations. Although this could be improved now that
            /// there's a better infrastructure in place for this, it's being left
            /// for a follow-up work.
            ///
            /// Since there's such a requirement, we need to check *just* positive
            /// implementations, otherwise things like:
            ///
            /// impl !Send for T {}
            ///
            /// won't be allowed unless there's an *explicit* implementation of `Send`
            /// for `T`
            hir::ItemImpl(_, hir::ImplPolarity::Positive, _, _, _, _) => {
                self.check_impl(item);
            }
            hir::ItemImpl(_, hir::ImplPolarity::Negative, _, Some(_), _, _) => {
                let item_def_id = ccx.tcx.map.local_def_id(item.id);
                let trait_ref = ccx.tcx.impl_trait_ref(item_def_id).unwrap();
                ccx.tcx.populate_implementations_for_trait_if_necessary(trait_ref.def_id);
                match ccx.tcx.lang_items.to_builtin_kind(trait_ref.def_id) {
                    Some(ty::BoundSend) | Some(ty::BoundSync) => {}
                    Some(_) | None => {
                        if !ccx.tcx.trait_has_default_impl(trait_ref.def_id) {
                            wfcheck::error_192(ccx, item.span);
                        }
                    }
                }
            }
            hir::ItemFn(..) => {
                self.check_item_type(item);
            }
            hir::ItemStatic(..) => {
                self.check_item_type(item);
            }
            hir::ItemConst(..) => {
                self.check_item_type(item);
            }
            hir::ItemStruct(ref struct_def, ref ast_generics) => {
                self.check_type_defn(item, |fcx| {
                    vec![struct_variant(fcx, struct_def)]
                });

                self.check_variances_for_type_defn(item, ast_generics);
            }
            hir::ItemEnum(ref enum_def, ref ast_generics) => {
                self.check_type_defn(item, |fcx| {
                    enum_variants(fcx, enum_def)
                });

                self.check_variances_for_type_defn(item, ast_generics);
            }
            hir::ItemTrait(_, _, _, ref items) => {
                let trait_predicates =
                    ccx.tcx.lookup_predicates(ccx.tcx.map.local_def_id(item.id));
                reject_non_type_param_bounds(ccx.tcx, item.span, &trait_predicates);
                if ccx.tcx.trait_has_default_impl(ccx.tcx.map.local_def_id(item.id)) {
                    if !items.is_empty() {
                        wfcheck::error_380(ccx, item.span);
                    }
                }
            }
            _ => {}
        }
    }

    fn with_fcx<F>(&mut self, item: &hir::Item, mut f: F) where
        F: for<'fcx> FnMut(&mut CheckTypeWellFormedVisitor<'ccx, 'tcx>, &FnCtxt<'fcx, 'tcx>),
    {
        let ccx = self.ccx;
        let item_def_id = ccx.tcx.map.local_def_id(item.id);
        let type_scheme = ccx.tcx.lookup_item_type(item_def_id);
        let type_predicates = ccx.tcx.lookup_predicates(item_def_id);
        reject_non_type_param_bounds(ccx.tcx, item.span, &type_predicates);
        let param_env = ccx.tcx.construct_parameter_environment(item.span,
                                                                &type_scheme.generics,
                                                                &type_predicates,
                                                                item.id);
        let tables = RefCell::new(ty::Tables::empty());
        let inh = Inherited::new(ccx.tcx, &tables, param_env);
        let fcx = blank_fn_ctxt(ccx, &inh, ty::FnConverging(type_scheme.ty), item.id);
        f(self, &fcx);
        fcx.select_all_obligations_or_error();
        regionck::regionck_item(&fcx, item.id, item.span, &[]);
    }

    /// In a type definition, we check that to ensure that the types of the fields are well-formed.
    fn check_type_defn<F>(&mut self, item: &hir::Item, mut lookup_fields: F) where
        F: for<'fcx> FnMut(&FnCtxt<'fcx, 'tcx>) -> Vec<AdtVariant<'tcx>>,
    {
        self.with_fcx(item, |this, fcx| {
            let variants = lookup_fields(fcx);
            let mut bounds_checker = BoundsChecker::new(fcx,
                                                        item.id,
                                                        Some(&mut this.cache));
            debug!("check_type_defn at bounds_checker.scope: {:?}", bounds_checker.scope);

            for variant in &variants {
                for field in &variant.fields {
                    // Regions are checked below.
                    bounds_checker.check_traits_in_ty(field.ty, field.span);
                }

                // For DST, all intermediate types must be sized.
                if let Some((_, fields)) = variant.fields.split_last() {
                    for field in fields {
                        fcx.register_builtin_bound(
                            field.ty,
                            ty::BoundSized,
                            traits::ObligationCause::new(field.span,
                                                         fcx.body_id,
                                                         traits::FieldSized));
                    }
                }
            }

            for field in variants.iter().flat_map(|v| v.fields.iter()) {
                fcx.register_old_wf_obligation(field.ty, field.span, traits::MiscObligation);
            }
        });
    }

    fn check_item_type(&mut self,
                       item: &hir::Item)
    {
        self.with_fcx(item, |this, fcx| {
            let mut bounds_checker = BoundsChecker::new(fcx,
                                                        item.id,
                                                        Some(&mut this.cache));
            debug!("check_item_type at bounds_checker.scope: {:?}", bounds_checker.scope);

            let item_def_id = fcx.tcx().map.local_def_id(item.id);
            let type_scheme = fcx.tcx().lookup_item_type(item_def_id);
            let item_ty = fcx.instantiate_type_scheme(item.span,
                                                      &fcx.inh
                                                          .infcx
                                                          .parameter_environment
                                                          .free_substs,
                                                      &type_scheme.ty);

            bounds_checker.check_traits_in_ty(item_ty, item.span);
        });
    }

    fn check_impl(&mut self,
                  item: &hir::Item)
    {
        self.with_fcx(item, |this, fcx| {
            let mut bounds_checker = BoundsChecker::new(fcx,
                                                        item.id,
                                                        Some(&mut this.cache));
            debug!("check_impl at bounds_checker.scope: {:?}", bounds_checker.scope);

            // Find the impl self type as seen from the "inside" --
            // that is, with all type parameters converted from bound
            // to free.
            let self_ty = fcx.tcx().node_id_to_type(item.id);
            let self_ty = fcx.instantiate_type_scheme(item.span,
                                                      &fcx.inh
                                                          .infcx
                                                          .parameter_environment
                                                          .free_substs,
                                                      &self_ty);

            bounds_checker.check_traits_in_ty(self_ty, item.span);

            // Similarly, obtain an "inside" reference to the trait
            // that the impl implements.
            let trait_ref = match fcx.tcx().impl_trait_ref(fcx.tcx().map.local_def_id(item.id)) {
                None => { return; }
                Some(t) => { t }
            };

            let trait_ref = fcx.instantiate_type_scheme(item.span,
                                                        &fcx.inh
                                                            .infcx
                                                            .parameter_environment
                                                            .free_substs,
                                                        &trait_ref);

            // We are stricter on the trait-ref in an impl than the
            // self-type.  In particular, we enforce region
            // relationships. The reason for this is that (at least
            // presently) "applying" an impl does not require that the
            // application site check the well-formedness constraints on the
            // trait reference. Instead, this is done at the impl site.
            // Arguably this is wrong and we should treat the trait-reference
            // the same way as we treat the self-type.
            bounds_checker.check_trait_ref(&trait_ref, item.span);

            let cause =
                traits::ObligationCause::new(
                    item.span,
                    fcx.body_id,
                    traits::ItemObligation(trait_ref.def_id));

            // Find the supertrait bounds. This will add `int:Bar`.
            let poly_trait_ref = ty::Binder(trait_ref);
            let predicates = fcx.tcx().lookup_super_predicates(poly_trait_ref.def_id());
            let predicates = predicates.instantiate_supertrait(fcx.tcx(), &poly_trait_ref);
            let predicates = {
                let selcx = &mut traits::SelectionContext::new(fcx.infcx());
                traits::normalize(selcx, cause.clone(), &predicates)
            };
            for predicate in predicates.value.predicates {
                fcx.register_predicate(traits::Obligation::new(cause.clone(), predicate));
            }
            for obligation in predicates.obligations {
                fcx.register_predicate(obligation);
            }
        });
    }

    fn check_variances_for_type_defn(&self,
                                     item: &hir::Item,
                                     ast_generics: &hir::Generics)
    {
        let item_def_id = self.tcx().map.local_def_id(item.id);
        let ty_predicates = self.tcx().lookup_predicates(item_def_id);
        let variances = self.tcx().item_variances(item_def_id);

        let mut constrained_parameters: HashSet<_> =
            variances.types
                     .iter_enumerated()
                     .filter(|&(_, _, &variance)| variance != ty::Bivariant)
                     .map(|(space, index, _)| self.param_ty(ast_generics, space, index))
                     .map(|p| Parameter::Type(p))
                     .collect();

        identify_constrained_type_params(self.tcx(),
                                         ty_predicates.predicates.as_slice(),
                                         None,
                                         &mut constrained_parameters);

        for (space, index, _) in variances.types.iter_enumerated() {
            let param_ty = self.param_ty(ast_generics, space, index);
            if constrained_parameters.contains(&Parameter::Type(param_ty)) {
                continue;
            }
            let span = self.ty_param_span(ast_generics, item, space, index);
            self.report_bivariance(span, param_ty.name);
        }

        for (space, index, &variance) in variances.regions.iter_enumerated() {
            if variance != ty::Bivariant {
                continue;
            }

            assert_eq!(space, TypeSpace);
            let span = ast_generics.lifetimes[index].lifetime.span;
            let name = ast_generics.lifetimes[index].lifetime.name;
            self.report_bivariance(span, name);
        }
    }

    fn param_ty(&self,
                ast_generics: &hir::Generics,
                space: ParamSpace,
                index: usize)
                -> ty::ParamTy
    {
        let name = match space {
            TypeSpace => ast_generics.ty_params[index].name,
            SelfSpace => special_idents::type_self.name,
            FnSpace => self.tcx().sess.bug("Fn space occupied?"),
        };

        ty::ParamTy { space: space, idx: index as u32, name: name }
    }

    fn ty_param_span(&self,
                     ast_generics: &hir::Generics,
                     item: &hir::Item,
                     space: ParamSpace,
                     index: usize)
                     -> Span
    {
        match space {
            TypeSpace => ast_generics.ty_params[index].span,
            SelfSpace => item.span,
            FnSpace => self.tcx().sess.span_bug(item.span, "Fn space occupied?"),
        }
    }

    fn report_bivariance(&self,
                         span: Span,
                         param_name: ast::Name)
    {
        wfcheck::error_392(self.tcx(), span, param_name);

        let suggested_marker_id = self.tcx().lang_items.phantom_data();
        match suggested_marker_id {
            Some(def_id) => {
                self.tcx().sess.fileline_help(
                    span,
                    &format!("consider removing `{}` or using a marker such as `{}`",
                             param_name,
                             self.tcx().item_path_str(def_id)));
            }
            None => {
                // no lang items, no help!
            }
        }
    }
}

// Reject any predicates that do not involve a type parameter.
fn reject_non_type_param_bounds<'tcx>(tcx: &ty::ctxt<'tcx>,
                                      span: Span,
                                      predicates: &ty::GenericPredicates<'tcx>) {
    for predicate in &predicates.predicates {
        match predicate {
            &ty::Predicate::Trait(ty::Binder(ref tr)) => {
                let found_param = tr.input_types().iter()
                                    .flat_map(|ty| ty.walk())
                                    .any(is_ty_param);
                if !found_param { report_bound_error(tcx, span, tr.self_ty() )}
            }
            &ty::Predicate::TypeOutlives(ty::Binder(ty::OutlivesPredicate(ty, _))) => {
                let found_param = ty.walk().any(|t| is_ty_param(t));
                if !found_param { report_bound_error(tcx, span, ty) }
            }
            _ => {}
        };
    }

    fn report_bound_error<'t>(tcx: &ty::ctxt<'t>,
                          span: Span,
                          bounded_ty: ty::Ty<'t>) {
        span_err!(tcx.sess, span, E0193,
            "cannot bound type `{}`, where clause \
                bounds may only be attached to types involving \
                type parameters",
                bounded_ty)
    }

    fn is_ty_param(ty: ty::Ty) -> bool {
        match &ty.sty {
            &ty::TyParam(_) => true,
            _ => false
        }
    }
}

fn reject_shadowing_type_parameters<'tcx>(tcx: &ty::ctxt<'tcx>,
                                          span: Span,
                                          generics: &ty::Generics<'tcx>) {
    let impl_params = generics.types.get_slice(subst::TypeSpace).iter()
        .map(|tp| tp.name).collect::<HashSet<_>>();

    for method_param in generics.types.get_slice(subst::FnSpace) {
        if impl_params.contains(&method_param.name) {
            wfcheck::error_194(tcx, span, method_param.name);
        }
    }
}

impl<'ccx, 'tcx, 'v> Visitor<'v> for CheckTypeWellFormedVisitor<'ccx, 'tcx> {
    fn visit_item(&mut self, i: &hir::Item) {
        self.check_item_well_formed(i);
        intravisit::walk_item(self, i);
    }

    fn visit_fn(&mut self,
                fk: FnKind<'v>, fd: &'v hir::FnDecl,
                b: &'v hir::Block, span: Span, id: ast::NodeId) {
        match fk {
            FnKind::Closure | FnKind::ItemFn(..) => {}
            FnKind::Method(..) => {
                match self.tcx().impl_or_trait_item(self.tcx().map.local_def_id(id)) {
                    ty::ImplOrTraitItem::MethodTraitItem(ty_method) => {
                        reject_shadowing_type_parameters(self.tcx(), span, &ty_method.generics)
                    }
                    _ => {}
                }
            }
        }
        intravisit::walk_fn(self, fk, fd, b, span)
    }

    fn visit_trait_item(&mut self, trait_item: &'v hir::TraitItem) {
        if let hir::MethodTraitItem(_, None) = trait_item.node {
            match self.tcx().impl_or_trait_item(self.tcx().map.local_def_id(trait_item.id)) {
                ty::ImplOrTraitItem::MethodTraitItem(ty_method) => {
                    reject_non_type_param_bounds(
                        self.tcx(),
                        trait_item.span,
                        &ty_method.predicates);
                    reject_shadowing_type_parameters(
                        self.tcx(),
                        trait_item.span,
                        &ty_method.generics);
                }
                _ => {}
            }
        }

        intravisit::walk_trait_item(self, trait_item)
    }
}

pub struct BoundsChecker<'cx,'tcx:'cx> {
    fcx: &'cx FnCtxt<'cx,'tcx>,
    span: Span,

    scope: region::CodeExtent,

    binding_count: usize,
    cache: Option<&'cx mut HashSet<Ty<'tcx>>>,
}

impl<'cx,'tcx> BoundsChecker<'cx,'tcx> {
    pub fn new(fcx: &'cx FnCtxt<'cx,'tcx>,
               scope: ast::NodeId,
               cache: Option<&'cx mut HashSet<Ty<'tcx>>>)
               -> BoundsChecker<'cx,'tcx> {
        let scope = fcx.tcx().region_maps.item_extent(scope);
        BoundsChecker { fcx: fcx, span: DUMMY_SP, scope: scope,
                        cache: cache, binding_count: 0 }
    }

    /// Given a trait ref like `A : Trait<B>`, where `Trait` is defined as (say):
    ///
    ///     trait Trait<B:OtherTrait> : Copy { ... }
    ///
    /// This routine will check that `B : OtherTrait` and `A : Trait<B>`. It will also recursively
    /// check that the types `A` and `B` are well-formed.
    ///
    /// Note that it does not (currently, at least) check that `A : Copy` (that check is delegated
    /// to the point where impl `A : Trait<B>` is implemented).
    pub fn check_trait_ref(&mut self, trait_ref: &ty::TraitRef<'tcx>, span: Span) {
        let trait_predicates = self.fcx.tcx().lookup_predicates(trait_ref.def_id);

        let bounds = self.fcx.instantiate_bounds(span,
                                                 trait_ref.substs,
                                                 &trait_predicates);

        self.fcx.add_obligations_for_parameters(
            traits::ObligationCause::new(
                span,
                self.fcx.body_id,
                traits::ItemObligation(trait_ref.def_id)),
            &bounds);

        for &ty in &trait_ref.substs.types {
            self.check_traits_in_ty(ty, span);
        }
    }

    fn check_traits_in_ty(&mut self, ty: Ty<'tcx>, span: Span) {
        self.span = span;
        // When checking types outside of a type def'n, we ignore
        // region obligations. See discussion below in fold_ty().
        self.binding_count += 1;
        ty.fold_with(self);
        self.binding_count -= 1;
    }
}

impl<'cx,'tcx> TypeFolder<'tcx> for BoundsChecker<'cx,'tcx> {
    fn tcx(&self) -> &ty::ctxt<'tcx> {
        self.fcx.tcx()
    }

    fn fold_binder<T>(&mut self, binder: &ty::Binder<T>) -> ty::Binder<T>
        where T : TypeFoldable<'tcx>
    {
        self.binding_count += 1;
        let value = self.fcx.tcx().liberate_late_bound_regions(
            self.scope,
            binder);
        debug!("BoundsChecker::fold_binder: late-bound regions replaced: {:?} at scope: {:?}",
               value, self.scope);
        let value = value.fold_with(self);
        self.binding_count -= 1;
        ty::Binder(value)
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        debug!("BoundsChecker t={:?}",
               t);

        match self.cache {
            Some(ref mut cache) => {
                if !cache.insert(t) {
                    // Already checked this type! Don't check again.
                    debug!("cached");
                    return t;
                }
            }
            None => { }
        }

        match t.sty{
            ty::TyStruct(def, substs) |
            ty::TyEnum(def, substs) => {
                let type_predicates = def.predicates(self.fcx.tcx());
                let bounds = self.fcx.instantiate_bounds(self.span, substs,
                                                         &type_predicates);

                if self.binding_count == 0 {
                    self.fcx.add_obligations_for_parameters(
                        traits::ObligationCause::new(self.span,
                                                     self.fcx.body_id,
                                                     traits::ItemObligation(def.did)),
                        &bounds);
                } else {
                    // There are two circumstances in which we ignore
                    // region obligations.
                    //
                    // The first is when we are inside of a closure
                    // type. This is because in that case the region
                    // obligations for the parameter types are things
                    // that the closure body gets to assume and the
                    // caller must prove at the time of call. In other
                    // words, if there is a type like `<'a, 'b> | &'a
                    // &'b int |`, it is well-formed, and caller will
                    // have to show that `'b : 'a` at the time of
                    // call.
                    //
                    // The second is when we are checking for
                    // well-formedness outside of a type def'n or fn
                    // body. This is for a similar reason: in general,
                    // we only do WF checking for regions in the
                    // result of expressions and type definitions, so
                    // to as allow for implicit where clauses.
                    //
                    // (I believe we should do the same for traits, but
                    // that will require an RFC. -nmatsakis)
                    let bounds = filter_to_trait_obligations(bounds);
                    self.fcx.add_obligations_for_parameters(
                        traits::ObligationCause::new(self.span,
                                                     self.fcx.body_id,
                                                     traits::ItemObligation(def.did)),
                        &bounds);
                }

                self.fold_substs(substs);
            }
            _ => {
                super_fold_ty(self, t);
            }
        }

        t // we're not folding to produce a new type, so just return `t` here
    }
}

///////////////////////////////////////////////////////////////////////////
// ADT

struct AdtVariant<'tcx> {
    fields: Vec<AdtField<'tcx>>,
}

struct AdtField<'tcx> {
    ty: Ty<'tcx>,
    span: Span,
}

fn struct_variant<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                            struct_def: &hir::VariantData)
                            -> AdtVariant<'tcx> {
    let fields =
        struct_def.fields().iter()
        .map(|field| {
            let field_ty = fcx.tcx().node_id_to_type(field.node.id);
            let field_ty = fcx.instantiate_type_scheme(field.span,
                                                       &fcx.inh
                                                           .infcx
                                                           .parameter_environment
                                                           .free_substs,
                                                       &field_ty);
            AdtField { ty: field_ty, span: field.span }
        })
        .collect();
    AdtVariant { fields: fields }
}

fn enum_variants<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                           enum_def: &hir::EnumDef)
                           -> Vec<AdtVariant<'tcx>> {
    enum_def.variants.iter()
        .map(|variant| struct_variant(fcx, &variant.node.data))
        .collect()
}

fn filter_to_trait_obligations<'tcx>(bounds: ty::InstantiatedPredicates<'tcx>)
                                     -> ty::InstantiatedPredicates<'tcx>
{
    let mut result = ty::InstantiatedPredicates::empty();
    for (space, _, predicate) in bounds.predicates.iter_enumerated() {
        match *predicate {
            ty::Predicate::Trait(..) |
            ty::Predicate::Projection(..) => {
                result.predicates.push(space, predicate.clone())
            }
            ty::Predicate::WellFormed(..) |
            ty::Predicate::ObjectSafe(..) |
            ty::Predicate::Equate(..) |
            ty::Predicate::TypeOutlives(..) |
            ty::Predicate::RegionOutlives(..) => {
            }
        }
    }
    result
}
