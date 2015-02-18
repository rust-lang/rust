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
use check::{FnCtxt, Inherited, blank_fn_ctxt, vtable, regionck};
use constrained_type_params::identify_constrained_type_params;
use CrateCtxt;
use middle::region;
use middle::subst::{self, TypeSpace, FnSpace, ParamSpace, SelfSpace};
use middle::traits;
use middle::ty::{self, Ty};
use middle::ty::liberate_late_bound_regions;
use middle::ty_fold::{TypeFolder, TypeFoldable, super_fold_ty};
use util::ppaux::{Repr, UserString};

use std::collections::HashSet;
use syntax::ast;
use syntax::ast_util::{local_def};
use syntax::attr;
use syntax::codemap::Span;
use syntax::parse::token::{self, special_idents};
use syntax::visit;
use syntax::visit::Visitor;

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
    fn check_item_well_formed(&mut self, item: &ast::Item) {
        let ccx = self.ccx;
        debug!("check_item_well_formed(it.id={}, it.ident={})",
               item.id,
               ty::item_path_str(ccx.tcx, local_def(item.id)));

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
            ast::ItemImpl(_, ast::ImplPolarity::Positive, _, _, _, _) => {
                self.check_impl(item);
            }
            ast::ItemImpl(_, ast::ImplPolarity::Negative, _, Some(ref tref), _, _) => {
                let trait_ref = ty::node_id_to_trait_ref(ccx.tcx, tref.ref_id);
                match ccx.tcx.lang_items.to_builtin_kind(trait_ref.def_id) {
                    Some(ty::BoundSend) | Some(ty::BoundSync) => {}
                    Some(_) | None => {
                        span_err!(ccx.tcx.sess, item.span, E0192,
                            "negative impls are currently \
                                     allowed just for `Send` and `Sync`")
                    }
                }
            }
            ast::ItemFn(..) => {
                self.check_item_type(item);
            }
            ast::ItemStatic(..) => {
                self.check_item_type(item);
            }
            ast::ItemConst(..) => {
                self.check_item_type(item);
            }
            ast::ItemStruct(ref struct_def, ref ast_generics) => {
                self.check_type_defn(item, |fcx| {
                    vec![struct_variant(fcx, &**struct_def)]
                });

                self.check_variances_for_type_defn(item, ast_generics);
            }
            ast::ItemEnum(ref enum_def, ref ast_generics) => {
                self.check_type_defn(item, |fcx| {
                    enum_variants(fcx, enum_def)
                });

                self.check_variances_for_type_defn(item, ast_generics);
            }
            ast::ItemTrait(_, ref ast_generics, _, _) => {
                let trait_predicates =
                    ty::lookup_predicates(ccx.tcx, local_def(item.id));
                reject_non_type_param_bounds(
                    ccx.tcx,
                    item.span,
                    &trait_predicates);
                self.check_variances(item, ast_generics, &trait_predicates,
                                     self.tcx().lang_items.phantom_fn());
            }
            _ => {}
        }
    }

    fn with_fcx<F>(&mut self, item: &ast::Item, mut f: F) where
        F: for<'fcx> FnMut(&mut CheckTypeWellFormedVisitor<'ccx, 'tcx>, &FnCtxt<'fcx, 'tcx>),
    {
        let ccx = self.ccx;
        let item_def_id = local_def(item.id);
        let type_scheme = ty::lookup_item_type(ccx.tcx, item_def_id);
        let type_predicates = ty::lookup_predicates(ccx.tcx, item_def_id);
        reject_non_type_param_bounds(ccx.tcx, item.span, &type_predicates);
        let param_env =
            ty::construct_parameter_environment(ccx.tcx,
                                                item.span,
                                                &type_scheme.generics,
                                                &type_predicates,
                                                item.id);
        let inh = Inherited::new(ccx.tcx, param_env);
        let fcx = blank_fn_ctxt(ccx, &inh, ty::FnConverging(type_scheme.ty), item.id);
        f(self, &fcx);
        vtable::select_all_fcx_obligations_or_error(&fcx);
        regionck::regionck_item(&fcx, item);
    }

    /// In a type definition, we check that to ensure that the types of the fields are well-formed.
    fn check_type_defn<F>(&mut self, item: &ast::Item, mut lookup_fields: F) where
        F: for<'fcx> FnMut(&FnCtxt<'fcx, 'tcx>) -> Vec<AdtVariant<'tcx>>,
    {
        self.with_fcx(item, |this, fcx| {
            let variants = lookup_fields(fcx);
            let mut bounds_checker = BoundsChecker::new(fcx,
                                                        item.span,
                                                        item.id,
                                                        Some(&mut this.cache));
            debug!("check_type_defn at bounds_checker.scope: {:?}", bounds_checker.scope);

             for variant in &variants {
                for field in &variant.fields {
                    // Regions are checked below.
                    bounds_checker.check_traits_in_ty(field.ty);
                }

                // For DST, all intermediate types must be sized.
                if variant.fields.len() > 0 {
                    for field in variant.fields.init() {
                        fcx.register_builtin_bound(
                            field.ty,
                            ty::BoundSized,
                            traits::ObligationCause::new(field.span,
                                                         fcx.body_id,
                                                         traits::FieldSized));
                    }
                }
            }

            let field_tys: Vec<Ty> =
                variants.iter().flat_map(|v| v.fields.iter().map(|f| f.ty)).collect();

            regionck::regionck_ensure_component_tys_wf(
                fcx, item.span, &field_tys);
        });
    }

    fn check_item_type(&mut self,
                       item: &ast::Item)
    {
        self.with_fcx(item, |this, fcx| {
            let mut bounds_checker = BoundsChecker::new(fcx,
                                                        item.span,
                                                        item.id,
                                                        Some(&mut this.cache));
            debug!("check_item_type at bounds_checker.scope: {:?}", bounds_checker.scope);

            let type_scheme = ty::lookup_item_type(fcx.tcx(), local_def(item.id));
            let item_ty = fcx.instantiate_type_scheme(item.span,
                                                      &fcx.inh.param_env.free_substs,
                                                      &type_scheme.ty);

            bounds_checker.check_traits_in_ty(item_ty);
        });
    }

    fn check_impl(&mut self,
                  item: &ast::Item)
    {
        self.with_fcx(item, |this, fcx| {
            let mut bounds_checker = BoundsChecker::new(fcx,
                                                        item.span,
                                                        item.id,
                                                        Some(&mut this.cache));
            debug!("check_impl at bounds_checker.scope: {:?}", bounds_checker.scope);

            // Find the impl self type as seen from the "inside" --
            // that is, with all type parameters converted from bound
            // to free.
            let self_ty = ty::node_id_to_type(fcx.tcx(), item.id);
            let self_ty = fcx.instantiate_type_scheme(item.span,
                                                      &fcx.inh.param_env.free_substs,
                                                      &self_ty);

            bounds_checker.check_traits_in_ty(self_ty);

            // Similarly, obtain an "inside" reference to the trait
            // that the impl implements.
            let trait_ref = match ty::impl_trait_ref(fcx.tcx(), local_def(item.id)) {
                None => { return; }
                Some(t) => { t }
            };

            let trait_ref = fcx.instantiate_type_scheme(item.span,
                                                        &fcx.inh.param_env.free_substs,
                                                        &trait_ref);

            // There are special rules that apply to drop.
            if
                fcx.tcx().lang_items.drop_trait() == Some(trait_ref.def_id) &&
                !attr::contains_name(&item.attrs, "unsafe_destructor")
            {
                match self_ty.sty {
                    ty::ty_struct(def_id, _) |
                    ty::ty_enum(def_id, _) => {
                        check_struct_safe_for_destructor(fcx, item.span, def_id);
                    }
                    _ => {
                        // Coherence already reports an error in this case.
                    }
                }
            }

            if fcx.tcx().lang_items.copy_trait() == Some(trait_ref.def_id) {
                // This is checked in coherence.
                return
            }

            // We are stricter on the trait-ref in an impl than the
            // self-type.  In particular, we enforce region
            // relationships. The reason for this is that (at least
            // presently) "applying" an impl does not require that the
            // application site check the well-formedness constraints on the
            // trait reference. Instead, this is done at the impl site.
            // Arguably this is wrong and we should treat the trait-reference
            // the same way as we treat the self-type.
            bounds_checker.check_trait_ref(&*trait_ref);

            let cause =
                traits::ObligationCause::new(
                    item.span,
                    fcx.body_id,
                    traits::ItemObligation(trait_ref.def_id));

            // Find the supertrait bounds. This will add `int:Bar`.
            let poly_trait_ref = ty::Binder(trait_ref);
            let predicates = ty::predicates_for_trait_ref(fcx.tcx(), &poly_trait_ref);
            let predicates = {
                let selcx = &mut traits::SelectionContext::new(fcx.infcx(), fcx);
                traits::normalize(selcx, cause.clone(), &predicates)
            };
            for predicate in predicates.value {
                fcx.register_predicate(traits::Obligation::new(cause.clone(), predicate));
            }
            for obligation in predicates.obligations {
                fcx.register_predicate(obligation);
            }
        });
    }

    fn check_variances_for_type_defn(&self,
                                     item: &ast::Item,
                                     ast_generics: &ast::Generics)
    {
        let item_def_id = local_def(item.id);
        let predicates = ty::lookup_predicates(self.tcx(), item_def_id);
        self.check_variances(item,
                             ast_generics,
                             &predicates,
                             self.tcx().lang_items.phantom_data());
    }

    fn check_variances(&self,
                       item: &ast::Item,
                       ast_generics: &ast::Generics,
                       ty_predicates: &ty::GenericPredicates<'tcx>,
                       suggested_marker_id: Option<ast::DefId>)
    {
        let variance_lang_items = &[
            self.tcx().lang_items.phantom_fn(),
            self.tcx().lang_items.phantom_data(),
        ];

        let item_def_id = local_def(item.id);
        let is_lang_item = variance_lang_items.iter().any(|n| *n == Some(item_def_id));
        if is_lang_item {
            return;
        }

        let variances = ty::item_variances(self.tcx(), item_def_id);

        let mut constrained_parameters: HashSet<_> =
            variances.types
            .iter_enumerated()
            .filter(|&(_, _, &variance)| variance != ty::Bivariant)
            .map(|(space, index, _)| self.param_ty(ast_generics, space, index))
            .collect();

        identify_constrained_type_params(self.tcx(),
                                         ty_predicates.predicates.as_slice(),
                                         None,
                                         &mut constrained_parameters);

        for (space, index, _) in variances.types.iter_enumerated() {
            let param_ty = self.param_ty(ast_generics, space, index);
            if constrained_parameters.contains(&param_ty) {
                continue;
            }
            let span = self.ty_param_span(ast_generics, item, space, index);
            self.report_bivariance(span, param_ty.name, suggested_marker_id);
        }

        for (space, index, &variance) in variances.regions.iter_enumerated() {
            if variance != ty::Bivariant {
                continue;
            }

            assert_eq!(space, TypeSpace);
            let span = ast_generics.lifetimes[index].lifetime.span;
            let name = ast_generics.lifetimes[index].lifetime.name;
            self.report_bivariance(span, name, suggested_marker_id);
        }
    }

    fn param_ty(&self,
                ast_generics: &ast::Generics,
                space: ParamSpace,
                index: usize)
                -> ty::ParamTy
    {
        let name = match space {
            TypeSpace => ast_generics.ty_params[index].ident.name,
            SelfSpace => special_idents::type_self.name,
            FnSpace => self.tcx().sess.bug("Fn space occupied?"),
        };

        ty::ParamTy { space: space, idx: index as u32, name: name }
    }

    fn ty_param_span(&self,
                     ast_generics: &ast::Generics,
                     item: &ast::Item,
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
                         param_name: ast::Name,
                         suggested_marker_id: Option<ast::DefId>)
    {
        self.tcx().sess.span_err(
            span,
            &format!("parameter `{}` is never used",
                     param_name.user_string(self.tcx()))[]);

        match suggested_marker_id {
            Some(def_id) => {
                self.tcx().sess.span_help(
                    span,
                    format!("consider removing `{}` or using a marker such as `{}`",
                            param_name.user_string(self.tcx()),
                            ty::item_path_str(self.tcx(), def_id)).as_slice());
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
    for predicate in predicates.predicates.iter() {
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
                bounded_ty.repr(tcx))
    }

    fn is_ty_param(ty: ty::Ty) -> bool {
        match &ty.sty {
            &ty::sty::ty_param(_) => true,
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
            span_err!(tcx.sess, span, E0194,
                "type parameter `{}` shadows another type parameter of the same name",
                          token::get_name(method_param.name));
        }
    }
}

impl<'ccx, 'tcx, 'v> Visitor<'v> for CheckTypeWellFormedVisitor<'ccx, 'tcx> {
    fn visit_item(&mut self, i: &ast::Item) {
        self.check_item_well_formed(i);
        visit::walk_item(self, i);
    }

    fn visit_fn(&mut self,
                fk: visit::FnKind<'v>, fd: &'v ast::FnDecl,
                b: &'v ast::Block, span: Span, id: ast::NodeId) {
        match fk {
            visit::FkFnBlock | visit::FkItemFn(..) => {}
            visit::FkMethod(..) => {
                match ty::impl_or_trait_item(self.tcx(), local_def(id)) {
                    ty::ImplOrTraitItem::MethodTraitItem(ty_method) => {
                        reject_shadowing_type_parameters(self.tcx(), span, &ty_method.generics)
                    }
                    _ => {}
                }
            }
        }
        visit::walk_fn(self, fk, fd, b, span)
    }

    fn visit_trait_item(&mut self, t: &'v ast::TraitItem) {
        match t {
            &ast::TraitItem::ProvidedMethod(_) |
            &ast::TraitItem::TypeTraitItem(_) => {},
            &ast::TraitItem::RequiredMethod(ref method) => {
                match ty::impl_or_trait_item(self.tcx(), local_def(method.id)) {
                    ty::ImplOrTraitItem::MethodTraitItem(ty_method) => {
                        reject_non_type_param_bounds(
                            self.tcx(),
                            method.span,
                            &ty_method.predicates);
                        reject_shadowing_type_parameters(
                            self.tcx(),
                            method.span,
                            &ty_method.generics);
                    }
                    _ => {}
                }
            }
        }

        visit::walk_trait_item(self, t)
    }
}

pub struct BoundsChecker<'cx,'tcx:'cx> {
    fcx: &'cx FnCtxt<'cx,'tcx>,
    span: Span,

    // This field is often attached to item impls; it is not clear
    // that `CodeExtent` is well-defined for such nodes, so pnkfelix
    // has left it as a NodeId rather than porting to CodeExtent.
    scope: ast::NodeId,

    binding_count: uint,
    cache: Option<&'cx mut HashSet<Ty<'tcx>>>,
}

impl<'cx,'tcx> BoundsChecker<'cx,'tcx> {
    pub fn new(fcx: &'cx FnCtxt<'cx,'tcx>,
               span: Span,
               scope: ast::NodeId,
               cache: Option<&'cx mut HashSet<Ty<'tcx>>>)
               -> BoundsChecker<'cx,'tcx> {
        BoundsChecker { fcx: fcx, span: span, scope: scope,
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
    pub fn check_trait_ref(&mut self, trait_ref: &ty::TraitRef<'tcx>) {
        let trait_predicates = ty::lookup_predicates(self.fcx.tcx(), trait_ref.def_id);

        let bounds = self.fcx.instantiate_bounds(self.span,
                                                 trait_ref.substs,
                                                 &trait_predicates);

        self.fcx.add_obligations_for_parameters(
            traits::ObligationCause::new(
                self.span,
                self.fcx.body_id,
                traits::ItemObligation(trait_ref.def_id)),
            &bounds);

        for &ty in trait_ref.substs.types.iter() {
            self.check_traits_in_ty(ty);
        }
    }

    pub fn check_ty(&mut self, ty: Ty<'tcx>) {
        ty.fold_with(self);
    }

    fn check_traits_in_ty(&mut self, ty: Ty<'tcx>) {
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
        where T : TypeFoldable<'tcx> + Repr<'tcx>
    {
        self.binding_count += 1;
        let value = liberate_late_bound_regions(
            self.fcx.tcx(),
            region::DestructionScopeData::new(self.scope),
            binder);
        debug!("BoundsChecker::fold_binder: late-bound regions replaced: {} at scope: {:?}",
               value.repr(self.tcx()), self.scope);
        let value = value.fold_with(self);
        self.binding_count -= 1;
        ty::Binder(value)
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        debug!("BoundsChecker t={}",
               t.repr(self.tcx()));

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
            ty::ty_struct(type_id, substs) |
            ty::ty_enum(type_id, substs) => {
                let type_predicates = ty::lookup_predicates(self.fcx.tcx(), type_id);
                let bounds = self.fcx.instantiate_bounds(self.span, substs,
                                                         &type_predicates);

                if self.binding_count == 0 {
                    self.fcx.add_obligations_for_parameters(
                        traits::ObligationCause::new(self.span,
                                                     self.fcx.body_id,
                                                     traits::ItemObligation(type_id)),
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
                                                     traits::ItemObligation(type_id)),
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
                            struct_def: &ast::StructDef)
                            -> AdtVariant<'tcx> {
    let fields =
        struct_def.fields
        .iter()
        .map(|field| {
            let field_ty = ty::node_id_to_type(fcx.tcx(), field.node.id);
            let field_ty = fcx.instantiate_type_scheme(field.span,
                                                       &fcx.inh.param_env.free_substs,
                                                       &field_ty);
            AdtField { ty: field_ty, span: field.span }
        })
        .collect();
    AdtVariant { fields: fields }
}

fn enum_variants<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                           enum_def: &ast::EnumDef)
                           -> Vec<AdtVariant<'tcx>> {
    enum_def.variants.iter()
        .map(|variant| {
            match variant.node.kind {
                ast::TupleVariantKind(ref args) if args.len() > 0 => {
                    let ctor_ty = ty::node_id_to_type(fcx.tcx(), variant.node.id);

                    // the regions in the argument types come from the
                    // enum def'n, and hence will all be early bound
                    let arg_tys =
                        ty::no_late_bound_regions(
                            fcx.tcx(), &ty::ty_fn_args(ctor_ty)).unwrap();
                    AdtVariant {
                        fields: args.iter().enumerate().map(|(index, arg)| {
                            let arg_ty = arg_tys[index];
                            let arg_ty =
                                fcx.instantiate_type_scheme(variant.span,
                                                            &fcx.inh.param_env.free_substs,
                                                            &arg_ty);
                            AdtField {
                                ty: arg_ty,
                                span: arg.ty.span
                            }
                        }).collect()
                    }
                }
                ast::TupleVariantKind(_) => {
                    AdtVariant {
                        fields: Vec::new()
                    }
                }
                ast::StructVariantKind(ref struct_def) => {
                    struct_variant(fcx, &**struct_def)
                }
            }
        })
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
            ty::Predicate::Equate(..) |
            ty::Predicate::TypeOutlives(..) |
            ty::Predicate::RegionOutlives(..) => {
            }
        }
    }
    result
}

///////////////////////////////////////////////////////////////////////////
// Special drop trait checking

fn check_struct_safe_for_destructor<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                              span: Span,
                                              struct_did: ast::DefId) {
    let struct_tpt = ty::lookup_item_type(fcx.tcx(), struct_did);
    if struct_tpt.generics.has_type_params(subst::TypeSpace)
        || struct_tpt.generics.has_region_params(subst::TypeSpace)
    {
        span_err!(fcx.tcx().sess, span, E0141,
                  "cannot implement a destructor on a structure \
                   with type parameters");
        span_note!(fcx.tcx().sess, span,
                   "use \"#[unsafe_destructor]\" on the implementation \
                    to force the compiler to allow this");
    }
}
