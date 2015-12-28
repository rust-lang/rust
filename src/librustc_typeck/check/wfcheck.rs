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
use check::{FnCtxt, Inherited, blank_fn_ctxt, regionck};
use constrained_type_params::{identify_constrained_type_params, Parameter};
use CrateCtxt;
use middle::def_id::DefId;
use middle::region::{CodeExtent};
use middle::subst::{self, TypeSpace, FnSpace, ParamSpace, SelfSpace};
use middle::traits;
use middle::ty::{self, Ty};
use middle::ty::fold::{TypeFolder};

use std::cell::RefCell;
use std::collections::HashSet;
use syntax::ast;
use syntax::codemap::{Span};
use syntax::parse::token::{special_idents};
use rustc_front::intravisit::{self, Visitor};
use rustc_front::hir;

pub struct CheckTypeWellFormedVisitor<'ccx, 'tcx:'ccx> {
    ccx: &'ccx CrateCtxt<'ccx, 'tcx>,
    code: traits::ObligationCauseCode<'tcx>,
}

impl<'ccx, 'tcx> CheckTypeWellFormedVisitor<'ccx, 'tcx> {
    pub fn new(ccx: &'ccx CrateCtxt<'ccx, 'tcx>)
               -> CheckTypeWellFormedVisitor<'ccx, 'tcx> {
        CheckTypeWellFormedVisitor {
            ccx: ccx,
            code: traits::ObligationCauseCode::MiscObligation
        }
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
            hir::ItemImpl(_, hir::ImplPolarity::Positive, _,
                          ref trait_ref, ref self_ty, _) => {
                self.check_impl(item, self_ty, trait_ref);
            }
            hir::ItemImpl(_, hir::ImplPolarity::Negative, _, Some(_), _, _) => {
                // FIXME(#27579) what amount of WF checking do we need for neg impls?

                let trait_ref = ccx.tcx.impl_trait_ref(ccx.tcx.map.local_def_id(item.id)).unwrap();
                ccx.tcx.populate_implementations_for_trait_if_necessary(trait_ref.def_id);
                match ccx.tcx.lang_items.to_builtin_kind(trait_ref.def_id) {
                    Some(ty::BoundSend) | Some(ty::BoundSync) => {}
                    Some(_) | None => {
                        if !ccx.tcx.trait_has_default_impl(trait_ref.def_id) {
                            error_192(ccx, item.span);
                        }
                    }
                }
            }
            hir::ItemFn(_, _, _, _, _, ref body) => {
                self.check_item_fn(item, body);
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
                self.check_trait(item, items);
            }
            _ => {}
        }
    }

    fn check_trait_or_impl_item(&mut self, item_id: ast::NodeId, span: Span) {
        let code = self.code.clone();
        self.with_fcx(item_id, span, |fcx, this| {
            let free_substs = &fcx.inh.infcx.parameter_environment.free_substs;
            let free_id_outlive = fcx.inh.infcx.parameter_environment.free_id_outlive;

            let item = fcx.tcx().impl_or_trait_item(fcx.tcx().map.local_def_id(item_id));

            let (mut implied_bounds, self_ty) = match item.container() {
                ty::TraitContainer(_) => (vec![], fcx.tcx().mk_self_type()),
                ty::ImplContainer(def_id) => (impl_implied_bounds(fcx, def_id, span),
                                              fcx.tcx().lookup_item_type(def_id).ty)
            };

            match item {
                ty::ConstTraitItem(assoc_const) => {
                    let ty = fcx.instantiate_type_scheme(span, free_substs, &assoc_const.ty);
                    fcx.register_wf_obligation(ty, span, code.clone());
                }
                ty::MethodTraitItem(method) => {
                    reject_shadowing_type_parameters(fcx.tcx(), span, &method.generics);
                    let method_ty = fcx.instantiate_type_scheme(span, free_substs, &method.fty);
                    let predicates = fcx.instantiate_bounds(span, free_substs, &method.predicates);
                    this.check_fn_or_method(fcx, span, &method_ty, &predicates,
                                            free_id_outlive, &mut implied_bounds);
                    this.check_method_receiver(fcx, span, &method,
                                               free_id_outlive, self_ty);
                }
                ty::TypeTraitItem(assoc_type) => {
                    if let Some(ref ty) = assoc_type.ty {
                        let ty = fcx.instantiate_type_scheme(span, free_substs, ty);
                        fcx.register_wf_obligation(ty, span, code.clone());
                    }
                }
            }

            implied_bounds
        })
    }

    fn with_item_fcx<F>(&mut self, item: &hir::Item, f: F) where
        F: for<'fcx> FnMut(&FnCtxt<'fcx, 'tcx>,
                           &mut CheckTypeWellFormedVisitor<'ccx,'tcx>) -> Vec<Ty<'tcx>>,
    {
        self.with_fcx(item.id, item.span, f)
    }

    fn with_fcx<F>(&mut self, id: ast::NodeId, span: Span, mut f: F) where
        F: for<'fcx> FnMut(&FnCtxt<'fcx, 'tcx>,
                           &mut CheckTypeWellFormedVisitor<'ccx,'tcx>) -> Vec<Ty<'tcx>>,
    {
        let ccx = self.ccx;
        let param_env = ty::ParameterEnvironment::for_item(ccx.tcx, id);
        let tables = RefCell::new(ty::Tables::empty());
        let inh = Inherited::new(ccx.tcx, &tables, param_env);
        let fcx = blank_fn_ctxt(ccx, &inh, ty::FnDiverging, id);
        let wf_tys = f(&fcx, self);
        fcx.select_all_obligations_or_error();
        regionck::regionck_item(&fcx, id, span, &wf_tys);
    }

    /// In a type definition, we check that to ensure that the types of the fields are well-formed.
    fn check_type_defn<F>(&mut self, item: &hir::Item, mut lookup_fields: F) where
        F: for<'fcx> FnMut(&FnCtxt<'fcx, 'tcx>) -> Vec<AdtVariant<'tcx>>,
    {
        self.with_item_fcx(item, |fcx, this| {
            let variants = lookup_fields(fcx);

            for variant in &variants {
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

                // All field types must be well-formed.
                for field in &variant.fields {
                    fcx.register_wf_obligation(field.ty, field.span, this.code.clone())
                }
            }

            let free_substs = &fcx.inh.infcx.parameter_environment.free_substs;
            let predicates = fcx.tcx().lookup_predicates(fcx.tcx().map.local_def_id(item.id));
            let predicates = fcx.instantiate_bounds(item.span, free_substs, &predicates);
            this.check_where_clauses(fcx, item.span, &predicates);

            vec![] // no implied bounds in a struct def'n
        });
    }

    fn check_trait(&mut self,
                   item: &hir::Item,
                   items: &[hir::TraitItem])
    {
        let trait_def_id = self.tcx().map.local_def_id(item.id);

        if self.ccx.tcx.trait_has_default_impl(trait_def_id) {
            if !items.is_empty() {
                error_380(self.ccx, item.span);
            }
        }

        self.with_item_fcx(item, |fcx, this| {
            let free_substs = &fcx.inh.infcx.parameter_environment.free_substs;
            let predicates = fcx.tcx().lookup_predicates(trait_def_id);
            let predicates = fcx.instantiate_bounds(item.span, free_substs, &predicates);
            this.check_where_clauses(fcx, item.span, &predicates);
            vec![]
        });
    }

    fn check_item_fn(&mut self,
                     item: &hir::Item,
                     body: &hir::Block)
    {
        self.with_item_fcx(item, |fcx, this| {
            let free_substs = &fcx.inh.infcx.parameter_environment.free_substs;
            let type_scheme = fcx.tcx().lookup_item_type(fcx.tcx().map.local_def_id(item.id));
            let item_ty = fcx.instantiate_type_scheme(item.span, free_substs, &type_scheme.ty);
            let bare_fn_ty = match item_ty.sty {
                ty::TyBareFn(_, ref bare_fn_ty) => bare_fn_ty,
                _ => {
                    this.tcx().sess.span_bug(item.span, "Fn item without bare fn type");
                }
            };

            let predicates = fcx.tcx().lookup_predicates(fcx.tcx().map.local_def_id(item.id));
            let predicates = fcx.instantiate_bounds(item.span, free_substs, &predicates);

            let mut implied_bounds = vec![];
            let free_id_outlive = fcx.tcx().region_maps.call_site_extent(item.id, body.id);
            this.check_fn_or_method(fcx, item.span, bare_fn_ty, &predicates,
                                    free_id_outlive, &mut implied_bounds);
            implied_bounds
        })
    }

    fn check_item_type(&mut self,
                       item: &hir::Item)
    {
        debug!("check_item_type: {:?}", item);

        self.with_item_fcx(item, |fcx, this| {
            let type_scheme = fcx.tcx().lookup_item_type(fcx.tcx().map.local_def_id(item.id));
            let item_ty = fcx.instantiate_type_scheme(item.span,
                                                      &fcx.inh
                                                          .infcx
                                                          .parameter_environment
                                                          .free_substs,
                                                      &type_scheme.ty);

            fcx.register_wf_obligation(item_ty, item.span, this.code.clone());

            vec![] // no implied bounds in a const etc
        });
    }

    fn check_impl(&mut self,
                  item: &hir::Item,
                  ast_self_ty: &hir::Ty,
                  ast_trait_ref: &Option<hir::TraitRef>)
    {
        debug!("check_impl: {:?}", item);

        self.with_item_fcx(item, |fcx, this| {
            let free_substs = &fcx.inh.infcx.parameter_environment.free_substs;
            let item_def_id = fcx.tcx().map.local_def_id(item.id);

            match *ast_trait_ref {
                Some(ref ast_trait_ref) => {
                    let trait_ref = fcx.tcx().impl_trait_ref(item_def_id).unwrap();
                    let trait_ref =
                        fcx.instantiate_type_scheme(
                            ast_trait_ref.path.span, free_substs, &trait_ref);
                    let obligations =
                        ty::wf::trait_obligations(fcx.infcx(),
                                                  fcx.body_id,
                                                  &trait_ref,
                                                  ast_trait_ref.path.span);
                    for obligation in obligations {
                        fcx.register_predicate(obligation);
                    }
                }
                None => {
                    let self_ty = fcx.tcx().node_id_to_type(item.id);
                    let self_ty = fcx.instantiate_type_scheme(item.span, free_substs, &self_ty);
                    fcx.register_wf_obligation(self_ty, ast_self_ty.span, this.code.clone());
                }
            }

            let predicates = fcx.tcx().lookup_predicates(item_def_id);
            let predicates = fcx.instantiate_bounds(item.span, free_substs, &predicates);
            this.check_where_clauses(fcx, item.span, &predicates);

            impl_implied_bounds(fcx, fcx.tcx().map.local_def_id(item.id), item.span)
        });
    }

    fn check_where_clauses<'fcx>(&mut self,
                                 fcx: &FnCtxt<'fcx,'tcx>,
                                 span: Span,
                                 predicates: &ty::InstantiatedPredicates<'tcx>)
    {
        let obligations =
            predicates.predicates
                      .iter()
                      .flat_map(|p| ty::wf::predicate_obligations(fcx.infcx(),
                                                                  fcx.body_id,
                                                                  p,
                                                                  span));

        for obligation in obligations {
            fcx.register_predicate(obligation);
        }
    }

    fn check_fn_or_method<'fcx>(&mut self,
                                fcx: &FnCtxt<'fcx,'tcx>,
                                span: Span,
                                fty: &ty::BareFnTy<'tcx>,
                                predicates: &ty::InstantiatedPredicates<'tcx>,
                                free_id_outlive: CodeExtent,
                                implied_bounds: &mut Vec<Ty<'tcx>>)
    {
        let free_substs = &fcx.inh.infcx.parameter_environment.free_substs;
        let fty = fcx.instantiate_type_scheme(span, free_substs, fty);
        let sig = fcx.tcx().liberate_late_bound_regions(free_id_outlive, &fty.sig);

        for &input_ty in &sig.inputs {
            fcx.register_wf_obligation(input_ty, span, self.code.clone());
        }
        implied_bounds.extend(sig.inputs);

        match sig.output {
            ty::FnConverging(output) => {
                fcx.register_wf_obligation(output, span, self.code.clone());

                // FIXME(#25759) return types should not be implied bounds
                implied_bounds.push(output);
            }
            ty::FnDiverging => { }
        }

        self.check_where_clauses(fcx, span, predicates);
    }

    fn check_method_receiver<'fcx>(&mut self,
                                   fcx: &FnCtxt<'fcx,'tcx>,
                                   span: Span,
                                   method: &ty::Method<'tcx>,
                                   free_id_outlive: CodeExtent,
                                   self_ty: ty::Ty<'tcx>)
    {
        // check that the type of the method's receiver matches the
        // method's first parameter.

        let free_substs = &fcx.inh.infcx.parameter_environment.free_substs;
        let fty = fcx.instantiate_type_scheme(span, free_substs, &method.fty);
        let sig = fcx.tcx().liberate_late_bound_regions(free_id_outlive, &fty.sig);

        debug!("check_method_receiver({:?},cat={:?},self_ty={:?},sig={:?})",
               method.name, method.explicit_self, self_ty, sig);

        let rcvr_ty = match method.explicit_self {
            ty::ExplicitSelfCategory::Static => return,
            ty::ExplicitSelfCategory::ByValue => self_ty,
            ty::ExplicitSelfCategory::ByReference(region, mutability) => {
                fcx.tcx().mk_ref(fcx.tcx().mk_region(region), ty::TypeAndMut {
                    ty: self_ty,
                    mutbl: mutability
                })
            }
            ty::ExplicitSelfCategory::ByBox => fcx.tcx().mk_box(self_ty)
        };
        let rcvr_ty = fcx.instantiate_type_scheme(span, free_substs, &rcvr_ty);
        let rcvr_ty = fcx.tcx().liberate_late_bound_regions(free_id_outlive,
                                                            &ty::Binder(rcvr_ty));

        debug!("check_method_receiver: receiver ty = {:?}", rcvr_ty);

        let _ = ::require_same_types(
            fcx.tcx(), Some(fcx.infcx()), false, span,
            sig.inputs[0], rcvr_ty,
            || "mismatched method receiver".to_owned()
        );
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
        error_392(self.tcx(), span, param_name);

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

fn reject_shadowing_type_parameters<'tcx>(tcx: &ty::ctxt<'tcx>,
                                          span: Span,
                                          generics: &ty::Generics<'tcx>) {
    let impl_params = generics.types.get_slice(subst::TypeSpace).iter()
        .map(|tp| tp.name).collect::<HashSet<_>>();

    for method_param in generics.types.get_slice(subst::FnSpace) {
        if impl_params.contains(&method_param.name) {
            error_194(tcx, span, method_param.name);
        }
    }
}

impl<'ccx, 'tcx, 'v> Visitor<'v> for CheckTypeWellFormedVisitor<'ccx, 'tcx> {
    fn visit_item(&mut self, i: &hir::Item) {
        debug!("visit_item: {:?}", i);
        self.check_item_well_formed(i);
        intravisit::walk_item(self, i);
    }

    fn visit_trait_item(&mut self, trait_item: &'v hir::TraitItem) {
        debug!("visit_trait_item: {:?}", trait_item);
        self.check_trait_or_impl_item(trait_item.id, trait_item.span);
        intravisit::walk_trait_item(self, trait_item)
    }

    fn visit_impl_item(&mut self, impl_item: &'v hir::ImplItem) {
        debug!("visit_impl_item: {:?}", impl_item);
        self.check_trait_or_impl_item(impl_item.id, impl_item.span);
        intravisit::walk_impl_item(self, impl_item)
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

fn impl_implied_bounds<'fcx,'tcx>(fcx: &FnCtxt<'fcx, 'tcx>,
                                  impl_def_id: DefId,
                                  span: Span)
                                  -> Vec<Ty<'tcx>>
{
    let free_substs = &fcx.inh.infcx.parameter_environment.free_substs;
    match fcx.tcx().impl_trait_ref(impl_def_id) {
        Some(ref trait_ref) => {
            // Trait impl: take implied bounds from all types that
            // appear in the trait reference.
            let trait_ref = fcx.instantiate_type_scheme(span, free_substs, trait_ref);
            trait_ref.substs.types.as_slice().to_vec()
        }

        None => {
            // Inherent impl: take implied bounds from the self type.
            let self_ty = fcx.tcx().lookup_item_type(impl_def_id).ty;
            let self_ty = fcx.instantiate_type_scheme(span, free_substs, &self_ty);
            vec![self_ty]
        }
    }
}

pub fn error_192<'ccx,'tcx>(ccx: &'ccx CrateCtxt<'ccx, 'tcx>, span: Span) {
    span_err!(ccx.tcx.sess, span, E0192,
              "negative impls are only allowed for traits with \
               default impls (e.g., `Send` and `Sync`)")
}

pub fn error_380<'ccx,'tcx>(ccx: &'ccx CrateCtxt<'ccx, 'tcx>, span: Span) {
    span_err!(ccx.tcx.sess, span, E0380,
              "traits with default impls (`e.g. unsafe impl \
               Trait for ..`) must have no methods or associated items")
}

pub fn error_392<'tcx>(tcx: &ty::ctxt<'tcx>, span: Span, param_name: ast::Name)  {
    span_err!(tcx.sess, span, E0392,
              "parameter `{}` is never used", param_name);
}

pub fn error_194<'tcx>(tcx: &ty::ctxt<'tcx>, span: Span, name: ast::Name) {
    span_err!(tcx.sess, span, E0194,
              "type parameter `{}` shadows another type parameter of the same name",
              name);
}
