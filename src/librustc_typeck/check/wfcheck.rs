// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use check::{Inherited, FnCtxt};
use constrained_type_params::{identify_constrained_type_params, Parameter};

use hir::def_id::DefId;
use rustc::traits::{self, ObligationCauseCode};
use rustc::ty::{self, Lift, Ty, TyCtxt};
use rustc::ty::util::ExplicitSelf;
use rustc::util::nodemap::{FxHashSet, FxHashMap};
use rustc::middle::lang_items;

use syntax::ast;
use syntax::feature_gate::{self, GateIssue};
use syntax_pos::Span;
use errors::{DiagnosticBuilder, DiagnosticId};

use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir;

pub struct CheckTypeWellFormedVisitor<'a, 'tcx:'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    code: ObligationCauseCode<'tcx>,
}

/// Helper type of a temporary returned by .for_item(...).
/// Necessary because we can't write the following bound:
/// F: for<'b, 'tcx> where 'gcx: 'tcx FnOnce(FnCtxt<'b, 'gcx, 'tcx>).
struct CheckWfFcxBuilder<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    inherited: super::InheritedBuilder<'a, 'gcx, 'tcx>,
    code: ObligationCauseCode<'gcx>,
    id: ast::NodeId,
    span: Span,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'a, 'gcx, 'tcx> CheckWfFcxBuilder<'a, 'gcx, 'tcx> {
    fn with_fcx<F>(&'tcx mut self, f: F) where
        F: for<'b> FnOnce(&FnCtxt<'b, 'gcx, 'tcx>,
                          &mut CheckTypeWellFormedVisitor<'b, 'gcx>) -> Vec<Ty<'tcx>>
    {
        let code = self.code.clone();
        let id = self.id;
        let span = self.span;
        let param_env = self.param_env;
        self.inherited.enter(|inh| {
            let fcx = FnCtxt::new(&inh, param_env, id);
            let wf_tys = f(&fcx, &mut CheckTypeWellFormedVisitor {
                tcx: fcx.tcx.global_tcx(),
                code,
            });
            fcx.select_all_obligations_or_error();
            fcx.regionck_item(id, span, &wf_tys);
        });
    }
}

impl<'a, 'gcx> CheckTypeWellFormedVisitor<'a, 'gcx> {
    pub fn new(tcx: TyCtxt<'a, 'gcx, 'gcx>)
               -> CheckTypeWellFormedVisitor<'a, 'gcx> {
        CheckTypeWellFormedVisitor {
            tcx,
            code: ObligationCauseCode::MiscObligation
        }
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
        let tcx = self.tcx;
        debug!("check_item_well_formed(it.id={}, it.name={})",
               item.id,
               tcx.item_path_str(tcx.hir.local_def_id(item.id)));

        match item.node {
            // Right now we check that every default trait implementation
            // has an implementation of itself. Basically, a case like:
            //
            // `impl Trait for T {}`
            //
            // has a requirement of `T: Trait` which was required for default
            // method implementations. Although this could be improved now that
            // there's a better infrastructure in place for this, it's being left
            // for a follow-up work.
            //
            // Since there's such a requirement, we need to check *just* positive
            // implementations, otherwise things like:
            //
            // impl !Send for T {}
            //
            // won't be allowed unless there's an *explicit* implementation of `Send`
            // for `T`
            hir::ItemImpl(_, polarity, defaultness, _, ref trait_ref, ref self_ty, _) => {
                let is_auto = tcx.impl_trait_ref(tcx.hir.local_def_id(item.id))
                                 .map_or(false, |trait_ref| tcx.trait_is_auto(trait_ref.def_id));
                if let (hir::Defaultness::Default { .. }, true) = (defaultness, is_auto) {
                    tcx.sess.span_err(item.span, "impls of auto traits cannot be default");
                }
                if polarity == hir::ImplPolarity::Positive {
                    self.check_impl(item, self_ty, trait_ref);
                } else {
                    // FIXME(#27579) what amount of WF checking do we need for neg impls?
                    if trait_ref.is_some() && !is_auto {
                        span_err!(tcx.sess, item.span, E0192,
                                  "negative impls are only allowed for \
                                   auto traits (e.g., `Send` and `Sync`)")
                    }
                }
            }
            hir::ItemFn(..) => {
                self.check_item_fn(item);
            }
            hir::ItemStatic(..) => {
                self.check_item_type(item);
            }
            hir::ItemConst(..) => {
                self.check_item_type(item);
            }
            hir::ItemStruct(ref struct_def, ref ast_generics) => {
                self.check_type_defn(item, false, |fcx| {
                    vec![fcx.non_enum_variant(struct_def)]
                });

                self.check_variances_for_type_defn(item, ast_generics);
            }
            hir::ItemUnion(ref struct_def, ref ast_generics) => {
                self.check_type_defn(item, true, |fcx| {
                    vec![fcx.non_enum_variant(struct_def)]
                });

                self.check_variances_for_type_defn(item, ast_generics);
            }
            hir::ItemEnum(ref enum_def, ref ast_generics) => {
                self.check_type_defn(item, true, |fcx| {
                    fcx.enum_variants(enum_def)
                });

                self.check_variances_for_type_defn(item, ast_generics);
            }
            hir::ItemTrait(..) => {
                self.check_trait(item);
            }
            _ => {}
        }
    }

    fn check_associated_item(&mut self,
                             item_id: ast::NodeId,
                             span: Span,
                             sig_if_method: Option<&hir::MethodSig>) {
        let code = self.code.clone();
        self.for_id(item_id, span).with_fcx(|fcx, this| {
            let item = fcx.tcx.associated_item(fcx.tcx.hir.local_def_id(item_id));

            let (mut implied_bounds, self_ty) = match item.container {
                ty::TraitContainer(_) => (vec![], fcx.tcx.mk_self_type()),
                ty::ImplContainer(def_id) => (fcx.impl_implied_bounds(def_id, span),
                                              fcx.tcx.type_of(def_id))
            };

            match item.kind {
                ty::AssociatedKind::Const => {
                    let ty = fcx.tcx.type_of(item.def_id);
                    let ty = fcx.normalize_associated_types_in(span, &ty);
                    fcx.register_wf_obligation(ty, span, code.clone());
                }
                ty::AssociatedKind::Method => {
                    reject_shadowing_type_parameters(fcx.tcx, item.def_id);
                    let sig = fcx.tcx.fn_sig(item.def_id);
                    let sig = fcx.normalize_associated_types_in(span, &sig);
                    this.check_fn_or_method(fcx, span, sig,
                                            item.def_id, &mut implied_bounds);
                    let sig_if_method = sig_if_method.expect("bad signature for method");
                    this.check_method_receiver(fcx, sig_if_method, &item, self_ty);
                }
                ty::AssociatedKind::Type => {
                    if item.defaultness.has_value() {
                        let ty = fcx.tcx.type_of(item.def_id);
                        let ty = fcx.normalize_associated_types_in(span, &ty);
                        fcx.register_wf_obligation(ty, span, code.clone());
                    }
                }
            }

            implied_bounds
        })
    }

    fn for_item<'tcx>(&self, item: &hir::Item)
                      -> CheckWfFcxBuilder<'a, 'gcx, 'tcx> {
        self.for_id(item.id, item.span)
    }

    fn for_id<'tcx>(&self, id: ast::NodeId, span: Span)
                    -> CheckWfFcxBuilder<'a, 'gcx, 'tcx> {
        let def_id = self.tcx.hir.local_def_id(id);
        CheckWfFcxBuilder {
            inherited: Inherited::build(self.tcx, def_id),
            code: self.code.clone(),
            id,
            span,
            param_env: self.tcx.param_env(def_id),
        }
    }

    /// In a type definition, we check that to ensure that the types of the fields are well-formed.
    fn check_type_defn<F>(&mut self, item: &hir::Item, all_sized: bool, mut lookup_fields: F)
        where F: for<'fcx, 'tcx> FnMut(&FnCtxt<'fcx, 'gcx, 'tcx>) -> Vec<AdtVariant<'tcx>>
    {
        self.for_item(item).with_fcx(|fcx, this| {
            let variants = lookup_fields(fcx);
            let def_id = fcx.tcx.hir.local_def_id(item.id);
            let packed = fcx.tcx.adt_def(def_id).repr.packed();

            for variant in &variants {
                // For DST, or when drop needs to copy things around, all
                // intermediate types must be sized.
                let needs_drop_copy = || {
                    packed && {
                        let ty = variant.fields.last().unwrap().ty;
                        let ty = fcx.tcx.erase_regions(&ty).lift_to_tcx(this.tcx)
                            .unwrap_or_else(|| {
                                span_bug!(item.span, "inference variables in {:?}", ty)
                            });
                        ty.needs_drop(this.tcx, this.tcx.param_env(def_id))
                    }
                };
                let unsized_len = if
                    all_sized ||
                    variant.fields.is_empty() ||
                    needs_drop_copy()
                {
                    0
                } else {
                    1
                };
                for field in &variant.fields[..variant.fields.len() - unsized_len] {
                    fcx.register_bound(
                        field.ty,
                        fcx.tcx.require_lang_item(lang_items::SizedTraitLangItem),
                        traits::ObligationCause::new(field.span,
                                                     fcx.body_id,
                                                     traits::FieldSized(match item.node.adt_kind() {
                                                        Some(i) => i,
                                                        None => bug!(),
                                                     })));
                }

                // All field types must be well-formed.
                for field in &variant.fields {
                    fcx.register_wf_obligation(field.ty, field.span, this.code.clone())
                }
            }

            self.check_where_clauses(fcx, item.span, def_id);

            vec![] // no implied bounds in a struct def'n
        });
    }

    fn check_trait(&mut self, item: &hir::Item) {
        let trait_def_id = self.tcx.hir.local_def_id(item.id);
        
        self.for_item(item).with_fcx(|fcx, _| {
            self.check_trait_where_clauses(fcx, item.span, trait_def_id);
            vec![]
        });
    }

    fn check_item_fn(&mut self, item: &hir::Item) {
        self.for_item(item).with_fcx(|fcx, this| {
            let def_id = fcx.tcx.hir.local_def_id(item.id);
            let sig = fcx.tcx.fn_sig(def_id);
            let sig = fcx.normalize_associated_types_in(item.span, &sig);
            let mut implied_bounds = vec![];
            this.check_fn_or_method(fcx, item.span, sig,
                                    def_id, &mut implied_bounds);
            implied_bounds
        })
    }

    fn check_item_type(&mut self,
                       item: &hir::Item)
    {
        debug!("check_item_type: {:?}", item);

        self.for_item(item).with_fcx(|fcx, this| {
            let ty = fcx.tcx.type_of(fcx.tcx.hir.local_def_id(item.id));
            let item_ty = fcx.normalize_associated_types_in(item.span, &ty);

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

        self.for_item(item).with_fcx(|fcx, this| {
            let item_def_id = fcx.tcx.hir.local_def_id(item.id);

            match *ast_trait_ref {
                Some(ref ast_trait_ref) => {
                    let trait_ref = fcx.tcx.impl_trait_ref(item_def_id).unwrap();
                    let trait_ref =
                        fcx.normalize_associated_types_in(
                            ast_trait_ref.path.span, &trait_ref);
                    let obligations =
                        ty::wf::trait_obligations(fcx,
                                                  fcx.param_env,
                                                  fcx.body_id,
                                                  &trait_ref,
                                                  ast_trait_ref.path.span);
                    for obligation in obligations {
                        fcx.register_predicate(obligation);
                    }
                }
                None => {
                    let self_ty = fcx.tcx.type_of(item_def_id);
                    let self_ty = fcx.normalize_associated_types_in(item.span, &self_ty);
                    fcx.register_wf_obligation(self_ty, ast_self_ty.span, this.code.clone());
                }
            }

            this.check_where_clauses(fcx, item.span, item_def_id);

            fcx.impl_implied_bounds(item_def_id, item.span)
        });
    }

    /// Checks where clauses and inline bounds that are declared on def_id.
    fn check_where_clauses<'fcx, 'tcx>(&mut self,
                                       fcx: &FnCtxt<'fcx, 'gcx, 'tcx>,
                                       span: Span,
                                       def_id: DefId) {
        self.inner_check_where_clauses(fcx, span, def_id, false)
    }

    fn check_trait_where_clauses<'fcx, 'tcx>(&mut self,
                                       fcx: &FnCtxt<'fcx, 'gcx, 'tcx>,
                                       span: Span,
                                       def_id: DefId) {
        self.inner_check_where_clauses(fcx, span, def_id, true)
    }

    /// Checks where clauses and inline bounds that are declared on def_id.
    fn inner_check_where_clauses<'fcx, 'tcx>(&mut self,
                                       fcx: &FnCtxt<'fcx, 'gcx, 'tcx>,
                                       span: Span,
                                       def_id: DefId,
                                       is_trait: bool)
    {
        use ty::subst::Subst;
        use rustc::ty::TypeFoldable;

        let mut predicates = fcx.tcx.predicates_of(def_id);
        let mut substituted_predicates = Vec::new();

        let generics = self.tcx.generics_of(def_id);
        let defaulted_params = generics.types.iter()
                                             .filter(|def| def.has_default &&
                                                     def.index >= generics.parent_count() as u32);
        for param_def in defaulted_params {
            // Defaults must be well-formed.
            let d = param_def.def_id;
            fcx.register_wf_obligation(fcx.tcx.type_of(d), fcx.tcx.def_span(d), self.code.clone());
            // Check the clauses are well-formed when the param is substituted by it's default.
            // In trait definitions, predicates as `Self: Trait` and `Self: Super` are problematic.
            // Therefore we skip such predicates. This means we check less than we could.
            for pred in predicates.predicates.iter().filter(|p| !(is_trait && p.has_self_ty())) {
                let mut skip = true;
                let substs = ty::subst::Substs::for_item(fcx.tcx, def_id, |def, _| {
                    // All regions are identity.
                    fcx.tcx.mk_region(ty::ReEarlyBound(def.to_early_bound_region_data()))
                }, |def, _| {
                    let identity_substs = fcx.tcx.mk_param_from_def(def);
                    if def.index != param_def.index {
                        identity_substs
                    } else {
                        let sized = fcx.tcx.lang_items().sized_trait();
                        let pred_is_sized = match pred {
                            ty::Predicate::Trait(p) => Some(p.def_id()) == sized,
                            _ => false,
                        };
                        let default_ty = fcx.tcx.type_of(def.def_id);
                        let default_is_self = match default_ty.sty {
                            ty::TyParam(ref p) => p.is_self(),
                            _ => false
                        };
                        // In trait defs, skip `Self: Sized` when `Self` is the default.
                        if is_trait && pred_is_sized && default_is_self {
                            identity_substs
                        } else {
                            skip = false;
                            default_ty
                        }
                    }
                });
                if !skip {
                    substituted_predicates.push(pred.subst(fcx.tcx, substs));
                }
            }
        }

        predicates.predicates.extend(substituted_predicates);
        let predicates = predicates.instantiate_identity(fcx.tcx);
        let predicates = fcx.normalize_associated_types_in(span, &predicates);

        let obligations =
            predicates.predicates
                      .iter()
                      .flat_map(|p| ty::wf::predicate_obligations(fcx,
                                                                  fcx.param_env,
                                                                  fcx.body_id,
                                                                  p,
                                                                  span));

        for obligation in obligations {
            fcx.register_predicate(obligation);
        }
    }

    fn check_fn_or_method<'fcx, 'tcx>(&mut self,
                                      fcx: &FnCtxt<'fcx, 'gcx, 'tcx>,
                                      span: Span,
                                      sig: ty::PolyFnSig<'tcx>,
                                      def_id: DefId,
                                      implied_bounds: &mut Vec<Ty<'tcx>>)
    {
        let sig = fcx.normalize_associated_types_in(span, &sig);
        let sig = fcx.tcx.liberate_late_bound_regions(def_id, &sig);

        for input_ty in sig.inputs() {
            fcx.register_wf_obligation(&input_ty, span, self.code.clone());
        }
        implied_bounds.extend(sig.inputs());

        fcx.register_wf_obligation(sig.output(), span, self.code.clone());

        // FIXME(#25759) return types should not be implied bounds
        implied_bounds.push(sig.output());

        self.check_where_clauses(fcx, span, def_id);
    }

    fn check_method_receiver<'fcx, 'tcx>(&mut self,
                                         fcx: &FnCtxt<'fcx, 'gcx, 'tcx>,
                                         method_sig: &hir::MethodSig,
                                         method: &ty::AssociatedItem,
                                         self_ty: Ty<'tcx>)
    {
        // check that the method has a valid receiver type, given the type `Self`
        debug!("check_method_receiver({:?}, self_ty={:?})",
               method, self_ty);

        if !method.method_has_self_argument {
            return;
        }

        let span = method_sig.decl.inputs[0].span;

        let sig = fcx.tcx.fn_sig(method.def_id);
        let sig = fcx.normalize_associated_types_in(span, &sig);
        let sig = fcx.tcx.liberate_late_bound_regions(method.def_id, &sig);

        debug!("check_method_receiver: sig={:?}", sig);

        let self_ty = fcx.normalize_associated_types_in(span, &self_ty);
        let self_ty = fcx.tcx.liberate_late_bound_regions(
            method.def_id,
            &ty::Binder(self_ty)
        );

        let self_arg_ty = sig.inputs()[0];

        let cause = fcx.cause(span, ObligationCauseCode::MethodReceiver);
        let self_arg_ty = fcx.normalize_associated_types_in(span, &self_arg_ty);
        let self_arg_ty = fcx.tcx.liberate_late_bound_regions(
            method.def_id,
            &ty::Binder(self_arg_ty)
        );

        let mut autoderef = fcx.autoderef(span, self_arg_ty).include_raw_pointers();

        loop {
            if let Some((potential_self_ty, _)) = autoderef.next() {
                debug!("check_method_receiver: potential self type `{:?}` to match `{:?}`",
                    potential_self_ty, self_ty);

                if fcx.infcx.can_eq(fcx.param_env, self_ty, potential_self_ty).is_ok() {
                    autoderef.finalize();
                    if let Some(mut err) = fcx.demand_eqtype_with_origin(
                        &cause, self_ty, potential_self_ty) {
                        err.emit();
                    }
                    break
                }
            } else {
                fcx.tcx.sess.diagnostic().mut_span_err(
                    span, &format!("invalid `self` type: {:?}", self_arg_ty))
                .note(&format!("type must be `{:?}` or a type that dereferences to it`", self_ty))
                .help("consider changing to `self`, `&self`, `&mut self`, or `self: Box<Self>`")
                .code(DiagnosticId::Error("E0307".into()))
                .emit();
                return
            }
        }

        let is_self_ty = |ty| fcx.infcx.can_eq(fcx.param_env, self_ty, ty).is_ok();
        let self_kind = ExplicitSelf::determine(self_arg_ty, is_self_ty);

        if !fcx.tcx.sess.features.borrow().arbitrary_self_types {
            match self_kind {
                ExplicitSelf::ByValue |
                ExplicitSelf::ByReference(_, _) |
                ExplicitSelf::ByBox => (),

                ExplicitSelf::ByRawPointer(_) => {
                    feature_gate::feature_err(
                        &fcx.tcx.sess.parse_sess,
                        "arbitrary_self_types",
                        span,
                        GateIssue::Language,
                        "raw pointer `self` is unstable")
                    .help("consider changing to `self`, `&self`, `&mut self`, or `self: Box<Self>`")
                    .emit();
                }

                ExplicitSelf::Other => {
                    feature_gate::feature_err(
                        &fcx.tcx.sess.parse_sess,
                        "arbitrary_self_types",
                        span,
                        GateIssue::Language,"arbitrary `self` types are unstable")
                    .help("consider changing to `self`, `&self`, `&mut self`, or `self: Box<Self>`")
                    .emit();
                }
            }
        }
    }

    fn check_variances_for_type_defn(&self,
                                     item: &hir::Item,
                                     ast_generics: &hir::Generics)
    {
        let item_def_id = self.tcx.hir.local_def_id(item.id);
        let ty = self.tcx.type_of(item_def_id);
        if self.tcx.has_error_field(ty) {
            return;
        }

        let ty_predicates = self.tcx.predicates_of(item_def_id);
        assert_eq!(ty_predicates.parent, None);
        let variances = self.tcx.variances_of(item_def_id);

        let mut constrained_parameters: FxHashSet<_> =
            variances.iter().enumerate()
                     .filter(|&(_, &variance)| variance != ty::Bivariant)
                     .map(|(index, _)| Parameter(index as u32))
                     .collect();

        identify_constrained_type_params(self.tcx,
                                         ty_predicates.predicates.as_slice(),
                                         None,
                                         &mut constrained_parameters);

        for (index, _) in variances.iter().enumerate() {
            if constrained_parameters.contains(&Parameter(index as u32)) {
                continue;
            }

            let (span, name) = match ast_generics.params[index] {
                hir::GenericParam::Lifetime(ref ld) => (ld.lifetime.span, ld.lifetime.name.name()),
                hir::GenericParam::Type(ref tp) => (tp.span, tp.name),
            };
            self.report_bivariance(span, name);
        }
    }

    fn report_bivariance(&self,
                         span: Span,
                         param_name: ast::Name)
    {
        let mut err = error_392(self.tcx, span, param_name);

        let suggested_marker_id = self.tcx.lang_items().phantom_data();
        match suggested_marker_id {
            Some(def_id) => {
                err.help(
                    &format!("consider removing `{}` or using a marker such as `{}`",
                             param_name,
                             self.tcx.item_path_str(def_id)));
            }
            None => {
                // no lang items, no help!
            }
        }
        err.emit();
    }
}

fn reject_shadowing_type_parameters(tcx: TyCtxt, def_id: DefId) {
    let generics = tcx.generics_of(def_id);
    let parent = tcx.generics_of(generics.parent.unwrap());
    let impl_params: FxHashMap<_, _> = parent.types
                                       .iter()
                                       .map(|tp| (tp.name, tp.def_id))
                                       .collect();

    for method_param in &generics.types {
        if impl_params.contains_key(&method_param.name) {
            // Tighten up the span to focus on only the shadowing type
            let type_span = tcx.def_span(method_param.def_id);

            // The expectation here is that the original trait declaration is
            // local so it should be okay to just unwrap everything.
            let trait_def_id = impl_params[&method_param.name];
            let trait_decl_span = tcx.def_span(trait_def_id);
            error_194(tcx, type_span, trait_decl_span, method_param.name);
        }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for CheckTypeWellFormedVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'v> {
        NestedVisitorMap::None
    }

    fn visit_item(&mut self, i: &hir::Item) {
        debug!("visit_item: {:?}", i);
        self.check_item_well_formed(i);
        intravisit::walk_item(self, i);
    }

    fn visit_trait_item(&mut self, trait_item: &'v hir::TraitItem) {
        debug!("visit_trait_item: {:?}", trait_item);
        let method_sig = match trait_item.node {
            hir::TraitItemKind::Method(ref sig, _) => Some(sig),
            _ => None
        };
        self.check_associated_item(trait_item.id, trait_item.span, method_sig);
        intravisit::walk_trait_item(self, trait_item)
    }

    fn visit_impl_item(&mut self, impl_item: &'v hir::ImplItem) {
        debug!("visit_impl_item: {:?}", impl_item);
        let method_sig = match impl_item.node {
            hir::ImplItemKind::Method(ref sig, _) => Some(sig),
            _ => None
        };
        self.check_associated_item(impl_item.id, impl_item.span, method_sig);
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

impl<'a, 'gcx, 'tcx> FnCtxt<'a, 'gcx, 'tcx> {
    fn non_enum_variant(&self, struct_def: &hir::VariantData) -> AdtVariant<'tcx> {
        let fields =
            struct_def.fields().iter()
            .map(|field| {
                let field_ty = self.tcx.type_of(self.tcx.hir.local_def_id(field.id));
                let field_ty = self.normalize_associated_types_in(field.span,
                                                                  &field_ty);
                AdtField { ty: field_ty, span: field.span }
            })
            .collect();
        AdtVariant { fields: fields }
    }

    fn enum_variants(&self, enum_def: &hir::EnumDef) -> Vec<AdtVariant<'tcx>> {
        enum_def.variants.iter()
            .map(|variant| self.non_enum_variant(&variant.node.data))
            .collect()
    }

    fn impl_implied_bounds(&self, impl_def_id: DefId, span: Span) -> Vec<Ty<'tcx>> {
        match self.tcx.impl_trait_ref(impl_def_id) {
            Some(ref trait_ref) => {
                // Trait impl: take implied bounds from all types that
                // appear in the trait reference.
                let trait_ref = self.normalize_associated_types_in(span, trait_ref);
                trait_ref.substs.types().collect()
            }

            None => {
                // Inherent impl: take implied bounds from the self type.
                let self_ty = self.tcx.type_of(impl_def_id);
                let self_ty = self.normalize_associated_types_in(span, &self_ty);
                vec![self_ty]
            }
        }
    }
}

fn error_392<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, span: Span, param_name: ast::Name)
                       -> DiagnosticBuilder<'tcx> {
    let mut err = struct_span_err!(tcx.sess, span, E0392,
                  "parameter `{}` is never used", param_name);
    err.span_label(span, "unused type parameter");
    err
}

fn error_194(tcx: TyCtxt, span: Span, trait_decl_span: Span, name: ast::Name) {
    struct_span_err!(tcx.sess, span, E0194,
              "type parameter `{}` shadows another type parameter of the same name",
              name)
        .span_label(span, "shadows another type parameter")
        .span_label(trait_decl_span, format!("first `{}` declared here", name))
        .emit();
}
