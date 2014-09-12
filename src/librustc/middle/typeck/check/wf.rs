// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::subst::{Subst};
use middle::traits;
use middle::ty;
use middle::ty_fold::{TypeFolder, TypeFoldable};
use middle::typeck::astconv::AstConv;
use middle::typeck::check::{FnCtxt, Inherited, blank_fn_ctxt, vtable2, regionck};
use middle::typeck::check::regionmanip::replace_late_bound_regions_in_fn_sig;
use middle::typeck::CrateCtxt;
use util::ppaux::Repr;

use std::collections::HashSet;
use syntax::ast;
use syntax::ast_util::{local_def};
use syntax::codemap::Span;
use syntax::visit;
use syntax::visit::Visitor;

pub struct CheckTypeWellFormedVisitor<'ccx, 'tcx:'ccx> {
    ccx: &'ccx CrateCtxt<'ccx, 'tcx>,
    cache: HashSet<ty::t>
}

impl<'ccx, 'tcx> CheckTypeWellFormedVisitor<'ccx, 'tcx> {
    pub fn new(ccx: &'ccx CrateCtxt<'ccx, 'tcx>) -> CheckTypeWellFormedVisitor<'ccx, 'tcx> {
        CheckTypeWellFormedVisitor { ccx: ccx, cache: HashSet::new() }
    }

    fn check_item_well_formed(&mut self, ccx: &CrateCtxt, item: &ast::Item) {
        /*!
         * Checks that the field types (in a struct def'n) or
         * argument types (in an enum def'n) are well-formed,
         * meaning that they do not require any constraints not
         * declared in the struct definition itself.
         * For example, this definition would be illegal:
         *
         *     struct Ref<'a, T> { x: &'a T }
         *
         * because the type did not declare that `T:'a`.
         *
         * We do this check as a pre-pass before checking fn bodies
         * because if these constraints are not included it frequently
         * leads to confusing errors in fn bodies. So it's better to check
         * the types first.
         */

        debug!("check_item_well_formed(it.id={}, it.ident={})",
               item.id,
               ty::item_path_str(ccx.tcx, local_def(item.id)));

        let ccx = self.ccx;
        match item.node {
            ast::ItemImpl(..) => {
                self.check_impl(item);
            }
            ast::ItemFn(..) => {
                self.check_item_type(item);
            }
            ast::ItemStatic(..) => {
                self.check_item_type(item);
            }
            ast::ItemStruct(..) => {
                self.check_type_defn(item, |fcx| {
                    ty::struct_fields(ccx.tcx, local_def(item.id),
                                      &fcx.inh.param_env.free_substs)
                        .iter()
                        .map(|f| f.mt.ty)
                        .collect()
                });
            }
            ast::ItemEnum(..) => {
                self.check_type_defn(item, |fcx| {
                    ty::substd_enum_variants(ccx.tcx, local_def(item.id),
                                             &fcx.inh.param_env.free_substs)
                        .iter()
                        .flat_map(|variant| {
                            variant.args
                                .iter()
                                .map(|&arg_ty| arg_ty)
                        })
                        .collect()
                });
            }
            _ => {}
        }
    }

    fn with_fcx(&mut self,
                ccx: &CrateCtxt,
                item: &ast::Item,
                f: |&mut CheckTypeWellFormedVisitor, &FnCtxt|) {
        let item_def_id = local_def(item.id);
        let polytype = ty::lookup_item_type(ccx.tcx, item_def_id);
        let param_env =
            ty::construct_parameter_environment(ccx.tcx,
                                                item.span,
                                                &polytype.generics,
                                                item.id);
        let inh = Inherited::new(ccx.tcx, param_env);
        let fcx = blank_fn_ctxt(ccx, &inh, polytype.ty, item.id);
        f(self, &fcx);
        vtable2::select_all_fcx_obligations_or_error(&fcx);
        regionck::regionck_item(&fcx, item);
        vtable2::check_builtin_bound_obligations(&fcx);
    }

    fn check_type_defn(&mut self,
                       item: &ast::Item,
                       lookup_fields: |&FnCtxt| -> Vec<ty::t>)
    {
        /*!
         * In a type definition, we check that to ensure that the types of the fields are
         * well-formed.
         */

        self.with_fcx(self.ccx, item, |this, fcx| {
            let field_tys = lookup_fields(fcx);
            let mut bounds_checker = BoundsChecker::new(fcx, item.span,
                                                        item.id, Some(&mut this.cache));
            for &ty in field_tys.iter() {
                // Regions are checked below.
                bounds_checker.check_traits_in_ty(ty);
            }

            regionck::regionck_ensure_component_tys_wf(
                fcx, item.span, field_tys.as_slice());
        });
    }

    fn check_item_type(&mut self,
                       item: &ast::Item)
    {
        self.with_fcx(self.ccx, item, |this, fcx| {
            let mut bounds_checker = BoundsChecker::new(fcx, item.span,
                                                        item.id, Some(&mut this.cache));
            let polytype = ty::lookup_item_type(fcx.tcx(), local_def(item.id));
            let item_ty = polytype.ty.subst(fcx.tcx(), &fcx.inh.param_env.free_substs);
            bounds_checker.check_traits_in_ty(item_ty);
        });
    }

    fn check_impl(&mut self,
                  item: &ast::Item)
    {
        self.with_fcx(self.ccx, item, |this, fcx| {
            let mut bounds_checker = BoundsChecker::new(fcx, item.span,
                                                        item.id, Some(&mut this.cache));

            let self_ty = ty::node_id_to_type(fcx.tcx(), item.id);
            let self_ty = self_ty.subst(fcx.tcx(), &fcx.inh.param_env.free_substs);

            bounds_checker.check_traits_in_ty(self_ty);

            let trait_ref = match ty::impl_trait_ref(fcx.tcx(), local_def(item.id)) {
                None => { return; }
                Some(t) => { t }
            };
            let trait_ref = (*trait_ref).subst(fcx.tcx(), &fcx.inh.param_env.free_substs);

            // We are stricter on the trait-ref in an impl than the
            // self-type.  In particular, we enforce region
            // relationships. The reason for this is that (at least
            // presently) "appyling" an impl does not require that the
            // application site check the well-formedness constraints on the
            // trait reference. Instead, this is done at the impl site.
            // Arguably this is wrong and we should treat the trait-reference
            // the same way as we treat the self-type.
            bounds_checker.check_trait_ref(&trait_ref);

            let trait_def = ty::lookup_trait_def(fcx.tcx(), trait_ref.def_id);

            let cause =
                traits::ObligationCause::new(
                    item.span,
                    traits::ItemObligation(trait_ref.def_id));

            // Find the supertrait bounds. This will add `int:Bar`.
            //
            // FIXME -- This is a bit ill-factored. There is very similar
            // code in traits::util::obligations_for_generics.
            fcx.add_region_obligations_for_type_parameter(item.span,
                                                          ty::ParamTy::for_self(trait_ref.def_id),
                                                          &trait_def.bounds,
                                                          trait_ref.self_ty());
            for builtin_bound in trait_def.bounds.builtin_bounds.iter() {
                fcx.register_obligation(
                    traits::obligation_for_builtin_bound(fcx.tcx(),
                                                         cause,
                                                         trait_ref.self_ty(),
                                                         builtin_bound));
            }
            for trait_bound in trait_def.bounds.trait_bounds.iter() {
                let trait_bound = trait_bound.subst(fcx.tcx(), &trait_ref.substs);
                fcx.register_obligation(
                    traits::Obligation::new(cause, trait_bound));
            }
        });
    }
}

impl<'ccx, 'tcx, 'v> Visitor<'v> for CheckTypeWellFormedVisitor<'ccx, 'tcx> {
    fn visit_item(&mut self, i: &'v ast::Item) {
        self.check_item_well_formed(self.ccx, i);
        visit::walk_item(self, i);
    }
}

pub struct BoundsChecker<'cx,'tcx:'cx> {
    fcx: &'cx FnCtxt<'cx,'tcx>,
    span: Span,
    scope_id: ast::NodeId,
    binding_count: uint,
    cache: Option<&'cx mut HashSet<ty::t>>,
}

impl<'cx,'tcx> BoundsChecker<'cx,'tcx> {
    pub fn new(fcx: &'cx FnCtxt<'cx,'tcx>,
               span: Span,
               scope_id: ast::NodeId,
               cache: Option<&'cx mut HashSet<ty::t>>)
               -> BoundsChecker<'cx,'tcx> {
        BoundsChecker { fcx: fcx, span: span, scope_id: scope_id,
                        cache: cache, binding_count: 0 }
    }

    pub fn check_trait_ref(&mut self, trait_ref: &ty::TraitRef) {
        /*!
         * Given a trait ref like `A : Trait<B>`, where `Trait` is
         * defined as (say):
         *
         *     trait Trait<B:OtherTrait> : Copy { ... }
         *
         * This routine will check that `B : OtherTrait` and `A :
         * Trait<B>`. It will also recursively check that the types
         * `A` and `B` are well-formed.
         *
         * Note that it does not (currently, at least)
         * check that `A : Copy` (that check is delegated to the point
         * where impl `A : Trait<B>` is implemented).
         */

        let trait_def = ty::lookup_trait_def(self.fcx.tcx(), trait_ref.def_id);

        self.fcx.add_obligations_for_parameters(
            traits::ObligationCause::new(
                self.span,
                traits::ItemObligation(trait_ref.def_id)),
            &trait_ref.substs,
            &trait_def.generics);

        for &ty in trait_ref.substs.types.iter() {
            self.check_traits_in_ty(ty);
        }
    }

    pub fn check_ty(&mut self, ty: ty::t) {
        ty.fold_with(self);
    }

    fn check_traits_in_ty(&mut self, ty: ty::t) {
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

    fn fold_ty(&mut self, t: ty::t) -> ty::t {
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

        match ty::get(t).sty{
            ty::ty_struct(type_id, ref substs) |
            ty::ty_enum(type_id, ref substs) => {
                let polytype = ty::lookup_item_type(self.fcx.tcx(), type_id);

                if self.binding_count == 0 {
                    self.fcx.add_obligations_for_parameters(
                        traits::ObligationCause::new(self.span,
                                                     traits::ItemObligation(type_id)),
                        substs,
                        &polytype.generics);
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
                    self.fcx.add_trait_obligations_for_generics(
                        traits::ObligationCause::new(self.span,
                                                     traits::ItemObligation(type_id)),
                        substs,
                        &polytype.generics);
                }

                self.fold_substs(substs);
            }
            ty::ty_bare_fn(ty::BareFnTy{sig: ref fn_sig, ..}) |
            ty::ty_closure(box ty::ClosureTy{sig: ref fn_sig, ..}) => {
                self.binding_count += 1;

                let (_, fn_sig) =
                    replace_late_bound_regions_in_fn_sig(
                        self.fcx.tcx(), fn_sig,
                        |br| ty::ReFree(ty::FreeRegion{scope_id: self.scope_id,
                                                       bound_region: br}));

                debug!("late-bound regions replaced: {}",
                       fn_sig.repr(self.tcx()));

                self.fold_sig(&fn_sig);

                self.binding_count -= 1;
            }
            ref sty => {
                self.fold_sty(sty);
            }
        }

        t // we're not folding to produce a new type, so just return `t` here
    }
}
