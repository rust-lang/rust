// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*

# Collect phase

The collect phase of type check has the job of visiting all items,
determining their type, and writing that type into the `tcx.tcache`
table.  Despite its name, this table does not really operate as a
*cache*, at least not for the types of items defined within the
current crate: we assume that after the collect phase, the types of
all local items will be present in the table.

Unlike most of the types that are present in Rust, the types computed
for each item are in fact polytypes.  In "layman's terms", this means
that they are generic types that may have type parameters (more
mathematically phrased, they are universally quantified over a set of
type parameters).  Polytypes are represented by an instance of
`ty::ty_param_bounds_and_ty`.  This combines the core type along with
a list of the bounds for each parameter.  Type parameters themselves
are represented as `ty_param()` instances.

*/

use core::prelude::*;

use metadata::csearch;
use middle::ty::InstantiatedTraitRef;
use middle::ty::{substs, ty_param_bounds_and_ty, ty_param_substs_and_ty};
use middle::ty;
use middle::typeck::astconv::{AstConv, ty_of_arg};
use middle::typeck::astconv::{ast_ty_to_ty};
use middle::typeck::astconv;
use middle::typeck::infer;
use middle::typeck::rscope::*;
use middle::typeck::rscope;
use middle::typeck::{CrateCtxt, lookup_def_tcx, no_params, write_ty_to_tcx,
                     write_tpt_to_tcx};
use util::common::{indenter, pluralize};
use util::ppaux;

use core::vec;
use syntax::abi::AbiSet;
use syntax::ast::{RegionTyParamBound, TraitTyParamBound};
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::{local_def, split_trait_methods};
use syntax::ast_util::{trait_method_to_ty_method};
use syntax::ast_util;
use syntax::codemap::span;
use syntax::codemap;
use syntax::print::pprust::path_to_str;
use syntax::visit;
use syntax::opt_vec::OptVec;
use syntax::opt_vec;

pub fn collect_item_types(ccx: @mut CrateCtxt, crate: @ast::crate) {

    // FIXME (#2592): hooking into the "intrinsic" root module is crude.
    // There ought to be a better approach. Attributes?

    for crate.node.module.items.each |crate_item| {
        if crate_item.ident
            == ::syntax::parse::token::special_idents::intrinsic {

            match crate_item.node {
              ast::item_mod(ref m) => {
                for m.items.each |intrinsic_item| {
                    let def_id = ast::def_id { crate: ast::local_crate,
                                               node: intrinsic_item.id };
                    let substs = substs {
                        self_r: None,
                        self_ty: None,
                        tps: ~[]
                    };

                    match intrinsic_item.node {
                      ast::item_trait(*) => {
                        let ty = ty::mk_trait(ccx.tcx,
                                              def_id,
                                              substs,
                                              ty::BareTraitStore);
                        ccx.tcx.intrinsic_defs.insert
                            (intrinsic_item.ident, (def_id, ty));
                      }

                      ast::item_enum(*) => {
                        let ty = ty::mk_enum(ccx.tcx, def_id, substs);
                        ccx.tcx.intrinsic_defs.insert
                            (intrinsic_item.ident, (def_id, ty));
                      }

                      ast::item_struct(*) => {
                        let ty = ty::mk_struct(ccx.tcx, def_id, substs);
                        ccx.tcx.intrinsic_defs.insert
                            (intrinsic_item.ident, (def_id, ty));
                      }

                      _ => {}
                    }
                }
              }
              _ => { }
            }
            break;
        }
    }

    visit::visit_crate(
        *crate, (),
        visit::mk_simple_visitor(@visit::SimpleVisitor {
            visit_item: |a| convert(ccx, a),
            visit_foreign_item: |a|convert_foreign(ccx, a),
            .. *visit::default_simple_visitor()
        }));
}

impl CrateCtxt {
    fn to_ty<RS:region_scope + Copy + Durable>(
        &self, rs: &RS, ast_ty: @ast::Ty) -> ty::t
    {
        ast_ty_to_ty(self, rs, ast_ty)
    }
}

impl AstConv for CrateCtxt {
    fn tcx(&self) -> ty::ctxt { self.tcx }

    fn get_item_ty(&self, id: ast::def_id) -> ty::ty_param_bounds_and_ty {
        if id.crate != ast::local_crate {
            csearch::get_type(self.tcx, id)
        } else {
            match self.tcx.items.find(&id.node) {
              Some(&ast_map::node_item(item, _)) => {
                ty_of_item(self, item)
              }
              Some(&ast_map::node_foreign_item(foreign_item, _, _, _)) => {
                ty_of_foreign_item(self, foreign_item)
              }
              ref x => {
                self.tcx.sess.bug(fmt!("unexpected sort of item \
                                        in get_item_ty(): %?", (*x)));
              }
            }
        }
    }

    fn ty_infer(&self, span: span) -> ty::t {
        self.tcx.sess.span_bug(span,
                               ~"found `ty_infer` in unexpected place");
    }
}

pub fn get_enum_variant_types(ccx: &CrateCtxt,
                              enum_ty: ty::t,
                              variants: &[ast::variant],
                              generics: &ast::Generics,
                              rp: Option<ty::region_variance>) {
    let tcx = ccx.tcx;

    // Create a set of parameter types shared among all the variants.
    for variants.each |variant| {
        let region_parameterization =
            RegionParameterization::from_variance_and_generics(rp, generics);

        // Nullary enum constructors get turned into constants; n-ary enum
        // constructors get turned into functions.
        let result_ty;
        match variant.node.kind {
            ast::tuple_variant_kind(ref args) if args.len() > 0 => {
                let rs = type_rscope(region_parameterization);
                let input_tys = args.map(|va| ccx.to_ty(&rs, va.ty));
                result_ty = Some(ty::mk_ctor_fn(tcx, input_tys, enum_ty));
            }

            ast::tuple_variant_kind(_) => {
                result_ty = Some(enum_ty);
            }

            ast::struct_variant_kind(struct_def) => {
                let tpt = ty_param_bounds_and_ty {
                    bounds: ty_param_bounds(ccx, generics),
                    region_param: rp,
                    ty: enum_ty
                };

                convert_struct(ccx,
                               rp,
                               struct_def,
                               generics,
                               tpt,
                               variant.node.id);

                let input_tys = struct_def.fields.map(
                    |f| ty::node_id_to_type(ccx.tcx, f.node.id));
                result_ty = Some(ty::mk_ctor_fn(tcx, input_tys, enum_ty));
            }
        };

        match result_ty {
            None => {}
            Some(result_ty) => {
                let tpt = ty_param_bounds_and_ty {
                    bounds: ty_param_bounds(ccx, generics),
                    region_param: rp,
                    ty: result_ty
                };
                tcx.tcache.insert(local_def(variant.node.id), tpt);
                write_ty_to_tcx(tcx, variant.node.id, result_ty);
            }
        }
    }
}

pub fn ensure_trait_methods(ccx: &CrateCtxt,
                            id: ast::node_id,
                            trait_ty: ty::t) {
    fn store_methods<T>(ccx: &CrateCtxt,
                        id: ast::node_id,
                        stuff: &[T],
                        f: &fn(v: &T) -> ty::method) {
        ty::store_trait_methods(ccx.tcx, id, @stuff.map(f));
    }

    fn make_static_method_ty(ccx: &CrateCtxt,
                             am: &ast::ty_method,
                             rp: Option<ty::region_variance>,
                             m: ty::method,
                             // Take this as an argument b/c we may check
                             // the impl before the trait.
                             trait_ty: ty::t,
                             trait_bounds: @~[ty::param_bounds]) {
        // We need to create a typaram that replaces self. This param goes
        // *in between* the typarams from the trait and those from the
        // method (since its bound can depend on the trait? or
        // something like that).

        // build up a subst that shifts all of the parameters over
        // by one and substitute in a new type param for self

        let dummy_defid = ast::def_id {crate: 0, node: 0};

        let non_shifted_trait_tps = do vec::from_fn(trait_bounds.len()) |i| {
            ty::mk_param(ccx.tcx, i, dummy_defid)
        };
        let self_param = ty::mk_param(ccx.tcx, trait_bounds.len(),
                                      dummy_defid);
        let shifted_method_tps = do vec::from_fn(m.tps.len()) |i| {
            ty::mk_param(ccx.tcx, i + 1, dummy_defid)
        };

        let substs = substs {
            self_r: None,
            self_ty: Some(self_param),
            tps: non_shifted_trait_tps + shifted_method_tps
        };
        let ty = ty::subst(ccx.tcx,
                           &substs,
                           ty::mk_bare_fn(ccx.tcx, copy m.fty));
        let bounds = @(*trait_bounds + ~[@~[ty::bound_trait(trait_ty)]]
                       + *m.tps);
        ccx.tcx.tcache.insert(local_def(am.id),
                              ty_param_bounds_and_ty {
                                bounds: bounds,
                                region_param: rp,
                                ty: ty});
    }


    let tcx = ccx.tcx;
    let region_paramd = tcx.region_paramd_items.find(&id).map_consume(|x| *x);
    match *tcx.items.get(&id) {
      ast_map::node_item(@ast::item {
                node: ast::item_trait(ref generics, _, ref ms),
                _
            }, _) => {
        store_methods::<ast::trait_method>(ccx, id, *ms, |m| {
            let def_id;
            match *m {
                ast::required(ref ty_method) => {
                    def_id = local_def((*ty_method).id)
                }
                ast::provided(method) => def_id = local_def(method.id)
            }

            let trait_bounds = ty_param_bounds(ccx, generics);
            let ty_m = trait_method_to_ty_method(m);
            let method_ty = ty_of_ty_method(
                ccx,
                &ty_m,
                region_paramd,
                def_id,
                generics
            );
            if ty_m.self_ty.node == ast::sty_static {
                make_static_method_ty(ccx, &ty_m, region_paramd,
                                      method_ty, trait_ty,
                                      trait_bounds);
            }
            method_ty
        });
      }
      _ => { /* Ignore things that aren't traits */ }
    }
}

pub fn ensure_supertraits(ccx: &CrateCtxt,
                          id: ast::node_id,
                          sp: codemap::span,
                          rp: Option<ty::region_variance>,
                          trait_refs: &[@ast::trait_ref],
                          generics: &ast::Generics) {
    let tcx = ccx.tcx;
    if tcx.supertraits.contains_key(&local_def(id)) { return; }

    let mut instantiated = ~[];
    for trait_refs.each |trait_ref| {
        let (did, tpt) = instantiate_trait_ref(ccx, *trait_ref, rp, generics);
        if instantiated.any(|other_trait: &InstantiatedTraitRef|
                            { other_trait.def_id == did }) {
            // This means a trait inherited from the same supertrait more
            // than once.
            tcx.sess.span_err(sp, ~"Duplicate supertrait in trait \
                                     declaration");
            return;
        }
        instantiated.push(InstantiatedTraitRef { def_id: did, tpt: tpt });
    }
    tcx.supertraits.insert(local_def(id), @instantiated);
}

/**
 * Checks that a method from an impl/class conforms to the signature of
 * the same method as declared in the trait.
 *
 * # Parameters
 *
 * - impl_tps: the type params declared on the impl itself (not the method!)
 * - cm: info about the method we are checking
 * - trait_m: the method in the trait
 * - trait_substs: the substitutions used on the type of the trait
 * - self_ty: the self type of the impl
 */
pub fn compare_impl_method(tcx: ty::ctxt,
                           impl_tps: uint,
                           cm: &ConvertedMethod,
                           trait_m: &ty::method,
                           trait_substs: &ty::substs,
                           self_ty: ty::t) {
    debug!("compare_impl_method()");
    let _indenter = indenter();

    let impl_m = &cm.mty;

    // FIXME(#2687)---this check is too strict.  For example, a trait
    // method with self type `&self` or `&mut self` should be
    // implementable by an `&const self` method (the impl assumes less
    // than the trait provides).
    if impl_m.self_ty != trait_m.self_ty {
        if impl_m.self_ty == ast::sty_static {
            // Needs to be a fatal error because otherwise,
            // method::transform_self_type_for_method ICEs
            tcx.sess.span_fatal(cm.span,
                 fmt!("method `%s` is declared as \
                       static in its impl, but not in \
                       its trait", *tcx.sess.str_of(impl_m.ident)));
        }
        else if trait_m.self_ty == ast::sty_static {
            tcx.sess.span_fatal(cm.span,
                 fmt!("method `%s` is declared as \
                       static in its trait, but not in \
                       its impl", *tcx.sess.str_of(impl_m.ident)));
        }
        else {
            tcx.sess.span_err(
                cm.span,
                fmt!("method `%s`'s self type does \
                      not match the trait method's \
                      self type", *tcx.sess.str_of(impl_m.ident)));
        }
    }

    if impl_m.tps.len() != trait_m.tps.len() {
        tcx.sess.span_err(
            cm.span,
            fmt!("method `%s` has %u type %s, but its trait \
                  declaration has %u type %s",
                 *tcx.sess.str_of(trait_m.ident), impl_m.tps.len(),
                 pluralize(impl_m.tps.len(), ~"parameter"),
                 trait_m.tps.len(),
                 pluralize(trait_m.tps.len(), ~"parameter")));
        return;
    }

    if vec::len(impl_m.fty.sig.inputs) != vec::len(trait_m.fty.sig.inputs) {
        tcx.sess.span_err(
            cm.span,
            fmt!("method `%s` has %u parameters \
                  but the trait has %u",
                 *tcx.sess.str_of(trait_m.ident),
                 vec::len(impl_m.fty.sig.inputs),
                 vec::len(trait_m.fty.sig.inputs)));
        return;
    }

    // FIXME(#2687)---we should be checking that the bounds of the
    // trait imply the bounds of the subtype, but it appears
    // we are...not checking this.
    for trait_m.tps.eachi() |i, trait_param_bounds| {
        // For each of the corresponding impl ty param's bounds...
        let impl_param_bounds = impl_m.tps[i];
        // Make sure the bounds lists have the same length
        // Would be nice to use the ty param names in the error message,
        // but we don't have easy access to them here
        if impl_param_bounds.len() != trait_param_bounds.len() {
           tcx.sess.span_err(
               cm.span,
               fmt!("in method `%s`, \
                     type parameter %u has %u %s, but the same type \
                     parameter in its trait declaration has %u %s",
                    *tcx.sess.str_of(trait_m.ident),
                    i, impl_param_bounds.len(),
                    pluralize(impl_param_bounds.len(), ~"bound"),
                    trait_param_bounds.len(),
                    pluralize(trait_param_bounds.len(), ~"bound")));
           return;
        }
    }

    // Replace any references to the self region in the self type with
    // a free region.  So, for example, if the impl type is
    // "&'self str", then this would replace the self type with a free
    // region `self`.
    let dummy_self_r = ty::re_free(cm.body_id, ty::br_self);
    let self_ty = replace_bound_self(tcx, self_ty, dummy_self_r);

    // Perform substitutions so that the trait/impl methods are expressed
    // in terms of the same set of type/region parameters:
    // - replace trait type parameters with those from `trait_substs`,
    //   except with any reference to bound self replaced with `dummy_self_r`
    // - replace method parameters on the trait with fresh, dummy parameters
    //   that correspond to the parameters we will find on the impl
    // - replace self region with a fresh, dummy region
    let impl_fty = {
        let impl_fty = ty::mk_bare_fn(tcx, copy impl_m.fty);
        debug!("impl_fty (pre-subst): %s", ppaux::ty_to_str(tcx, impl_fty));
        replace_bound_self(tcx, impl_fty, dummy_self_r)
    };
    debug!("impl_fty: %s", ppaux::ty_to_str(tcx, impl_fty));
    let trait_fty = {
        let dummy_tps = do vec::from_fn((*trait_m.tps).len()) |i| {
            // hack: we don't know the def id of the impl tp, but it
            // is not important for unification
            ty::mk_param(tcx, i + impl_tps, ast::def_id {crate: 0, node: 0})
        };
        let trait_tps = trait_substs.tps.map(
            |t| replace_bound_self(tcx, *t, dummy_self_r));
        let substs = substs {
            self_r: Some(dummy_self_r),
            self_ty: Some(self_ty),
            tps: vec::append(trait_tps, dummy_tps)
        };
        let trait_fty = ty::mk_bare_fn(tcx, copy trait_m.fty);
        debug!("trait_fty (pre-subst): %s", ppaux::ty_to_str(tcx, trait_fty));
        ty::subst(tcx, &substs, trait_fty)
    };

    let infcx = infer::new_infer_ctxt(tcx);
    match infer::mk_subty(infcx, false, cm.span, impl_fty, trait_fty) {
        result::Ok(()) => {}
        result::Err(ref terr) => {
            tcx.sess.span_err(
                cm.span,
                fmt!("method `%s` has an incompatible type: %s",
                     *tcx.sess.str_of(trait_m.ident),
                     ty::type_err_to_str(tcx, terr)));
            ty::note_and_explain_type_err(tcx, terr);
        }
    }
    return;

    // Replaces bound references to the self region with `with_r`.
    fn replace_bound_self(tcx: ty::ctxt, ty: ty::t,
                          with_r: ty::Region) -> ty::t {
        do ty::fold_regions(tcx, ty) |r, _in_fn| {
            if r == ty::re_bound(ty::br_self) {with_r} else {r}
        }
    }
}

pub fn check_methods_against_trait(ccx: &CrateCtxt,
                                   generics: &ast::Generics,
                                   rp: Option<ty::region_variance>,
                                   selfty: ty::t,
                                   a_trait_ty: @ast::trait_ref,
                                   impl_ms: &[ConvertedMethod]) {

    let tcx = ccx.tcx;
    let (did, tpt) = instantiate_trait_ref(ccx, a_trait_ty, rp, generics);

    if did.crate == ast::local_crate {
        // NB: This is subtle. We need to do this on the type of the trait
        // item *itself*, not on the type that includes the parameter
        // substitutions provided by the programmer at this particular
        // trait ref. Otherwise, we will potentially overwrite the types of
        // the methods within the trait with bogus results. (See issue #3903.)

        match tcx.items.find(&did.node) {
            Some(&ast_map::node_item(item, _)) => {
                let tpt = ty_of_item(ccx, item);
                ensure_trait_methods(ccx, did.node, tpt.ty);
            }
            _ => {
                tcx.sess.bug(~"trait ref didn't resolve to trait");
            }
        }
    }

    // Check that each method we impl is a method on the trait
    // Trait methods we don't implement must be default methods, but if not
    // we'll catch it in coherence
    let trait_ms = ty::trait_methods(tcx, did);
    for impl_ms.each |impl_m| {
        match trait_ms.find(|trait_m| trait_m.ident == impl_m.mty.ident) {
            Some(ref trait_m) => {
                let num_impl_tps = generics.ty_params.len();
                compare_impl_method(
                    ccx.tcx, num_impl_tps, impl_m, trait_m,
                    &tpt.substs, selfty);
            }
            None => {
                // This method is not part of the trait
                tcx.sess.span_err(
                    impl_m.span,
                    fmt!("method `%s` is not a member of trait `%s`",
                         *tcx.sess.str_of(impl_m.mty.ident),
                         path_to_str(a_trait_ty.path, tcx.sess.intr())));
            }
        }
    }
} // fn

pub fn convert_field(ccx: &CrateCtxt,
                     rp: Option<ty::region_variance>,
                     bounds: @~[ty::param_bounds],
                     v: @ast::struct_field,
                     generics: &ast::Generics) {
    let region_parameterization =
        RegionParameterization::from_variance_and_generics(rp, generics);
    let tt = ccx.to_ty(&type_rscope(region_parameterization), v.node.ty);
    write_ty_to_tcx(ccx.tcx, v.node.id, tt);
    /* add the field to the tcache */
    ccx.tcx.tcache.insert(local_def(v.node.id),
                          ty::ty_param_bounds_and_ty {
                              bounds: bounds,
                              region_param: rp,
                              ty: tt
                          });
}

pub struct ConvertedMethod {
    mty: ty::method,
    id: ast::node_id,
    span: span,
    body_id: ast::node_id
}

pub fn convert_methods(ccx: &CrateCtxt,
                       ms: &[@ast::method],
                       rp: Option<ty::region_variance>,
                       rcvr_bounds: @~[ty::param_bounds],
                       rcvr_generics: &ast::Generics)
                    -> ~[ConvertedMethod] {

    let tcx = ccx.tcx;
    do vec::map(ms) |m| {
        let bounds = ty_param_bounds(ccx, &m.generics);
        let mty = ty_of_method(ccx, *m, rp, rcvr_generics, &m.generics);
        let fty = ty::mk_bare_fn(tcx, copy mty.fty);
        tcx.tcache.insert(
            local_def(m.id),

            // n.b.: the type of a method is parameterized by both
            // the tps on the receiver and those on the method itself
            ty_param_bounds_and_ty {
                bounds: @(vec::append(/*bad*/copy *rcvr_bounds, *bounds)),
                region_param: rp,
                ty: fty
            });
        write_ty_to_tcx(tcx, m.id, fty);
        ConvertedMethod {mty: mty, id: m.id,
                         span: m.span, body_id: m.body.node.id}
    }
}

pub fn ensure_no_ty_param_bounds(ccx: &CrateCtxt,
                                 span: span,
                                 generics: &ast::Generics,
                                 thing: &'static str) {
    for generics.ty_params.each |ty_param| {
        if ty_param.bounds.len() > 0 {
            ccx.tcx.sess.span_err(
                span,
                fmt!("trait bounds are not allowed in %s definitions",
                     thing));
        }
    }
}

pub fn convert(ccx: &CrateCtxt, it: @ast::item) {
    let tcx = ccx.tcx;
    let rp = tcx.region_paramd_items.find(&it.id).map_consume(|x| *x);
    debug!("convert: item %s with id %d rp %?",
           *tcx.sess.str_of(it.ident), it.id, rp);
    match it.node {
      // These don't define types.
      ast::item_foreign_mod(_) | ast::item_mod(_) => {}
      ast::item_enum(ref enum_definition, ref generics) => {
        ensure_no_ty_param_bounds(ccx, it.span, generics, "enumeration");
        let tpt = ty_of_item(ccx, it);
        write_ty_to_tcx(tcx, it.id, tpt.ty);
        get_enum_variant_types(ccx,
                               tpt.ty,
                               enum_definition.variants,
                               generics,
                               rp);
      }
      ast::item_impl(ref generics, trait_ref, selfty, ref ms) => {
        let i_bounds = ty_param_bounds(ccx, generics);
        let region_parameterization =
            RegionParameterization::from_variance_and_generics(rp, generics);
        let selfty = ccx.to_ty(&type_rscope(region_parameterization), selfty);
        write_ty_to_tcx(tcx, it.id, selfty);
        tcx.tcache.insert(local_def(it.id),
                          ty_param_bounds_and_ty {
                            bounds: i_bounds,
                            region_param: rp,
                            ty: selfty});

        // XXX: Bad copy of `ms` below.
        let cms = convert_methods(ccx, *ms, rp, i_bounds, generics);
        for trait_ref.each |t| {
            check_methods_against_trait(ccx, generics, rp, selfty, *t, cms);
        }
      }
      ast::item_trait(ref generics, ref supertraits, ref trait_methods) => {
        let tpt = ty_of_item(ccx, it);
        debug!("item_trait(it.id=%d, tpt.ty=%s)",
               it.id, ppaux::ty_to_str(tcx, tpt.ty));
        write_ty_to_tcx(tcx, it.id, tpt.ty);
        ensure_trait_methods(ccx, it.id, tpt.ty);
        ensure_supertraits(ccx, it.id, it.span, rp, *supertraits, generics);

        let (_, provided_methods) =
            split_trait_methods(*trait_methods);
        let (bounds, _) = mk_substs(ccx, generics, rp);
        let _ = convert_methods(ccx, provided_methods, rp, bounds, generics);
      }
      ast::item_struct(struct_def, ref generics) => {
        ensure_no_ty_param_bounds(ccx, it.span, generics, "structure");

        // Write the class type
        let tpt = ty_of_item(ccx, it);
        write_ty_to_tcx(tcx, it.id, tpt.ty);
        tcx.tcache.insert(local_def(it.id), tpt);

        convert_struct(ccx, rp, struct_def, generics, tpt, it.id);
      }
      ast::item_ty(_, ref generics) => {
        ensure_no_ty_param_bounds(ccx, it.span, generics, "type");
        let tpt = ty_of_item(ccx, it);
        write_ty_to_tcx(tcx, it.id, tpt.ty);
      }
      _ => {
        // This call populates the type cache with the converted type
        // of the item in passing. All we have to do here is to write
        // it into the node type table.
        let tpt = ty_of_item(ccx, it);
        write_ty_to_tcx(tcx, it.id, tpt.ty);
      }
    }
}

pub fn convert_struct(ccx: &CrateCtxt,
                      rp: Option<ty::region_variance>,
                      struct_def: @ast::struct_def,
                      generics: &ast::Generics,
                      tpt: ty::ty_param_bounds_and_ty,
                      id: ast::node_id) {
    let tcx = ccx.tcx;

    for struct_def.dtor.each |dtor| {
        let region_parameterization =
            RegionParameterization::from_variance_and_generics(rp, generics);

        // Write the dtor type
        let t_dtor = ty::mk_bare_fn(
            tcx,
            astconv::ty_of_bare_fn(
                ccx,
                &type_rscope(region_parameterization),
                ast::impure_fn,
                AbiSet::Rust(),
                &opt_vec::Empty,
                &ast_util::dtor_dec()));
        write_ty_to_tcx(tcx, dtor.node.id, t_dtor);
        tcx.tcache.insert(local_def(dtor.node.id),
                          ty_param_bounds_and_ty {
                            bounds: tpt.bounds,
                            region_param: rp,
                            ty: t_dtor});
    };

    // Write the type of each of the members
    for struct_def.fields.each |f| {
       convert_field(ccx, rp, tpt.bounds, *f, generics);
    }
    let (_, substs) = mk_substs(ccx, generics, rp);
    let selfty = ty::mk_struct(tcx, local_def(id), substs);

    // If this struct is enum-like or tuple-like, create the type of its
    // constructor.
    match struct_def.ctor_id {
        None => {}
        Some(ctor_id) => {
            if struct_def.fields.len() == 0 {
                // Enum-like.
                write_ty_to_tcx(tcx, ctor_id, selfty);
                tcx.tcache.insert(local_def(ctor_id), tpt);
            } else if struct_def.fields[0].node.kind == ast::unnamed_field {
                // Tuple-like.
                let inputs =
                    struct_def.fields.map(
                        |field| ccx.tcx.tcache.get(
                            &local_def(field.node.id)).ty);
                let ctor_fn_ty = ty::mk_ctor_fn(tcx, inputs, selfty);
                write_ty_to_tcx(tcx, ctor_id, ctor_fn_ty);
                tcx.tcache.insert(local_def(ctor_id), ty_param_bounds_and_ty {
                    bounds: tpt.bounds,
                    region_param: tpt.region_param,
                    ty: ctor_fn_ty
                });
            }
        }
    }
}

pub fn convert_foreign(ccx: &CrateCtxt, i: @ast::foreign_item) {
    // As above, this call populates the type table with the converted
    // type of the foreign item. We simply write it into the node type
    // table.
    let tpt = ty_of_foreign_item(ccx, i);
    write_ty_to_tcx(ccx.tcx, i.id, tpt.ty);
    ccx.tcx.tcache.insert(local_def(i.id), tpt);
}

pub fn ty_of_method(ccx: &CrateCtxt,
                    m: @ast::method,
                    rp: Option<ty::region_variance>,
                    rcvr_generics: &ast::Generics,
                    method_generics: &ast::Generics)
                 -> ty::method {
    let rscope = MethodRscope::new(m.self_ty.node,
                                   rp,
                                   rcvr_generics);
    ty::method {
        ident: m.ident,
        tps: ty_param_bounds(ccx, &m.generics),
        fty: astconv::ty_of_bare_fn(ccx,
                                    &rscope,
                                    m.purity,
                                    AbiSet::Rust(),
                                    &method_generics.lifetimes,
                                    &m.decl),
        self_ty: m.self_ty.node,
        vis: m.vis,
        def_id: local_def(m.id)
    }
}

pub fn ty_of_ty_method(self: &CrateCtxt,
                       m: &ast::ty_method,
                       rp: Option<ty::region_variance>,
                       id: ast::def_id,
                       generics: &ast::Generics)
                    -> ty::method {
    let rscope = MethodRscope::new(m.self_ty.node, rp, generics);
    ty::method {
        ident: m.ident,
        tps: ty_param_bounds(self, &m.generics),
        fty: astconv::ty_of_bare_fn(self,
                                    &rscope,
                                    m.purity,
                                    AbiSet::Rust(),
                                    &m.generics.lifetimes,
                                    &m.decl),
        // assume public, because this is only invoked on trait methods
        self_ty: m.self_ty.node,
        vis: ast::public,
        def_id: id
    }
}

/*
  Instantiates the path for the given trait reference, assuming that
  it's bound to a valid trait type. Returns the def_id for the defining
  trait. Fails if the type is a type other than an trait type.
 */
pub fn instantiate_trait_ref(ccx: &CrateCtxt,
                             t: @ast::trait_ref,
                             rp: Option<ty::region_variance>,
                             generics: &ast::Generics)
    -> (ast::def_id, ty_param_substs_and_ty) {

    let sp = t.path.span, err = ~"can only implement trait types",
        sess = ccx.tcx.sess;

    let rp = RegionParameterization::from_variance_and_generics(rp, generics);

    let rscope = type_rscope(rp);

    match lookup_def_tcx(ccx.tcx, t.path.span, t.ref_id) {
      ast::def_ty(t_id) => {
        let tpt = astconv::ast_path_to_ty(ccx, &rscope, t_id, t.path);

        write_tpt_to_tcx(ccx.tcx, t.ref_id, &tpt);

        match ty::get(tpt.ty).sty {
           ty::ty_trait(*) => {
              (t_id, tpt)
           }
           _ => sess.span_fatal(sp, err),
        }
      }
      _ => sess.span_fatal(sp, err)
    }
}

pub fn ty_of_item(ccx: &CrateCtxt, it: @ast::item)
               -> ty::ty_param_bounds_and_ty {
    let def_id = local_def(it.id);
    let tcx = ccx.tcx;
    match tcx.tcache.find(&def_id) {
      Some(&tpt) => return tpt,
      _ => {}
    }
    let rp = tcx.region_paramd_items.find(&it.id).map_consume(|x| *x);
    match it.node {
      ast::item_const(t, _) => {
        let typ = ccx.to_ty(&empty_rscope, t);
        let tpt = no_params(typ);
        tcx.tcache.insert(local_def(it.id), tpt);
        return tpt;
      }
      ast::item_fn(ref decl, purity, _, ref generics, _) => {
        let bounds = ty_param_bounds(ccx, generics);
        let tofd = astconv::ty_of_bare_fn(ccx,
                                          &empty_rscope,
                                          purity,
                                          AbiSet::Rust(),
                                          &generics.lifetimes,
                                          decl);
        let tpt = ty_param_bounds_and_ty {
            bounds: bounds,
            region_param: None,
            ty: ty::mk_bare_fn(ccx.tcx, tofd)
        };
        debug!("type of %s (id %d) is %s",
               *tcx.sess.str_of(it.ident),
               it.id,
               ppaux::ty_to_str(tcx, tpt.ty));
        ccx.tcx.tcache.insert(local_def(it.id), tpt);
        return tpt;
      }
      ast::item_ty(t, ref generics) => {
        match tcx.tcache.find(&local_def(it.id)) {
          Some(&tpt) => return tpt,
          None => { }
        }

        let rp = tcx.region_paramd_items.find(&it.id).map_consume(|x| *x);
        let region_parameterization =
            RegionParameterization::from_variance_and_generics(rp, generics);
        let tpt = {
            let ty = {
                let t0 = ccx.to_ty(&type_rscope(region_parameterization), t);
                // Do not associate a def id with a named, parameterized type
                // like "foo<X>".  This is because otherwise ty_to_str will
                // print the name as merely "foo", as it has no way to
                // reconstruct the value of X.
                if generics.is_parameterized() { t0 } else {
                    ty::mk_with_id(tcx, t0, def_id)
                }
            };
            ty_param_bounds_and_ty {
                bounds: ty_param_bounds(ccx, generics),
                region_param: rp,
                ty: ty
            }
        };

        tcx.tcache.insert(local_def(it.id), tpt);
        return tpt;
      }
      ast::item_enum(_, ref generics) => {
        // Create a new generic polytype.
        let (bounds, substs) = mk_substs(ccx, generics, rp);
        let t = ty::mk_enum(tcx, local_def(it.id), substs);
        let tpt = ty_param_bounds_and_ty {
            bounds: bounds,
            region_param: rp,
            ty: t
        };
        tcx.tcache.insert(local_def(it.id), tpt);
        return tpt;
      }
      ast::item_trait(ref generics, _, _) => {
        let (bounds, substs) = mk_substs(ccx, generics, rp);
        let t = ty::mk_trait(tcx,
                             local_def(it.id),
                             substs,
                             ty::BareTraitStore);
        let tpt = ty_param_bounds_and_ty {
            bounds: bounds,
            region_param: rp,
            ty: t
        };
        tcx.tcache.insert(local_def(it.id), tpt);
        return tpt;
      }
      ast::item_struct(_, ref generics) => {
          let (bounds, substs) = mk_substs(ccx, generics, rp);
          let t = ty::mk_struct(tcx, local_def(it.id), substs);
          let tpt = ty_param_bounds_and_ty {
            bounds: bounds,
            region_param: rp,
            ty: t
          };
          tcx.tcache.insert(local_def(it.id), tpt);
          return tpt;
      }
      ast::item_impl(*) | ast::item_mod(_) |
      ast::item_foreign_mod(_) => fail!(),
      ast::item_mac(*) => fail!(~"item macros unimplemented")
    }
}

pub fn ty_of_foreign_item(ccx: &CrateCtxt, it: @ast::foreign_item)
    -> ty::ty_param_bounds_and_ty {
    match it.node {
        ast::foreign_item_fn(ref fn_decl, _, ref generics) => {
            ty_of_foreign_fn_decl(ccx,
                                  fn_decl,
                                  local_def(it.id),
                                  generics)
        }
        ast::foreign_item_const(t) => {
            ty::ty_param_bounds_and_ty {
                bounds: @~[],
                region_param: None,
                ty: ast_ty_to_ty(ccx, &empty_rscope, t)
            }
        }
    }
}

// Translate the AST's notion of ty param bounds (which are an enum consisting
// of a newtyped Ty or a region) to ty's notion of ty param bounds, which can
// either be user-defined traits, or one of the four built-in traits (formerly
// known as kinds): Const, Copy, Durable, and Send.
pub fn compute_bounds(ccx: &CrateCtxt,
                      ast_bounds: @OptVec<ast::TyParamBound>)
                   -> ty::param_bounds {
    @ast_bounds.flat_map_to_vec(|b| {
        match b {
            &TraitTyParamBound(b) => {
                let li = &ccx.tcx.lang_items;
                let ity = ast_ty_to_ty(ccx, &empty_rscope, b);
                match ty::get(ity).sty {
                    ty::ty_trait(did, _, _) => {
                        if did == li.owned_trait() {
                            ~[ty::bound_owned]
                        } else if did == li.copy_trait() {
                            ~[ty::bound_copy]
                        } else if did == li.const_trait() {
                            ~[ty::bound_const]
                        } else if did == li.durable_trait() {
                            ~[ty::bound_durable]
                        } else {
                            // Must be a user-defined trait
                            ~[ty::bound_trait(ity)]
                        }
                    }
                    _ => {
                        ccx.tcx.sess.span_err(
                            (*b).span, ~"type parameter bounds must be \
                                         trait types");
                        ~[]
                    }
                }
            }
            &RegionTyParamBound => ~[ty::bound_durable]
        }
    })
}

pub fn ty_param_bounds(ccx: &CrateCtxt,
                       generics: &ast::Generics)
                    -> @~[ty::param_bounds] {
    @do generics.ty_params.map_to_vec |param| {
        match ccx.tcx.ty_param_bounds.find(&param.id) {
          Some(&bs) => bs,
          None => {
            let bounds = compute_bounds(ccx, param.bounds);
            ccx.tcx.ty_param_bounds.insert(param.id, bounds);
            bounds
          }
        }
    }
}

pub fn ty_of_foreign_fn_decl(ccx: &CrateCtxt,
                             decl: &ast::fn_decl,
                             def_id: ast::def_id,
                             generics: &ast::Generics)
                          -> ty::ty_param_bounds_and_ty {
    let bounds = ty_param_bounds(ccx, generics);
    let region_param_names = RegionParamNames::from_generics(generics);
    let rb = in_binding_rscope(&empty_rscope, region_param_names);
    let input_tys = decl.inputs.map(|a| ty_of_arg(ccx, &rb, *a, None) );
    let output_ty = ast_ty_to_ty(ccx, &rb, decl.output);

    let t_fn = ty::mk_bare_fn(
        ccx.tcx,
        ty::BareFnTy {
            abis: AbiSet::Rust(),
            purity: ast::unsafe_fn,
            sig: ty::FnSig {bound_lifetime_names: opt_vec::Empty,
                            inputs: input_tys,
                            output: output_ty}
        });
    let tpt = ty_param_bounds_and_ty {
        bounds: bounds,
        region_param: None,
        ty: t_fn
    };
    ccx.tcx.tcache.insert(def_id, tpt);
    return tpt;
}

pub fn mk_generics(ccx: &CrateCtxt, generics: &ast::Generics)
    -> (@~[ty::param_bounds], ~[ty::t])
{
    let mut i = 0u;
    let bounds = ty_param_bounds(ccx, generics);
    (bounds,
     generics.ty_params.map_to_vec(|atp| {
         let t = ty::mk_param(ccx.tcx, i, local_def(atp.id));
         i += 1u;
         t
     }))
}

pub fn mk_substs(ccx: &CrateCtxt,
                 generics: &ast::Generics,
                 rp: Option<ty::region_variance>)
              -> (@~[ty::param_bounds], ty::substs)
{
    let (bounds, params) = mk_generics(ccx, generics);
    let self_r = rscope::bound_self_region(rp);
    (bounds, substs { self_r: self_r, self_ty: None, tps: params })
}
