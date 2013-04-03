// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use middle::resolve::Impl;
use middle::ty::{param_ty, substs};
use middle::ty;
use middle::typeck::check::{FnCtxt, impl_self_ty};
use middle::typeck::check::{structurally_resolved_type};
use middle::typeck::infer::fixup_err_to_str;
use middle::typeck::infer::{resolve_and_force_all_but_regions, resolve_type};
use middle::typeck::infer;
use middle::typeck::{CrateCtxt, vtable_origin, vtable_param, vtable_res};
use middle::typeck::vtable_static;
use util::common::indenter;
use util::ppaux::tys_to_str;
use util::ppaux;

use core::result::{Ok, Err};
use core::result;
use core::uint;
use core::vec;
use core::hashmap::HashSet;
use syntax::ast;
use syntax::ast_util;
use syntax::codemap::span;
use syntax::print::pprust::expr_to_str;
use syntax::visit;

// vtable resolution looks for places where trait bounds are
// subsituted in and figures out which vtable is used. There is some
// extra complication thrown in to support early "opportunistic"
// vtable resolution. This is a hacky mechanism that is invoked while
// typechecking function calls (after typechecking non-closure
// arguments and before typechecking closure arguments) in the hope of
// solving for the trait parameters from the impl. (For example,
// determining that if a parameter bounded by BaseIter<A> is
// instantiated with Option<int>, that A = int.)
//
// In early resolution mode, no vtables are recorded, and a number of
// errors are ignored. Early resolution only works if a type is
// *fully* resolved. (We could be less restrictive than that, but it
// would require much more care, and this seems to work decently in
// practice.)

/// Location info records the span and ID of the expression or item that is
/// responsible for this vtable instantiation. (This may not be an expression
/// if the vtable instantiation is being performed as part of "deriving".)
pub struct LocationInfo {
    span: span,
    id: ast::node_id
}

/// A vtable context includes an inference context, a crate context, and a
/// callback function to call in case of type error.
pub struct VtableContext {
    ccx: @mut CrateCtxt,
    infcx: @mut infer::InferCtxt
}

pub impl VtableContext {
    fn tcx(&const self) -> ty::ctxt { self.ccx.tcx }
}

pub fn has_trait_bounds(tps: ~[ty::param_bounds]) -> bool {
    vec::any(tps, |bs| {
        bs.any(|b| {
            match b { &ty::bound_trait(_) => true, _ => false }
        })
    })
}

pub fn lookup_vtables(vcx: &VtableContext,
                      location_info: &LocationInfo,
                      bounds: @~[ty::param_bounds],
                      substs: &ty::substs,
                      is_early: bool) -> vtable_res {
    debug!("lookup_vtables(location_info=%?,
            # bounds=%?, \
            substs=%s",
           location_info,
           bounds.len(),
           ty::substs_to_str(vcx.tcx(), substs));
    let _i = indenter();

    let tcx = vcx.tcx();
    let mut result = ~[], i = 0u;
    for substs.tps.each |ty| {
        for ty::iter_bound_traits_and_supertraits(
            tcx, bounds[i]) |trait_ty|
        {
            debug!("about to subst: %?, %?",
                   ppaux::ty_to_str(tcx, trait_ty),
                   ty::substs_to_str(tcx, substs));

            let new_substs = substs {
                self_ty: Some(*ty),
                ../*bad*/copy *substs
            };
            let trait_ty = ty::subst(tcx, &new_substs, trait_ty);

            debug!("after subst: %?",
                   ppaux::ty_to_str(tcx, trait_ty));

            match lookup_vtable(vcx, location_info, *ty, trait_ty, is_early) {
                Some(vtable) => result.push(vtable),
                None => {
                    vcx.tcx().sess.span_fatal(
                        location_info.span,
                        fmt!("failed to find an implementation of \
                              trait %s for %s",
                             ppaux::ty_to_str(vcx.tcx(), trait_ty),
                             ppaux::ty_to_str(vcx.tcx(), *ty)));
                }
            }
        }
        i += 1u;
    }
    debug!("lookup_vtables result(\
            location_info=%?,
            # bounds=%?, \
            substs=%s, \
            result=%?",
           location_info,
           bounds.len(),
           ty::substs_to_str(vcx.tcx(), substs),
           result);
    @result
}

pub fn fixup_substs(vcx: &VtableContext, location_info: &LocationInfo,
                    id: ast::def_id, +substs: ty::substs,
                    is_early: bool) -> Option<ty::substs> {
    let tcx = vcx.tcx();
    // use a dummy type just to package up the substs that need fixing up
    let t = ty::mk_trait(tcx,
                         id, substs,
                         ty::RegionTraitStore(ty::re_static));
    do fixup_ty(vcx, location_info, t, is_early).map |t_f| {
        match ty::get(*t_f).sty {
          ty::ty_trait(_, ref substs_f, _) => (/*bad*/copy *substs_f),
          _ => fail!(~"t_f should be a trait")
        }
    }
}

pub fn relate_trait_tys(vcx: &VtableContext, location_info: &LocationInfo,
                        exp_trait_ty: ty::t, act_trait_ty: ty::t) {
    demand_suptype(vcx, location_info.span, exp_trait_ty, act_trait_ty)
}

// Look up the vtable to use when treating an item of type `t` as if it has
// type `trait_ty`
pub fn lookup_vtable(vcx: &VtableContext,
                     location_info: &LocationInfo,
                     ty: ty::t,
                     trait_ty: ty::t,
                     is_early: bool)
                  -> Option<vtable_origin> {
    debug!("lookup_vtable(ty=%s, trait_ty=%s)",
           vcx.infcx.ty_to_str(ty), vcx.infcx.ty_to_str(trait_ty));
    let _i = indenter();

    let tcx = vcx.tcx();
    let (trait_id, trait_substs, trait_store) = match ty::get(trait_ty).sty {
        ty::ty_trait(did, ref substs, store) =>
            (did, (/*bad*/copy *substs), store),
        _ => tcx.sess.impossible_case(location_info.span,
                                      "lookup_vtable: \
                                       don't know how to handle a non-trait")
    };
    let ty = match fixup_ty(vcx, location_info, ty, is_early) {
        Some(ty) => ty,
        None => {
            // fixup_ty can only fail if this is early resolution
            assert!(is_early);
            // The type has unconstrained type variables in it, so we can't
            // do early resolution on it. Return some completely bogus vtable
            // information: we aren't storing it anyways.
            return Some(vtable_param(0, 0));
        }
    };

    match ty::get(ty).sty {
        ty::ty_param(param_ty {idx: n, def_id: did}) => {
            let mut n_bound = 0;
            let bounds = *tcx.ty_param_bounds.get(&did.node);
            for ty::iter_bound_traits_and_supertraits(
                tcx, bounds) |ity| {
                debug!("checking bounds trait %?",
                       vcx.infcx.ty_to_str(ity));

                match ty::get(ity).sty {
                    ty::ty_trait(idid, ref isubsts, _) => {
                        if trait_id == idid {
                            debug!("(checking vtable) @0 \
                                    relating ty to trait \
                                    ty with did %?",
                                   idid);

                            // Convert `ity` so that it has the right vstore.
                            let ity = ty::mk_trait(vcx.tcx(),
                                                   idid,
                                                   copy *isubsts,
                                                   trait_store);

                            relate_trait_tys(vcx, location_info,
                                             trait_ty, ity);
                            let vtable = vtable_param(n, n_bound);
                            debug!("found param vtable: %?",
                                   vtable);
                            return Some(vtable);
                        }
                    }
                    _ => tcx.sess.impossible_case(
                        location_info.span,
                        "lookup_vtable: in loop, \
                         don't know how to handle a \
                         non-trait ity")
                }

                n_bound += 1;
            }
        }

        _ => {
            let mut found = ~[];

            let mut impls_seen = HashSet::new();

            match vcx.ccx.coherence_info.extension_methods.find(&trait_id) {
                None => {
                    // Nothing found. Continue.
                }
                Some(implementations) => {
                    let implementations: &mut ~[@Impl] = *implementations;
                    // implementations is the list of all impls in scope for
                    // trait_ty. (Usually, there's just one.)
                    for uint::range(0, implementations.len()) |i| {
                        let im = implementations[i];

                        // im is one specific impl of trait_ty.

                        // First, ensure we haven't processed this impl yet.
                        if impls_seen.contains(&im.did) {
                            loop;
                        }
                        impls_seen.insert(im.did);

                        // ty::impl_traits gives us the list of all
                        // traits that im implements. Again, usually
                        // there's just one.
                        //
                        // For example, if im represented the struct
                        // in:
                        //
                        //   struct foo : baz<int>, bar, quux { ... }
                        //
                        // then ty::impl_traits would return
                        //
                        //   ~[baz<int>, bar, quux]
                        //
                        // For each of the traits foo implements, if
                        // it's the same trait as trait_ty, we need to
                        // unify it with trait_ty in order to get all
                        // the ty vars sorted out.
                        for vec::each(ty::impl_traits(tcx,
                                                      im.did,
                                                      trait_store)) |of_ty| {
                            match ty::get(*of_ty).sty {
                                ty::ty_trait(id, _, _) => {
                                    // Not the trait we're looking for
                                    if id != trait_id { loop; }
                                }
                                _ => { /* ok */ }
                            }

                            // At this point, we know that of_ty is
                            // the same trait as trait_ty, but
                            // possibly applied to different substs.
                            //
                            // Next, we check whether the "for" ty in
                            // the impl is compatible with the type
                            // that we're casting to a trait. That is,
                            // if im is:
                            //
                            // impl<T> self_ty<T>: some_trait<T> { ... }
                            //
                            // we check whether self_ty<T> is the type
                            // of the thing that we're trying to cast
                            // to some_trait.  If not, then we try the next
                            // impl.
                            let ty::ty_param_substs_and_ty {
                                substs: substs,
                                ty: for_ty
                            } = impl_self_ty(vcx, location_info, im.did);
                            match infer::mk_subty(vcx.infcx,
                                                  false,
                                                  location_info.span,
                                                  ty,
                                                  for_ty) {
                                result::Err(_) => loop,
                                result::Ok(()) => ()
                            }

                            // Now, in the previous example, for_ty is
                            // bound to the type self_ty, and substs
                            // is bound to [T].
                            debug!("The self ty is %s and its substs are %s",
                                   vcx.infcx.ty_to_str(for_ty),
                                   tys_to_str(vcx.tcx(), substs.tps));

                            // Next, we unify trait_ty -- the type
                            // that we want to cast to -- with of_ty
                            // -- the trait that im implements. At
                            // this point, we require that they be
                            // unifiable with each other -- that's
                            // what relate_trait_tys does.
                            //
                            // For example, in the above example,
                            // of_ty would be some_trait<T>, so we
                            // would be unifying trait_ty<U> (for some
                            // value of U) with some_trait<T>. This
                            // would fail if T and U weren't
                            // compatible.

                            debug!("(checking vtable) @2 relating trait \
                                    ty %s to of_ty %s",
                                   vcx.infcx.ty_to_str(trait_ty),
                                   vcx.infcx.ty_to_str(*of_ty));
                            let of_ty = ty::subst(tcx, &substs, *of_ty);
                            relate_trait_tys(vcx, location_info, trait_ty,
                                             of_ty);

                            // Recall that trait_ty -- the trait type
                            // we're casting to -- is the trait with
                            // id trait_id applied to the substs
                            // trait_substs. Now we extract out the
                            // types themselves from trait_substs.

                            let trait_tps = /*bad*/copy trait_substs.tps;

                            debug!("Casting to a trait ty whose substs \
                                    (trait_tps) are %s",
                                   tys_to_str(vcx.tcx(), trait_tps));

                            // Recall that substs is the impl self
                            // type's list of substitutions. That is,
                            // if this is an impl of some trait for
                            // foo<T, U>, then substs is [T,
                            // U]. substs might contain type
                            // variables, so we call fixup_substs to
                            // resolve them.

                            let substs_f = match fixup_substs(vcx,
                                                              location_info,
                                                              trait_id,
                                                              substs,
                                                              is_early) {
                                Some(ref substs) => (/*bad*/copy *substs),
                                None => {
                                    assert!(is_early);
                                    // Bail out with a bogus answer
                                    return Some(vtable_param(0, 0));
                                }
                            };

                            debug!("The fixed-up substs are %s - \
                                    they will be unified with the bounds for \
                                    the target ty, %s",
                                   tys_to_str(vcx.tcx(), substs_f.tps),
                                   tys_to_str(vcx.tcx(), trait_tps));

                            // Next, we unify the fixed-up
                            // substitutions for the impl self ty with
                            // the substitutions from the trait type
                            // that we're trying to cast
                            // to. connect_trait_tps requires these
                            // lists of types to unify pairwise.

                            let im_bs = ty::lookup_item_type(tcx,
                                                             im.did).bounds;
                            connect_trait_tps(vcx,
                                              location_info,
                                              /*bad*/copy substs_f.tps,
                                              trait_tps,
                                              im.did,
                                              trait_store);
                            let subres = lookup_vtables(
                                vcx, location_info, im_bs, &substs_f,
                                is_early);

                            // Finally, we register that we found a
                            // matching impl, and record the def ID of
                            // the impl as well as the resolved list
                            // of type substitutions for the target
                            // trait.
                            found.push(
                                vtable_static(im.did,
                                              /*bad*/copy substs_f.tps,
                                              subres));
                        }
                    }
                }
            }

            match found.len() {
                0 => { /* fallthrough */ }
                1 => { return Some(/*bad*/copy found[0]); }
                _ => {
                    if !is_early {
                        vcx.tcx().sess.span_err(
                            location_info.span,
                            ~"multiple applicable methods in scope");
                    }
                    return Some(/*bad*/copy found[0]);
                }
            }
        }
    }

    return None;
}

pub fn fixup_ty(vcx: &VtableContext,
                location_info: &LocationInfo,
                ty: ty::t,
                is_early: bool) -> Option<ty::t> {
    let tcx = vcx.tcx();
    match resolve_type(vcx.infcx, ty, resolve_and_force_all_but_regions) {
        Ok(new_type) => Some(new_type),
        Err(e) if !is_early => {
            tcx.sess.span_fatal(
                location_info.span,
                fmt!("cannot determine a type \
                      for this bounded type parameter: %s",
                     fixup_err_to_str(e)))
        }
        Err(_) => {
            None
        }
    }
}

// Version of demand::suptype() that takes a vtable context instead of a
// function context.
pub fn demand_suptype(vcx: &VtableContext, sp: span, e: ty::t, a: ty::t) {
    // NB: Order of actual, expected is reversed.
    match infer::mk_subty(vcx.infcx, false, sp, a, e) {
        result::Ok(()) => {} // Ok.
        result::Err(ref err) => {
            vcx.infcx.report_mismatched_types(sp, e, a, err);
        }
    }
}

pub fn connect_trait_tps(vcx: &VtableContext,
                         location_info: &LocationInfo,
                         impl_tys: ~[ty::t],
                         trait_tys: ~[ty::t],
                         impl_did: ast::def_id,
                         store: ty::TraitStore) {
    let tcx = vcx.tcx();

    // XXX: This should work for multiple traits.
    let ity = ty::impl_traits(tcx, impl_did, store)[0];
    let trait_ty = ty::subst_tps(tcx, impl_tys, None, ity);
    debug!("(connect trait tps) trait type is %?, impl did is %?",
           ty::get(trait_ty).sty, impl_did);
    match ty::get(trait_ty).sty {
     ty::ty_trait(_, ref substs, _) => {
         for vec::each2((*substs).tps, trait_tys) |a, b| {
            demand_suptype(vcx, location_info.span, *a, *b);
         }
      }
     _ => tcx.sess.impossible_case(location_info.span, "connect_trait_tps: \
            don't know how to handle a non-trait ty")
    }
}

pub fn insert_vtables(fcx: @mut FnCtxt,
                      callee_id: ast::node_id,
                      vtables: vtable_res) {
    debug!("insert_vtables(callee_id=%d, vtables=%?)",
           callee_id, vtables.map(|v| v.to_str(fcx.tcx())));
    fcx.inh.vtable_map.insert(callee_id, vtables);
}

pub fn location_info_for_expr(expr: @ast::expr) -> LocationInfo {
    LocationInfo {
        span: expr.span,
        id: expr.id
    }
}

pub fn early_resolve_expr(ex: @ast::expr,
                          &&fcx: @mut FnCtxt,
                          is_early: bool) {
    debug!("vtable: early_resolve_expr() ex with id %? (early: %b): %s",
           ex.id, is_early, expr_to_str(ex, fcx.tcx().sess.intr()));
    let _indent = indenter();

    let cx = fcx.ccx;
    match ex.node {
      ast::expr_path(*) => {
        for fcx.opt_node_ty_substs(ex.id) |substs| {
            let def = *cx.tcx.def_map.get(&ex.id);
            let did = ast_util::def_id_of_def(def);
            let item_ty = ty::lookup_item_type(cx.tcx, did);
            debug!("early resolve expr: def %? %?, %?, %?", ex.id, did, def,
                   fcx.infcx().ty_to_str(item_ty.ty));
            if has_trait_bounds(/*bad*/copy *item_ty.bounds) {
                for item_ty.bounds.each |bounds| {
                    debug!("early_resolve_expr: looking up vtables for bound \
                            %s",
                           ty::param_bounds_to_str(fcx.tcx(), *bounds));
                }
                let vcx = VtableContext { ccx: fcx.ccx, infcx: fcx.infcx() };
                let vtbls = lookup_vtables(&vcx, &location_info_for_expr(ex),
                                           item_ty.bounds, substs, is_early);
                if !is_early {
                    insert_vtables(fcx, ex.id, vtbls);
                }
            }
        }
      }

      ast::expr_paren(e) => {
          early_resolve_expr(e, fcx, is_early);
      }

      // Must resolve bounds on methods with bounded params
      ast::expr_binary(*) |
      ast::expr_unary(*) | ast::expr_assign_op(*) |
      ast::expr_index(*) | ast::expr_method_call(*) => {
        match ty::method_call_bounds(cx.tcx, fcx.inh.method_map, ex.id) {
          Some(bounds) => {
            if has_trait_bounds(/*bad*/copy *bounds) {
                let callee_id = match ex.node {
                  ast::expr_field(_, _, _) => ex.id,
                  _ => ex.callee_id
                };

                let substs = fcx.node_ty_substs(callee_id);
                let vcx = VtableContext { ccx: fcx.ccx, infcx: fcx.infcx() };
                let vtbls = lookup_vtables(&vcx, &location_info_for_expr(ex),
                                           bounds, &substs, is_early);
                if !is_early {
                    insert_vtables(fcx, callee_id, vtbls);
                }
            }
          }
          None => ()
        }
      }
      ast::expr_cast(src, _) => {
          let target_ty = fcx.expr_ty(ex);
          match ty::get(target_ty).sty {
              ty::ty_trait(_, _, store) => {
                  // Look up vtables for the type we're casting to,
                  // passing in the source and target type.  The source
                  // must be a pointer type suitable to the object sigil,
                  // e.g.: `@x as @Trait`, `&x as &Trait` or `~x as ~Trait`
                  let ty = structurally_resolved_type(fcx, ex.span,
                                                      fcx.expr_ty(src));
                  match (&ty::get(ty).sty, store) {
                      (&ty::ty_box(mt), ty::BoxTraitStore) |
                      // XXX: Bare trait store is deprecated.
                      (&ty::ty_uniq(mt), ty::UniqTraitStore) |
                      (&ty::ty_rptr(_, mt), ty::RegionTraitStore(*)) => {
                          let location_info =
                              &location_info_for_expr(ex);
                          let vcx = VtableContext {
                              ccx: fcx.ccx,
                              infcx: fcx.infcx()
                          };
                          let vtable_opt =
                              lookup_vtable(&vcx,
                                            location_info,
                                            mt.ty,
                                            target_ty,
                                            is_early);
                          match vtable_opt {
                              Some(vtable) => {
                                  // Map this expression to that
                                  // vtable (that is: "ex has vtable
                                  // <vtable>")
                                  if !is_early {
                                      insert_vtables(fcx, ex.id, @~[vtable]);
                                  }
                              }
                              None => {
                                  fcx.tcx().sess.span_err(
                                      ex.span,
                                      fmt!("failed to find an implementation \
                                            of trait %s for %s",
                                           fcx.infcx().ty_to_str(target_ty),
                                           fcx.infcx().ty_to_str(mt.ty)));
                              }
                          }

                          // Now, if this is &trait, we need to link the
                          // regions.
                          match (&ty::get(ty).sty, store) {
                              (&ty::ty_rptr(ra, _),
                               ty::RegionTraitStore(rb)) => {
                                  infer::mk_subr(fcx.infcx(),
                                                 false,
                                                 ex.span,
                                                 rb,
                                                 ra);
                              }
                              _ => {}
                          }
                      }

                      (_, ty::BareTraitStore) => {
                          fcx.ccx.tcx.sess.span_err(
                              ex.span,
                              ~"a sigil (`@`, `~`, or `&`) must be specified \
                                when casting to a trait");
                      }

                      (_, ty::BoxTraitStore) => {
                          fcx.ccx.tcx.sess.span_err(
                              ex.span,
                              fmt!("can only cast an @-pointer \
                                    to an @-object, not a %s",
                                   ty::ty_sort_str(fcx.tcx(), ty)));
                      }

                      (_, ty::UniqTraitStore) => {
                          fcx.ccx.tcx.sess.span_err(
                              ex.span,
                              fmt!("can only cast an ~-pointer \
                                    to a ~-object, not a %s",
                                   ty::ty_sort_str(fcx.tcx(), ty)));
                      }

                      (_, ty::RegionTraitStore(_)) => {
                          fcx.ccx.tcx.sess.span_err(
                              ex.span,
                              fmt!("can only cast an &-pointer \
                                    to an &-object, not a %s",
                                   ty::ty_sort_str(fcx.tcx(), ty)));
                      }
                  }
              }
              _ => { /* not a cast to a trait; ignore */ }
          }
      }
      _ => ()
    }
}

pub fn resolve_expr(ex: @ast::expr,
                    &&fcx: @mut FnCtxt,
                    v: visit::vt<@mut FnCtxt>) {
    early_resolve_expr(ex, fcx, false);
    visit::visit_expr(ex, fcx, v);
}

// Detect points where a trait-bounded type parameter is
// instantiated, resolve the impls for the parameters.
pub fn resolve_in_block(fcx: @mut FnCtxt, bl: &ast::blk) {
    visit::visit_block(bl, fcx, visit::mk_vt(@visit::Visitor {
        visit_expr: resolve_expr,
        visit_item: |_,_,_| {},
        .. *visit::default_visitor()
    }));
}


