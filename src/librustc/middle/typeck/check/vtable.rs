// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use middle::ty::param_ty;
use middle::ty;
use middle::typeck::check::{FnCtxt, impl_self_ty};
use middle::typeck::check::{structurally_resolved_type};
use middle::typeck::infer::fixup_err_to_str;
use middle::typeck::infer::{resolve_and_force_all_but_regions, resolve_type};
use middle::typeck::infer;
use middle::typeck::{CrateCtxt, vtable_origin, vtable_res, vtable_param_res};
use middle::typeck::{vtable_static, vtable_param, impl_res};
use middle::typeck::{param_numbered, param_self, param_index};
use middle::subst::Subst;
use util::common::indenter;
use util::ppaux;

use std::hashmap::HashSet;
use std::result;
use syntax::ast;
use syntax::ast_util;
use syntax::codemap::span;
use syntax::print::pprust::expr_to_str;
use syntax::oldvisit;

// vtable resolution looks for places where trait bounds are
// substituted in and figures out which vtable is used. There is some
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
//
// While resolution on a single type requires the type to be fully
// resolved, when resolving a substitution against a list of bounds,
// we do not require all of the types to be resolved in advance.
// Furthermore, we process substitutions in reverse order, which
// allows resolution on later parameters to give information on
// earlier params referenced by the typeclass bounds.
// It may be better to do something more clever, like processing fully
// resolved types first.


/// Location info records the span and ID of the expression or item that is
/// responsible for this vtable instantiation. (This may not be an expression
/// if the vtable instantiation is being performed as part of "deriving".)
pub struct LocationInfo {
    span: span,
    id: ast::NodeId
}

/// A vtable context includes an inference context, a crate context, and a
/// callback function to call in case of type error.
pub struct VtableContext {
    ccx: @mut CrateCtxt,
    infcx: @mut infer::InferCtxt
}

impl VtableContext {
    pub fn tcx(&self) -> ty::ctxt { self.ccx.tcx }
}

fn has_trait_bounds(type_param_defs: &[ty::TypeParameterDef]) -> bool {
    type_param_defs.iter().any(
        |type_param_def| !type_param_def.bounds.trait_bounds.is_empty())
}

fn lookup_vtables(vcx: &VtableContext,
                  location_info: &LocationInfo,
                  type_param_defs: &[ty::TypeParameterDef],
                  substs: &ty::substs,
                  is_early: bool) -> vtable_res {
    debug!("lookup_vtables(location_info=%?, \
            type_param_defs=%s, \
            substs=%s",
           location_info,
           type_param_defs.repr(vcx.tcx()),
           substs.repr(vcx.tcx()));
    let _i = indenter();


    // We do this backwards for reasons discussed above.
    assert_eq!(substs.tps.len(), type_param_defs.len());
    let mut result =
        substs.tps.rev_iter()
        .zip(type_param_defs.rev_iter())
        .map(|(ty, def)|
                   lookup_vtables_for_param(vcx, location_info, Some(substs),
                                            &*def.bounds, *ty, is_early))
        .to_owned_vec();
    result.reverse();

    assert_eq!(substs.tps.len(), result.len());
    debug!("lookup_vtables result(\
            location_info=%?, \
            type_param_defs=%s, \
            substs=%s, \
            result=%s)",
           location_info,
           type_param_defs.repr(vcx.tcx()),
           substs.repr(vcx.tcx()),
           result.repr(vcx.tcx()));
    @result
}

fn lookup_vtables_for_param(vcx: &VtableContext,
                            location_info: &LocationInfo,
                            // None for substs means the identity
                            substs: Option<&ty::substs>,
                            type_param_bounds: &ty::ParamBounds,
                            ty: ty::t,
                            is_early: bool) -> vtable_param_res {
    let tcx = vcx.tcx();

    // ty is the value supplied for the type parameter A...
    let mut param_result = ~[];

    do ty::each_bound_trait_and_supertraits(tcx, type_param_bounds.trait_bounds) |trait_ref| {
        // ...and here trait_ref is each bound that was declared on A,
        // expressed in terms of the type parameters.

        // Substitute the values of the type parameters that may
        // appear in the bound.
        let trait_ref = substs.map_default(trait_ref, |substs| {
            debug!("about to subst: %s, %s",
                   trait_ref.repr(tcx), substs.repr(tcx));
            trait_ref.subst(tcx, *substs)
        });

        debug!("after subst: %s", trait_ref.repr(tcx));

        match lookup_vtable(vcx, location_info, ty, trait_ref, is_early) {
            Some(vtable) => param_result.push(vtable),
            None => {
                vcx.tcx().sess.span_fatal(
                    location_info.span,
                    fmt!("failed to find an implementation of \
                          trait %s for %s",
                         vcx.infcx.trait_ref_to_str(trait_ref),
                         vcx.infcx.ty_to_str(ty)));
            }
        }
        true
    };

    debug!("lookup_vtables_for_param result(\
            location_info=%?, \
            type_param_bounds=%s, \
            ty=%s, \
            result=%s)",
           location_info,
           type_param_bounds.repr(vcx.tcx()),
           ty.repr(vcx.tcx()),
           param_result.repr(vcx.tcx()));

    return @param_result;
}

fn relate_trait_refs(vcx: &VtableContext,
                     location_info: &LocationInfo,
                     act_trait_ref: @ty::TraitRef,
                     exp_trait_ref: @ty::TraitRef)
{
    /*!
     *
     * Checks that an implementation of `act_trait_ref` is suitable
     * for use where `exp_trait_ref` is required and reports an
     * error otherwise.
     */

    match infer::mk_sub_trait_refs(vcx.infcx,
                                   false,
                                   infer::RelateTraitRefs(location_info.span),
                                   act_trait_ref,
                                   exp_trait_ref)
    {
        result::Ok(()) => {} // Ok.
        result::Err(ref err) => {
            // There is an error, but we need to do some work to make
            // the message good.
            // Resolve any type vars in the trait refs
            let r_act_trait_ref =
                vcx.infcx.resolve_type_vars_in_trait_ref_if_possible(act_trait_ref);
            let r_exp_trait_ref =
                vcx.infcx.resolve_type_vars_in_trait_ref_if_possible(exp_trait_ref);
            // Only print the message if there aren't any previous type errors
            // inside the types.
            if !ty::trait_ref_contains_error(&r_act_trait_ref) &&
                !ty::trait_ref_contains_error(&r_exp_trait_ref)
            {
                let tcx = vcx.tcx();
                tcx.sess.span_err(
                    location_info.span,
                    fmt!("expected %s, but found %s (%s)",
                         ppaux::trait_ref_to_str(tcx, &r_exp_trait_ref),
                         ppaux::trait_ref_to_str(tcx, &r_act_trait_ref),
                         ty::type_err_to_str(tcx, err)));
            }
        }
    }
}

// Look up the vtable implementing the trait `trait_ref` at type `t`
fn lookup_vtable(vcx: &VtableContext,
                 location_info: &LocationInfo,
                 ty: ty::t,
                 trait_ref: @ty::TraitRef,
                 is_early: bool)
    -> Option<vtable_origin>
{
    debug!("lookup_vtable(ty=%s, trait_ref=%s)",
           vcx.infcx.ty_to_str(ty),
           vcx.infcx.trait_ref_to_str(trait_ref));
    let _i = indenter();

    let tcx = vcx.tcx();

    let ty = match fixup_ty(vcx, location_info, ty, is_early) {
        Some(ty) => ty,
        None => {
            // fixup_ty can only fail if this is early resolution
            assert!(is_early);
            // The type has unconstrained type variables in it, so we can't
            // do early resolution on it. Return some completely bogus vtable
            // information: we aren't storing it anyways.
            return Some(vtable_param(param_self, 0));
        }
    };

    // If the type is self or a param, we look at the trait/supertrait
    // bounds to see if they include the trait we are looking for.
    let vtable_opt = match ty::get(ty).sty {
        ty::ty_param(param_ty {idx: n, def_id: did}) => {
            let type_param_def = tcx.ty_param_defs.get(&did.node);
            lookup_vtable_from_bounds(vcx, location_info,
                                      type_param_def.bounds.trait_bounds,
                                      param_numbered(n),
                                      trait_ref)
        }

        ty::ty_self(trait_id) => {
            let self_trait_ref = ty::lookup_trait_def(tcx, trait_id).trait_ref;
            lookup_vtable_from_bounds(vcx, location_info,
                                      &[self_trait_ref],
                                      param_self,
                                      trait_ref)
        }

        // Default case just falls through
        _ => None
    };

    if vtable_opt.is_some() { return vtable_opt; }

    // If we aren't a self type or param, or it was, but we didn't find it,
    // do a search.
    return search_for_vtable(vcx, location_info,
                             ty, trait_ref, is_early)
}

// Given a list of bounds on a type, search those bounds to see if any
// of them are the vtable we are looking for.
fn lookup_vtable_from_bounds(vcx: &VtableContext,
                             location_info: &LocationInfo,
                             bounds: &[@ty::TraitRef],
                             param: param_index,
                             trait_ref: @ty::TraitRef)
    -> Option<vtable_origin> {
    let tcx = vcx.tcx();

    let mut n_bound = 0;
    let mut ret = None;
    do ty::each_bound_trait_and_supertraits(tcx, bounds) |bound_trait_ref| {
        debug!("checking bounds trait %s",
               bound_trait_ref.repr(vcx.tcx()));

        if bound_trait_ref.def_id == trait_ref.def_id {
            relate_trait_refs(vcx,
                              location_info,
                              bound_trait_ref,
                              trait_ref);
            let vtable = vtable_param(param, n_bound);
            debug!("found param vtable: %?",
                   vtable);
            ret = Some(vtable);
            false
        } else {
            n_bound += 1;
            true
        }
    };
    ret
}

fn search_for_vtable(vcx: &VtableContext,
                     location_info: &LocationInfo,
                     ty: ty::t,
                     trait_ref: @ty::TraitRef,
                     is_early: bool)
    -> Option<vtable_origin>
{
    let tcx = vcx.tcx();

    let mut found = ~[];
    let mut impls_seen = HashSet::new();

    // XXX: this is a bad way to do this, since we do
    // pointless allocations.
    let impls = tcx.trait_impls.find(&trait_ref.def_id)
        .map_default(@mut ~[], |x| **x);
    // impls is the list of all impls in scope for trait_ref.
    for im in impls.iter() {
        // im is one specific impl of trait_ref.

        // First, ensure we haven't processed this impl yet.
        if impls_seen.contains(&im.did) {
            loop;
        }
        impls_seen.insert(im.did);

        // ty::impl_traits gives us the trait im implements.
        //
        // If foo implements a trait t, and if t is the same trait as
        // trait_ref, we need to unify it with trait_ref in order to
        // get all the ty vars sorted out.
        let r = ty::impl_trait_ref(tcx, im.did);
        let of_trait_ref = r.expect("trait_ref missing on trait impl");
        if of_trait_ref.def_id != trait_ref.def_id { loop; }

        // At this point, we know that of_trait_ref is the same trait
        // as trait_ref, but possibly applied to different substs.
        //
        // Next, we check whether the "for" ty in the impl is
        // compatible with the type that we're casting to a
        // trait. That is, if im is:
        //
        // impl<T> some_trait<T> for self_ty<T> { ... }
        //
        // we check whether self_ty<T> is the type of the thing that
        // we're trying to cast to some_trait.  If not, then we try
        // the next impl.
        //
        // XXX: document a bit more what this means
        //
        // FIXME(#5781) this should be mk_eqty not mk_subty
        let ty::ty_param_substs_and_ty {
            substs: substs,
            ty: for_ty
        } = impl_self_ty(vcx, location_info, im.did);
        match infer::mk_subty(vcx.infcx,
                              false,
                              infer::RelateSelfType(
                                  location_info.span),
                              ty,
                              for_ty) {
            result::Err(_) => loop,
            result::Ok(()) => ()
        }

        // Now, in the previous example, for_ty is bound to
        // the type self_ty, and substs is bound to [T].
        debug!("The self ty is %s and its substs are %s",
               vcx.infcx.ty_to_str(for_ty),
               vcx.infcx.tys_to_str(substs.tps));

        // Next, we unify trait_ref -- the type that we want to cast
        // to -- with of_trait_ref -- the trait that im implements. At
        // this point, we require that they be unifiable with each
        // other -- that's what relate_trait_refs does.
        //
        // For example, in the above example, of_trait_ref would be
        // some_trait<T>, so we would be unifying trait_ref<U> (for
        // some value of U) with some_trait<T>. This would fail if T
        // and U weren't compatible.

        debug!("(checking vtable) @2 relating trait \
                ty %s to of_trait_ref %s",
               vcx.infcx.trait_ref_to_str(trait_ref),
               vcx.infcx.trait_ref_to_str(of_trait_ref));

        let of_trait_ref = of_trait_ref.subst(tcx, &substs);
        relate_trait_refs(vcx, location_info, of_trait_ref, trait_ref);


        // Recall that trait_ref -- the trait type we're casting to --
        // is the trait with id trait_ref.def_id applied to the substs
        // trait_ref.substs.

        // Resolve any sub bounds. Note that there still may be free
        // type variables in substs. This might still be OK: the
        // process of looking up bounds might constrain some of them.
        let im_generics =
            ty::lookup_item_type(tcx, im.did).generics;
        let subres = lookup_vtables(vcx, location_info,
                                    *im_generics.type_param_defs, &substs,
                                    is_early);


        // substs might contain type variables, so we call
        // fixup_substs to resolve them.
        let substs_f = match fixup_substs(vcx,
                                          location_info,
                                          trait_ref.def_id,
                                          substs,
                                          is_early) {
            Some(ref substs) => (*substs).clone(),
            None => {
                assert!(is_early);
                // Bail out with a bogus answer
                return Some(vtable_param(param_self, 0));
            }
        };

        debug!("The fixed-up substs are %s - \
                they will be unified with the bounds for \
                the target ty, %s",
               vcx.infcx.tys_to_str(substs_f.tps),
               vcx.infcx.trait_ref_to_str(trait_ref));

        // Next, we unify the fixed-up substitutions for the impl self
        // ty with the substitutions from the trait type that we're
        // trying to cast to. connect_trait_tps requires these lists
        // of types to unify pairwise.
        // I am a little confused about this, since it seems to be
        // very similar to the relate_trait_refs we already do,
        // but problems crop up if it is removed, so... -sully
        connect_trait_tps(vcx, location_info, &substs_f, trait_ref, im.did);

        // Finally, we register that we found a matching impl, and
        // record the def ID of the impl as well as the resolved list
        // of type substitutions for the target trait.
        found.push(vtable_static(im.did, substs_f.tps.clone(), subres));
    }

    match found.len() {
        0 => { return None }
        1 => return Some(found[0].clone()),
        _ => {
            if !is_early {
                vcx.tcx().sess.span_err(
                    location_info.span,
                    "multiple applicable methods in scope");
            }
            return Some(found[0].clone());
        }
    }
}


fn fixup_substs(vcx: &VtableContext,
                location_info: &LocationInfo,
                id: ast::def_id,
                substs: ty::substs,
                is_early: bool)
                -> Option<ty::substs> {
    let tcx = vcx.tcx();
    // use a dummy type just to package up the substs that need fixing up
    let t = ty::mk_trait(tcx,
                         id, substs,
                         ty::RegionTraitStore(ty::re_static),
                         ast::m_imm,
                         ty::EmptyBuiltinBounds());
    do fixup_ty(vcx, location_info, t, is_early).map |t_f| {
        match ty::get(*t_f).sty {
          ty::ty_trait(_, ref substs_f, _, _, _) => (*substs_f).clone(),
          _ => fail!("t_f should be a trait")
        }
    }
}

fn fixup_ty(vcx: &VtableContext,
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

fn connect_trait_tps(vcx: &VtableContext,
                     location_info: &LocationInfo,
                     impl_substs: &ty::substs,
                     trait_ref: @ty::TraitRef,
                     impl_did: ast::def_id)
{
    let tcx = vcx.tcx();

    let impl_trait_ref = match ty::impl_trait_ref(tcx, impl_did) {
        Some(t) => t,
        None => vcx.tcx().sess.span_bug(location_info.span,
                                  "connect_trait_tps invoked on a type impl")
    };

    let impl_trait_ref = impl_trait_ref.subst(tcx, impl_substs);
    relate_trait_refs(vcx, location_info, impl_trait_ref, trait_ref);
}

fn insert_vtables(fcx: @mut FnCtxt,
                  callee_id: ast::NodeId,
                  vtables: vtable_res) {
    debug!("insert_vtables(callee_id=%d, vtables=%?)",
           callee_id, vtables.repr(fcx.tcx()));
    fcx.inh.vtable_map.insert(callee_id, vtables);
}

pub fn location_info_for_expr(expr: @ast::expr) -> LocationInfo {
    LocationInfo {
        span: expr.span,
        id: expr.id
    }
}
pub fn location_info_for_item(item: @ast::item) -> LocationInfo {
    LocationInfo {
        span: item.span,
        id: item.id
    }
}

pub fn early_resolve_expr(ex: @ast::expr,
                          fcx: @mut FnCtxt,
                          is_early: bool) {
    debug!("vtable: early_resolve_expr() ex with id %? (early: %b): %s",
           ex.id, is_early, expr_to_str(ex, fcx.tcx().sess.intr()));
    let _indent = indenter();

    let cx = fcx.ccx;
    match ex.node {
      ast::expr_path(*) => {
        do fcx.opt_node_ty_substs(ex.id) |substs| {
            debug!("vtable resolution on parameter bounds for expr %s",
                   ex.repr(fcx.tcx()));
            let def = cx.tcx.def_map.get_copy(&ex.id);
            let did = ast_util::def_id_of_def(def);
            let item_ty = ty::lookup_item_type(cx.tcx, did);
            debug!("early resolve expr: def %? %?, %?, %s", ex.id, did, def,
                   fcx.infcx().ty_to_str(item_ty.ty));
            if has_trait_bounds(*item_ty.generics.type_param_defs) {
                debug!("early_resolve_expr: looking up vtables for type params %s",
                       item_ty.generics.type_param_defs.repr(fcx.tcx()));
                let vcx = VtableContext { ccx: fcx.ccx, infcx: fcx.infcx() };
                let vtbls = lookup_vtables(&vcx, &location_info_for_expr(ex),
                                           *item_ty.generics.type_param_defs,
                                           substs, is_early);
                if !is_early {
                    insert_vtables(fcx, ex.id, vtbls);
                }
            }
            true
        };
      }

      ast::expr_paren(e) => {
          early_resolve_expr(e, fcx, is_early);
      }

      // Must resolve bounds on methods with bounded params
      ast::expr_binary(callee_id, _, _, _) |
      ast::expr_unary(callee_id, _, _) |
      ast::expr_assign_op(callee_id, _, _, _) |
      ast::expr_index(callee_id, _, _) |
      ast::expr_method_call(callee_id, _, _, _, _, _) => {
        match ty::method_call_type_param_defs(cx.tcx, fcx.inh.method_map, ex.id) {
          Some(type_param_defs) => {
            debug!("vtable resolution on parameter bounds for method call %s",
                   ex.repr(fcx.tcx()));
            if has_trait_bounds(*type_param_defs) {
                let substs = fcx.node_ty_substs(callee_id);
                let vcx = VtableContext { ccx: fcx.ccx, infcx: fcx.infcx() };
                let vtbls = lookup_vtables(&vcx, &location_info_for_expr(ex),
                                           *type_param_defs, &substs, is_early);
                if !is_early {
                    insert_vtables(fcx, callee_id, vtbls);
                }
            }
          }
          None => ()
        }
      }
      ast::expr_cast(src, _) => {
          debug!("vtable resolution on expr %s", ex.repr(fcx.tcx()));
          let target_ty = fcx.expr_ty(ex);
          match ty::get(target_ty).sty {
              // Bounds of type's contents are not checked here, but in kind.rs.
              ty::ty_trait(target_def_id, ref target_substs, store,
                           target_mutbl, _bounds) => {
                  fn mutability_allowed(a_mutbl: ast::mutability,
                                        b_mutbl: ast::mutability) -> bool {
                      a_mutbl == b_mutbl ||
                      (a_mutbl == ast::m_mutbl && b_mutbl == ast::m_imm)
                  }
                  // Look up vtables for the type we're casting to,
                  // passing in the source and target type.  The source
                  // must be a pointer type suitable to the object sigil,
                  // e.g.: `@x as @Trait`, `&x as &Trait` or `~x as ~Trait`
                  let ty = structurally_resolved_type(fcx, ex.span,
                                                      fcx.expr_ty(src));
                  match (&ty::get(ty).sty, store) {
                      (&ty::ty_box(mt), ty::BoxTraitStore) |
                      (&ty::ty_uniq(mt), ty::UniqTraitStore) |
                      (&ty::ty_rptr(_, mt), ty::RegionTraitStore(*))
                        if !mutability_allowed(mt.mutbl, target_mutbl) => {
                          fcx.tcx().sess.span_err(ex.span,
                                                  fmt!("types differ in mutability"));
                      }

                      (&ty::ty_box(mt), ty::BoxTraitStore) |
                      (&ty::ty_uniq(mt), ty::UniqTraitStore) |
                      (&ty::ty_rptr(_, mt), ty::RegionTraitStore(*)) => {
                          let location_info =
                              &location_info_for_expr(ex);
                          let vcx = VtableContext {
                              ccx: fcx.ccx,
                              infcx: fcx.infcx()
                          };
                          let target_trait_ref = @ty::TraitRef {
                              def_id: target_def_id,
                              substs: ty::substs {
                                  tps: target_substs.tps.clone(),
                                  regions: target_substs.regions.clone(),
                                  self_ty: Some(mt.ty)
                              }
                          };

                          let param_bounds = ty::ParamBounds {
                              builtin_bounds: ty::EmptyBuiltinBounds(),
                              trait_bounds: ~[target_trait_ref]
                          };
                          let vtables =
                                lookup_vtables_for_param(&vcx,
                                                         location_info,
                                                         None,
                                                         &param_bounds,
                                                         mt.ty,
                                                         is_early);

                          if !is_early {
                              insert_vtables(fcx, ex.id, @~[vtables]);
                          }

                          // Now, if this is &trait, we need to link the
                          // regions.
                          match (&ty::get(ty).sty, store) {
                              (&ty::ty_rptr(ra, _),
                               ty::RegionTraitStore(rb)) => {
                                  infer::mk_subr(fcx.infcx(),
                                                 false,
                                                 infer::RelateObjectBound(
                                                     ex.span),
                                                 rb,
                                                 ra);
                              }
                              _ => {}
                          }
                      }

                      (_, ty::UniqTraitStore) => {
                          fcx.ccx.tcx.sess.span_err(
                              ex.span,
                              fmt!("can only cast an ~-pointer \
                                    to a ~-object, not a %s",
                                   ty::ty_sort_str(fcx.tcx(), ty)));
                      }

                      (_, ty::BoxTraitStore) => {
                          fcx.ccx.tcx.sess.span_err(
                              ex.span,
                              fmt!("can only cast an @-pointer \
                                    to an @-object, not a %s",
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

fn resolve_expr(ex: @ast::expr,
                (fcx, v): (@mut FnCtxt,
                           oldvisit::vt<@mut FnCtxt>)) {
    early_resolve_expr(ex, fcx, false);
    oldvisit::visit_expr(ex, (fcx, v));
}

pub fn resolve_impl(ccx: @mut CrateCtxt, impl_item: @ast::item) {
    let def_id = ast_util::local_def(impl_item.id);
    match ty::impl_trait_ref(ccx.tcx, def_id) {
        None => {},
        Some(trait_ref) => {
            let infcx = infer::new_infer_ctxt(ccx.tcx);
            let vcx = VtableContext { ccx: ccx, infcx: infcx };
            let loc_info = location_info_for_item(impl_item);

            // First, check that the impl implements any trait bounds
            // on the trait.
            let trait_def = ty::lookup_trait_def(ccx.tcx, trait_ref.def_id);
            let vtbls = lookup_vtables(&vcx,
                                       &loc_info,
                                       *trait_def.generics.type_param_defs,
                                       &trait_ref.substs,
                                       false);

            // Now, locate the vtable for the impl itself. The real
            // purpose of this is to check for supertrait impls,
            // but that falls out of doing this.
            let param_bounds = ty::ParamBounds {
                builtin_bounds: ty::EmptyBuiltinBounds(),
                trait_bounds: ~[trait_ref]
            };
            let t = ty::node_id_to_type(ccx.tcx, impl_item.id);
            debug!("=== Doing a self lookup now.");
            // Right now, we don't have any place to store this.
            // We will need to make one so we can use this information
            // for compiling default methods that refer to supertraits.
            let self_vtable_res =
                lookup_vtables_for_param(&vcx, &loc_info, None,
                                         &param_bounds, t, false);


            let res = impl_res {
                trait_vtables: vtbls,
                self_vtables: self_vtable_res
            };
            ccx.tcx.impl_vtables.insert(def_id, res);
        }
    }
}

// Detect points where a trait-bounded type parameter is
// instantiated, resolve the impls for the parameters.
pub fn resolve_in_block(fcx: @mut FnCtxt, bl: &ast::Block) {
    oldvisit::visit_block(bl, (fcx, oldvisit::mk_vt(@oldvisit::Visitor {
        visit_expr: resolve_expr,
        visit_item: |_,_| {},
        .. *oldvisit::default_visitor()
    })));
}
