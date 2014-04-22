// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use middle::ty;
use middle::ty::{AutoAddEnv, AutoDerefRef, AutoObject, param_ty};
use middle::ty_fold::TypeFolder;
use middle::typeck::astconv::AstConv;
use middle::typeck::check::{FnCtxt, impl_self_ty};
use middle::typeck::check::{structurally_resolved_type};
use middle::typeck::infer::fixup_err_to_str;
use middle::typeck::infer::{resolve_and_force_all_but_regions, resolve_type};
use middle::typeck::infer;
use middle::typeck::{vtable_origin, vtable_res, vtable_param_res};
use middle::typeck::{vtable_static, vtable_param, impl_res};
use middle::typeck::{param_numbered, param_self, param_index};
use middle::typeck::MethodCall;
use middle::subst::Subst;
use util::common::indenter;
use util::ppaux;
use util::ppaux::Repr;

use std::rc::Rc;
use collections::HashSet;
use syntax::ast;
use syntax::ast_util;
use syntax::codemap::Span;
use syntax::print::pprust::expr_to_str;
use syntax::visit;
use syntax::visit::Visitor;

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

/// A vtable context includes an inference context, a crate context, and a
/// callback function to call in case of type error.
pub struct VtableContext<'a> {
    pub infcx: &'a infer::InferCtxt<'a>,
    pub param_env: &'a ty::ParameterEnvironment,
}

impl<'a> VtableContext<'a> {
    pub fn tcx(&self) -> &'a ty::ctxt { self.infcx.tcx }
}

fn has_trait_bounds(type_param_defs: &[ty::TypeParameterDef]) -> bool {
    type_param_defs.iter().any(
        |type_param_def| !type_param_def.bounds.trait_bounds.is_empty())
}

fn lookup_vtables(vcx: &VtableContext,
                  span: Span,
                  type_param_defs: &[ty::TypeParameterDef],
                  substs: &ty::substs,
                  is_early: bool) -> vtable_res {
    debug!("lookup_vtables(span={:?}, \
            type_param_defs={}, \
            substs={}",
           span,
           type_param_defs.repr(vcx.tcx()),
           substs.repr(vcx.tcx()));

    // We do this backwards for reasons discussed above.
    assert_eq!(substs.tps.len(), type_param_defs.len());
    let mut result: Vec<vtable_param_res> =
        substs.tps.iter()
        .rev()
        .zip(type_param_defs.rev_iter())
        .map(|(ty, def)|
            lookup_vtables_for_param(vcx, span, Some(substs),
                                     &*def.bounds, *ty, is_early))
        .collect();
    result.reverse();

    assert_eq!(substs.tps.len(), result.len());
    debug!("lookup_vtables result(\
            span={:?}, \
            type_param_defs={}, \
            substs={}, \
            result={})",
           span,
           type_param_defs.repr(vcx.tcx()),
           substs.repr(vcx.tcx()),
           result.repr(vcx.tcx()));
    result
}

fn lookup_vtables_for_param(vcx: &VtableContext,
                            span: Span,
                            // None for substs means the identity
                            substs: Option<&ty::substs>,
                            type_param_bounds: &ty::ParamBounds,
                            ty: ty::t,
                            is_early: bool) -> vtable_param_res {
    let tcx = vcx.tcx();

    // ty is the value supplied for the type parameter A...
    let mut param_result = Vec::new();

    ty::each_bound_trait_and_supertraits(tcx,
                                         type_param_bounds.trait_bounds
                                                          .as_slice(),
                                         |trait_ref| {
        // ...and here trait_ref is each bound that was declared on A,
        // expressed in terms of the type parameters.

        ty::populate_implementations_for_trait_if_necessary(tcx,
                                                            trait_ref.def_id);

        // Substitute the values of the type parameters that may
        // appear in the bound.
        let trait_ref = substs.as_ref().map_or(trait_ref.clone(), |substs| {
            debug!("about to subst: {}, {}",
                   trait_ref.repr(tcx), substs.repr(tcx));
            trait_ref.subst(tcx, *substs)
        });

        debug!("after subst: {}", trait_ref.repr(tcx));

        match lookup_vtable(vcx, span, ty, trait_ref.clone(), is_early) {
            Some(vtable) => param_result.push(vtable),
            None => {
                vcx.tcx().sess.span_fatal(span,
                    format!("failed to find an implementation of \
                          trait {} for {}",
                         vcx.infcx.trait_ref_to_str(&*trait_ref),
                         vcx.infcx.ty_to_str(ty)));
            }
        }
        true
    });

    debug!("lookup_vtables_for_param result(\
            span={:?}, \
            type_param_bounds={}, \
            ty={}, \
            result={})",
           span,
           type_param_bounds.repr(vcx.tcx()),
           ty.repr(vcx.tcx()),
           param_result.repr(vcx.tcx()));

    param_result
}

fn relate_trait_refs(vcx: &VtableContext,
                     span: Span,
                     act_trait_ref: Rc<ty::TraitRef>,
                     exp_trait_ref: Rc<ty::TraitRef>) {
    /*!
     *
     * Checks that an implementation of `act_trait_ref` is suitable
     * for use where `exp_trait_ref` is required and reports an
     * error otherwise.
     */

    match infer::mk_sub_trait_refs(vcx.infcx,
                                   false,
                                   infer::RelateTraitRefs(span),
                                   act_trait_ref.clone(),
                                   exp_trait_ref.clone()) {
        Ok(()) => {} // Ok.
        Err(ref err) => {
            // There is an error, but we need to do some work to make
            // the message good.
            // Resolve any type vars in the trait refs
            let r_act_trait_ref =
                vcx.infcx.resolve_type_vars_in_trait_ref_if_possible(&*act_trait_ref);
            let r_exp_trait_ref =
                vcx.infcx.resolve_type_vars_in_trait_ref_if_possible(&*exp_trait_ref);
            // Only print the message if there aren't any previous type errors
            // inside the types.
            if !ty::trait_ref_contains_error(&r_act_trait_ref) &&
                !ty::trait_ref_contains_error(&r_exp_trait_ref)
            {
                let tcx = vcx.tcx();
                tcx.sess.span_err(span,
                    format!("expected {}, but found {} ({})",
                         ppaux::trait_ref_to_str(tcx, &r_exp_trait_ref),
                         ppaux::trait_ref_to_str(tcx, &r_act_trait_ref),
                         ty::type_err_to_str(tcx, err)));
            }
        }
    }
}

// Look up the vtable implementing the trait `trait_ref` at type `t`
fn lookup_vtable(vcx: &VtableContext,
                 span: Span,
                 ty: ty::t,
                 trait_ref: Rc<ty::TraitRef>,
                 is_early: bool)
                 -> Option<vtable_origin> {
    debug!("lookup_vtable(ty={}, trait_ref={})",
           vcx.infcx.ty_to_str(ty),
           vcx.infcx.trait_ref_to_str(&*trait_ref));
    let _i = indenter();

    let ty = match fixup_ty(vcx, span, ty, is_early) {
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
        ty::ty_param(param_ty {idx: n, ..}) => {
            let env_bounds = &vcx.param_env.type_param_bounds;
            if env_bounds.len() > n {
                let type_param_bounds: &[Rc<ty::TraitRef>] =
                    env_bounds.get(n).trait_bounds.as_slice();
                lookup_vtable_from_bounds(vcx, span,
                                          type_param_bounds,
                                          param_numbered(n),
                                          trait_ref.clone())
            } else {
                None
            }
        }

        ty::ty_self(_) => {
            let self_param_bound = vcx.param_env.self_param_bound.clone().unwrap();
            lookup_vtable_from_bounds(vcx, span,
                                      [self_param_bound],
                                      param_self,
                                      trait_ref.clone())
        }

        // Default case just falls through
        _ => None
    };

    if vtable_opt.is_some() { return vtable_opt; }

    // If we aren't a self type or param, or it was, but we didn't find it,
    // do a search.
    search_for_vtable(vcx, span, ty, trait_ref, is_early)
}

// Given a list of bounds on a type, search those bounds to see if any
// of them are the vtable we are looking for.
fn lookup_vtable_from_bounds(vcx: &VtableContext,
                             span: Span,
                             bounds: &[Rc<ty::TraitRef>],
                             param: param_index,
                             trait_ref: Rc<ty::TraitRef>)
                             -> Option<vtable_origin> {
    let tcx = vcx.tcx();

    let mut n_bound = 0;
    let mut ret = None;
    ty::each_bound_trait_and_supertraits(tcx, bounds, |bound_trait_ref| {
        debug!("checking bounds trait {}",
               bound_trait_ref.repr(vcx.tcx()));

        if bound_trait_ref.def_id == trait_ref.def_id {
            relate_trait_refs(vcx, span, bound_trait_ref, trait_ref.clone());
            let vtable = vtable_param(param, n_bound);
            debug!("found param vtable: {:?}",
                   vtable);
            ret = Some(vtable);
            false
        } else {
            n_bound += 1;
            true
        }
    });
    ret
}

fn search_for_vtable(vcx: &VtableContext,
                     span: Span,
                     ty: ty::t,
                     trait_ref: Rc<ty::TraitRef>,
                     is_early: bool)
                     -> Option<vtable_origin> {
    let tcx = vcx.tcx();

    let mut found = Vec::new();
    let mut impls_seen = HashSet::new();

    // Load the implementations from external metadata if necessary.
    ty::populate_implementations_for_trait_if_necessary(tcx,
                                                        trait_ref.def_id);

    let impls = match tcx.trait_impls.borrow().find_copy(&trait_ref.def_id) {
        Some(impls) => impls,
        None => {
            return None;
        }
    };
    // impls is the list of all impls in scope for trait_ref.
    for &impl_did in impls.borrow().iter() {
        // im is one specific impl of trait_ref.

        // First, ensure we haven't processed this impl yet.
        if impls_seen.contains(&impl_did) {
            continue;
        }
        impls_seen.insert(impl_did);

        // ty::impl_traits gives us the trait im implements.
        //
        // If foo implements a trait t, and if t is the same trait as
        // trait_ref, we need to unify it with trait_ref in order to
        // get all the ty vars sorted out.
        let r = ty::impl_trait_ref(tcx, impl_did);
        let of_trait_ref = r.expect("trait_ref missing on trait impl");
        if of_trait_ref.def_id != trait_ref.def_id { continue; }

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
        // FIXME: document a bit more what this means
        //
        // FIXME(#5781) this should be mk_eqty not mk_subty
        let ty::ty_param_substs_and_ty {
            substs: substs,
            ty: for_ty
        } = impl_self_ty(vcx, span, impl_did);
        match infer::mk_subty(vcx.infcx,
                              false,
                              infer::RelateSelfType(span),
                              ty,
                              for_ty) {
            Err(_) => continue,
            Ok(()) => ()
        }

        // Now, in the previous example, for_ty is bound to
        // the type self_ty, and substs is bound to [T].
        debug!("The self ty is {} and its substs are {}",
               vcx.infcx.ty_to_str(for_ty),
               vcx.infcx.tys_to_str(substs.tps.as_slice()));

        // Next, we unify trait_ref -- the type that we want to cast
        // to -- with of_trait_ref -- the trait that im implements. At
        // this point, we require that they be unifiable with each
        // other -- that's what relate_trait_refs does.
        //
        // For example, in the above example, of_trait_ref would be
        // some_trait<T>, so we would be unifying trait_ref<U> (for
        // some value of U) with some_trait<T>. This would fail if T
        // and U weren't compatible.

        debug!("(checking vtable) \\#2 relating trait \
                ty {} to of_trait_ref {}",
               vcx.infcx.trait_ref_to_str(&*trait_ref),
               vcx.infcx.trait_ref_to_str(&*of_trait_ref));

        let of_trait_ref = of_trait_ref.subst(tcx, &substs);
        relate_trait_refs(vcx, span, of_trait_ref, trait_ref.clone());


        // Recall that trait_ref -- the trait type we're casting to --
        // is the trait with id trait_ref.def_id applied to the substs
        // trait_ref.substs.

        // Resolve any sub bounds. Note that there still may be free
        // type variables in substs. This might still be OK: the
        // process of looking up bounds might constrain some of them.
        let im_generics =
            ty::lookup_item_type(tcx, impl_did).generics;
        let subres = lookup_vtables(vcx, span,
                                    im_generics.type_param_defs(), &substs,
                                    is_early);


        // substs might contain type variables, so we call
        // fixup_substs to resolve them.
        let substs_f = match fixup_substs(vcx, span,
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

        debug!("The fixed-up substs are {} - \
                they will be unified with the bounds for \
                the target ty, {}",
               vcx.infcx.tys_to_str(substs_f.tps.as_slice()),
               vcx.infcx.trait_ref_to_str(&*trait_ref));

        // Next, we unify the fixed-up substitutions for the impl self
        // ty with the substitutions from the trait type that we're
        // trying to cast to. connect_trait_tps requires these lists
        // of types to unify pairwise.
        // I am a little confused about this, since it seems to be
        // very similar to the relate_trait_refs we already do,
        // but problems crop up if it is removed, so... -sully
        connect_trait_tps(vcx, span, &substs_f, trait_ref.clone(), impl_did);

        // Finally, we register that we found a matching impl, and
        // record the def ID of the impl as well as the resolved list
        // of type substitutions for the target trait.
        found.push(vtable_static(impl_did, substs_f.tps.clone(), subres));
    }

    match found.len() {
        0 => { return None }
        1 => return Some(found.get(0).clone()),
        _ => {
            if !is_early {
                vcx.tcx().sess.span_err(span, "multiple applicable methods in scope");
            }
            return Some(found.get(0).clone());
        }
    }
}


fn fixup_substs(vcx: &VtableContext,
                span: Span,
                id: ast::DefId,
                substs: ty::substs,
                is_early: bool)
                -> Option<ty::substs> {
    let tcx = vcx.tcx();
    // use a dummy type just to package up the substs that need fixing up
    let t = ty::mk_trait(tcx,
                         id, substs,
                         ty::RegionTraitStore(ty::ReStatic, ast::MutImmutable),
                         ty::EmptyBuiltinBounds());
    fixup_ty(vcx, span, t, is_early).map(|t_f| {
        match ty::get(t_f).sty {
          ty::ty_trait(ref inner) => inner.substs.clone(),
          _ => fail!("t_f should be a trait")
        }
    })
}

fn fixup_ty(vcx: &VtableContext,
            span: Span,
            ty: ty::t,
            is_early: bool)
            -> Option<ty::t> {
    let tcx = vcx.tcx();
    match resolve_type(vcx.infcx, ty, resolve_and_force_all_but_regions) {
        Ok(new_type) => Some(new_type),
        Err(e) if !is_early => {
            tcx.sess.span_fatal(span,
                format!("cannot determine a type \
                      for this bounded type parameter: {}",
                     fixup_err_to_str(e)))
        }
        Err(_) => {
            None
        }
    }
}

fn connect_trait_tps(vcx: &VtableContext,
                     span: Span,
                     impl_substs: &ty::substs,
                     trait_ref: Rc<ty::TraitRef>,
                     impl_did: ast::DefId) {
    let tcx = vcx.tcx();

    let impl_trait_ref = match ty::impl_trait_ref(tcx, impl_did) {
        Some(t) => t,
        None => vcx.tcx().sess.span_bug(span,
                                  "connect_trait_tps invoked on a type impl")
    };

    let impl_trait_ref = impl_trait_ref.subst(tcx, impl_substs);
    relate_trait_refs(vcx, span, impl_trait_ref, trait_ref);
}

fn insert_vtables(fcx: &FnCtxt, vtable_key: MethodCall, vtables: vtable_res) {
    debug!("insert_vtables(vtable_key={}, vtables={:?})",
           vtable_key, vtables.repr(fcx.tcx()));
    fcx.inh.vtable_map.borrow_mut().insert(vtable_key, vtables);
}

pub fn early_resolve_expr(ex: &ast::Expr, fcx: &FnCtxt, is_early: bool) {
    debug!("vtable: early_resolve_expr() ex with id {:?} (early: {}): {}",
           ex.id, is_early, expr_to_str(ex));
    let _indent = indenter();

    let cx = fcx.ccx;
    let resolve_object_cast = |src: &ast::Expr, target_ty: ty::t| {
      match ty::get(target_ty).sty {
          // Bounds of type's contents are not checked here, but in kind.rs.
          ty::ty_trait(~ty::TyTrait {
              def_id: target_def_id, substs: ref target_substs, store, ..
          }) => {
              fn mutability_allowed(a_mutbl: ast::Mutability,
                                    b_mutbl: ast::Mutability) -> bool {
                  a_mutbl == b_mutbl ||
                  (a_mutbl == ast::MutMutable && b_mutbl == ast::MutImmutable)
              }
              // Look up vtables for the type we're casting to,
              // passing in the source and target type.  The source
              // must be a pointer type suitable to the object sigil,
              // e.g.: `&x as &Trait` or `~x as ~Trait`
              let ty = structurally_resolved_type(fcx, ex.span,
                                                  fcx.expr_ty(src));
              match (&ty::get(ty).sty, store) {
                  (&ty::ty_rptr(_, mt), ty::RegionTraitStore(_, mutbl))
                    if !mutability_allowed(mt.mutbl, mutbl) => {
                      fcx.tcx().sess.span_err(ex.span,
                                              format!("types differ in mutability"));
                  }

                  (&ty::ty_uniq(..), ty::UniqTraitStore) |
                  (&ty::ty_rptr(..), ty::RegionTraitStore(..)) => {
                    let typ = match &ty::get(ty).sty {
                        &ty::ty_box(typ) | &ty::ty_uniq(typ) => typ,
                        &ty::ty_rptr(_, mt) => mt.ty,
                        _ => fail!("shouldn't get here"),
                    };

                      let vcx = fcx.vtable_context();
                      let target_trait_ref = Rc::new(ty::TraitRef {
                          def_id: target_def_id,
                          substs: ty::substs {
                              tps: target_substs.tps.clone(),
                              regions: target_substs.regions.clone(),
                              self_ty: Some(typ)
                          }
                      });

                      let param_bounds = ty::ParamBounds {
                          builtin_bounds: ty::EmptyBuiltinBounds(),
                          trait_bounds: vec!(target_trait_ref)
                      };
                      let vtables =
                            lookup_vtables_for_param(&vcx,
                                                     ex.span,
                                                     None,
                                                     &param_bounds,
                                                     typ,
                                                     is_early);

                      if !is_early {
                          insert_vtables(fcx, MethodCall::expr(ex.id), vec!(vtables));
                      }

                      // Now, if this is &trait, we need to link the
                      // regions.
                      match (&ty::get(ty).sty, store) {
                          (&ty::ty_rptr(ra, _),
                           ty::RegionTraitStore(rb, _)) => {
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
                          format!("can only cast an ~-pointer \
                                to a ~-object, not a {}",
                               ty::ty_sort_str(fcx.tcx(), ty)));
                  }

                  (_, ty::RegionTraitStore(..)) => {
                      fcx.ccx.tcx.sess.span_err(
                          ex.span,
                          format!("can only cast an &-pointer \
                                to an &-object, not a {}",
                               ty::ty_sort_str(fcx.tcx(), ty)));
                  }
              }
          }
          _ => { /* not a cast to a trait; ignore */ }
      }
    };
    match ex.node {
      ast::ExprPath(..) => {
        fcx.opt_node_ty_substs(ex.id, |substs| {
            debug!("vtable resolution on parameter bounds for expr {}",
                   ex.repr(fcx.tcx()));
            let def = cx.tcx.def_map.borrow().get_copy(&ex.id);
            let did = ast_util::def_id_of_def(def);
            let item_ty = ty::lookup_item_type(cx.tcx, did);
            debug!("early resolve expr: def {:?} {:?}, {:?}, {}", ex.id, did, def,
                   fcx.infcx().ty_to_str(item_ty.ty));
            if has_trait_bounds(item_ty.generics.type_param_defs()) {
                debug!("early_resolve_expr: looking up vtables for type params {}",
                       item_ty.generics.type_param_defs().repr(fcx.tcx()));
                let vcx = fcx.vtable_context();
                let vtbls = lookup_vtables(&vcx, ex.span,
                                           item_ty.generics.type_param_defs(),
                                           substs, is_early);
                if !is_early {
                    insert_vtables(fcx, MethodCall::expr(ex.id), vtbls);
                }
            }
            true
        });
      }

      // Must resolve bounds on methods with bounded params
      ast::ExprBinary(_, _, _) |
      ast::ExprUnary(_, _) |
      ast::ExprAssignOp(_, _, _) |
      ast::ExprIndex(_, _) |
      ast::ExprMethodCall(_, _, _) => {
        match fcx.inh.method_map.borrow().find(&MethodCall::expr(ex.id)) {
          Some(method) => {
            debug!("vtable resolution on parameter bounds for method call {}",
                   ex.repr(fcx.tcx()));
            let type_param_defs = ty::method_call_type_param_defs(cx.tcx, method.origin);
            if has_trait_bounds(type_param_defs.as_slice()) {
                let substs = fcx.method_ty_substs(ex.id);
                let vcx = fcx.vtable_context();
                let vtbls = lookup_vtables(&vcx, ex.span,
                                           type_param_defs.as_slice(),
                                           &substs, is_early);
                if !is_early {
                    insert_vtables(fcx, MethodCall::expr(ex.id), vtbls);
                }
            }
          }
          None => {}
        }
      }
      ast::ExprCast(src, _) => {
          debug!("vtable resolution on expr {}", ex.repr(fcx.tcx()));
          let target_ty = fcx.expr_ty(ex);
          resolve_object_cast(src, target_ty);
      }
      _ => ()
    }

    // Search for auto-adjustments to find trait coercions
    match fcx.inh.adjustments.borrow().find(&ex.id) {
        Some(adjustment) => {
            match *adjustment {
                AutoDerefRef(adj) => {
                    for autoderef in range(0, adj.autoderefs) {
                        let method_call = MethodCall::autoderef(ex.id, autoderef as u32);
                        match fcx.inh.method_map.borrow().find(&method_call) {
                            Some(method) => {
                                debug!("vtable resolution on parameter bounds for autoderef {}",
                                       ex.repr(fcx.tcx()));
                                let type_param_defs =
                                    ty::method_call_type_param_defs(cx.tcx, method.origin);
                                if has_trait_bounds(type_param_defs.deref().as_slice()) {
                                    let vcx = fcx.vtable_context();
                                    let vtbls = lookup_vtables(&vcx, ex.span,
                                                               type_param_defs.deref()
                                                               .as_slice(),
                                                               &method.substs, is_early);
                                    if !is_early {
                                        insert_vtables(fcx, method_call, vtbls);
                                    }
                                }
                            }
                            None => {}
                        }
                    }
                }
                AutoObject(store,
                           bounds,
                           def_id,
                           ref substs) => {
                    debug!("doing trait adjustment for expr {} {} \
                            (early? {})",
                           ex.id,
                           ex.repr(fcx.tcx()),
                           is_early);

                    let object_ty = ty::mk_trait(cx.tcx, def_id,
                                                 substs.clone(),
                                                 store, bounds);
                    resolve_object_cast(ex, object_ty);
                }
                AutoAddEnv(..) => {}
            }
        }
        None => {}
    }
}

pub fn resolve_impl(tcx: &ty::ctxt,
                    impl_item: &ast::Item,
                    impl_generics: &ty::Generics,
                    impl_trait_ref: &ty::TraitRef) {
    let param_env = ty::construct_parameter_environment(
        tcx,
        None,
        impl_generics.type_param_defs(),
        [],
        impl_generics.region_param_defs(),
        [],
        impl_item.id);

    let impl_trait_ref = impl_trait_ref.subst(tcx, &param_env.free_substs);

    let infcx = &infer::new_infer_ctxt(tcx);
    let vcx = VtableContext { infcx: infcx, param_env: &param_env };

    // First, check that the impl implements any trait bounds
    // on the trait.
    let trait_def = ty::lookup_trait_def(tcx, impl_trait_ref.def_id);
    let vtbls = lookup_vtables(&vcx, impl_item.span,
                               trait_def.generics.type_param_defs(),
                               &impl_trait_ref.substs,
                               false);

    // Now, locate the vtable for the impl itself. The real
    // purpose of this is to check for supertrait impls,
    // but that falls out of doing this.
    let param_bounds = ty::ParamBounds {
        builtin_bounds: ty::EmptyBuiltinBounds(),
        trait_bounds: vec!(Rc::new(impl_trait_ref))
    };
    let t = ty::node_id_to_type(tcx, impl_item.id);
    let t = t.subst(tcx, &param_env.free_substs);
    debug!("=== Doing a self lookup now.");

    // Right now, we don't have any place to store this.
    // We will need to make one so we can use this information
    // for compiling default methods that refer to supertraits.
    let self_vtable_res =
        lookup_vtables_for_param(&vcx, impl_item.span, None,
                                 &param_bounds, t, false);


    let res = impl_res {
        trait_vtables: vtbls,
        self_vtables: self_vtable_res
    };
    let impl_def_id = ast_util::local_def(impl_item.id);

    tcx.impl_vtables.borrow_mut().insert(impl_def_id, res);
}

/// Resolve vtables for a method call after typeck has finished.
/// Used by trans to monomorphize artificial method callees (e.g. drop).
pub fn trans_resolve_method(tcx: &ty::ctxt, id: ast::NodeId,
                            substs: &ty::substs) -> Option<vtable_res> {
    let generics = ty::lookup_item_type(tcx, ast_util::local_def(id)).generics;
    let type_param_defs = &*generics.type_param_defs;
    if has_trait_bounds(type_param_defs.as_slice()) {
        let vcx = VtableContext {
            infcx: &infer::new_infer_ctxt(tcx),
            param_env: &ty::construct_parameter_environment(tcx, None, [], [], [], [], id)
        };

        Some(lookup_vtables(&vcx,
                            tcx.map.span(id),
                            type_param_defs.as_slice(),
                            substs,
                            false))
    } else {
        None
    }
}

impl<'a, 'b> visit::Visitor<()> for &'a FnCtxt<'b> {
    fn visit_expr(&mut self, ex: &ast::Expr, _: ()) {
        early_resolve_expr(ex, *self, false);
        visit::walk_expr(self, ex, ());
    }
    fn visit_item(&mut self, _: &ast::Item, _: ()) {
        // no-op
    }
}

// Detect points where a trait-bounded type parameter is
// instantiated, resolve the impls for the parameters.
pub fn resolve_in_block(mut fcx: &FnCtxt, bl: &ast::Block) {
    visit::walk_block(&mut fcx, bl, ());
}
