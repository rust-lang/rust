// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use middle::ty;
use middle::ty::{AutoDerefRef, ParamTy};
use middle::ty_fold::TypeFolder;
use middle::typeck::astconv::AstConv;
use middle::typeck::check::{FnCtxt, impl_self_ty};
use middle::typeck::check::{structurally_resolved_type};
use middle::typeck::check::regionmanip;
use middle::typeck::check::writeback;
use middle::typeck::infer::fixup_err_to_string;
use middle::typeck::infer::{resolve_and_force_all_but_regions, resolve_type};
use middle::typeck::infer;
use middle::typeck::{MethodCall, TypeAndSubsts};
use middle::typeck::{param_index, vtable_error, vtable_origin, vtable_param};
use middle::typeck::{vtable_param_res, vtable_res, vtable_static};
use middle::typeck::{vtable_unboxed_closure};
use middle::subst;
use middle::subst::{Subst, VecPerParamSpace};
use util::common::indenter;
use util::nodemap::DefIdMap;
use util::ppaux;
use util::ppaux::Repr;

use std::cell::RefCell;
use std::rc::Rc;
use std::collections::HashSet;
use syntax::ast;
use syntax::ast_util;
use syntax::codemap::Span;
use syntax::print::pprust::expr_to_string;
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

/// A vtable context includes an inference context, a parameter environment,
/// and a list of unboxed closure types.
pub struct VtableContext<'a> {
    pub infcx: &'a infer::InferCtxt<'a>,
    pub param_env: &'a ty::ParameterEnvironment,
    pub unboxed_closures: &'a RefCell<DefIdMap<ty::UnboxedClosure>>,
}

impl<'a> VtableContext<'a> {
    pub fn tcx(&self) -> &'a ty::ctxt { self.infcx.tcx }
}

fn lookup_vtables(vcx: &VtableContext,
                  span: Span,
                  type_param_defs: &VecPerParamSpace<ty::TypeParameterDef>,
                  substs: &subst::Substs,
                  is_early: bool)
                  -> VecPerParamSpace<vtable_param_res> {
    debug!("lookup_vtables(\
           type_param_defs={}, \
           substs={}",
           type_param_defs.repr(vcx.tcx()),
           substs.repr(vcx.tcx()));

    // We do this backwards for reasons discussed above.
    let result = type_param_defs.map_rev(|def| {
        let ty = *substs.types.get(def.space, def.index);
        lookup_vtables_for_param(vcx, span, Some(substs),
                                 &*def.bounds, ty, is_early)
    });

    debug!("lookup_vtables result(\
            type_param_defs={}, \
            substs={}, \
            result={})",
           type_param_defs.repr(vcx.tcx()),
           substs.repr(vcx.tcx()),
           result.repr(vcx.tcx()));

    result
}

fn lookup_vtables_for_param(vcx: &VtableContext,
                            span: Span,
                            // None for substs means the identity
                            substs: Option<&subst::Substs>,
                            type_param_bounds: &ty::ParamBounds,
                            ty: ty::t,
                            is_early: bool)
                            -> vtable_param_res {
    let tcx = vcx.tcx();

    debug!("lookup_vtables_for_param(ty={}, type_param_bounds={}, is_early={})",
           ty.repr(vcx.tcx()),
           type_param_bounds.repr(vcx.tcx()),
           is_early);

    // ty is the value supplied for the type parameter A...
    let mut param_result = Vec::new();

    ty::each_bound_trait_and_supertraits(tcx,
                                         type_param_bounds.trait_bounds
                                                          .as_slice(),
                                         |trait_ref| {
        // ...and here trait_ref is each bound that was declared on A,
        // expressed in terms of the type parameters.

        debug!("matching ty={} trait_ref={}",
               ty.repr(vcx.tcx()),
               trait_ref.repr(vcx.tcx()));

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
                vcx.tcx().sess.span_err(span,
                    format!("failed to find an implementation of \
                          trait {} for {}",
                         vcx.infcx.trait_ref_to_string(&*trait_ref),
                         vcx.infcx.ty_to_string(ty)).as_slice());
                param_result.push(vtable_error)
            }
        }
        true
    });

    debug!("lookup_vtables_for_param result(\
            type_param_bounds={}, \
            ty={}, \
            result={})",
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
                span_err!(tcx.sess, span, E0095, "expected {}, found {} ({})",
                          ppaux::trait_ref_to_string(tcx, &r_exp_trait_ref),
                          ppaux::trait_ref_to_string(tcx, &r_act_trait_ref),
                          ty::type_err_to_str(tcx, err));
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
                 -> Option<vtable_origin>
{
    debug!("lookup_vtable(ty={}, trait_ref={})",
           ty.repr(vcx.tcx()),
           trait_ref.repr(vcx.tcx()));
    let _i = indenter();

    let ty = match fixup_ty(vcx, span, ty, is_early) {
        Some(ty) => ty,
        None => {
            // fixup_ty can only fail if this is early resolution
            assert!(is_early);
            // The type has unconstrained type variables in it, so we can't
            // do early resolution on it. Return some completely bogus vtable
            // information: we aren't storing it anyways.
            return Some(vtable_error);
        }
    };

    if ty::type_is_error(ty) {
        return Some(vtable_error);
    }

    // If the type is self or a param, we look at the trait/supertrait
    // bounds to see if they include the trait we are looking for.
    let vtable_opt = match ty::get(ty).sty {
        ty::ty_param(ParamTy {space, idx: n, ..}) => {
            let env_bounds = &vcx.param_env.bounds;
            let type_param_bounds = &env_bounds.get(space, n).trait_bounds;
            lookup_vtable_from_bounds(vcx,
                                      span,
                                      type_param_bounds.as_slice(),
                                      param_index {
                                          space: space,
                                          index: n,
                                      },
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

fn search_for_unboxed_closure_vtable(vcx: &VtableContext,
                                     span: Span,
                                     ty: ty::t,
                                     trait_ref: Rc<ty::TraitRef>)
                                     -> Option<vtable_origin> {
    let tcx = vcx.tcx();
    let closure_def_id = match ty::get(ty).sty {
        ty::ty_unboxed_closure(closure_def_id, _) => closure_def_id,
        _ => return None,
    };

    let fn_traits = [
        (ty::FnUnboxedClosureKind, tcx.lang_items.fn_trait()),
        (ty::FnMutUnboxedClosureKind, tcx.lang_items.fn_mut_trait()),
        (ty::FnOnceUnboxedClosureKind, tcx.lang_items.fn_once_trait()),
    ];
    for tuple in fn_traits.iter() {
        let kind = match tuple {
            &(kind, Some(ref fn_trait)) if *fn_trait == trait_ref.def_id => {
                kind
            }
            _ => continue,
        };

        // Check to see whether the argument and return types match.
        let unboxed_closures = tcx.unboxed_closures.borrow();
        let closure_type = match unboxed_closures.find(&closure_def_id) {
            Some(closure) => {
                if closure.kind != kind {
                    continue
                }
                closure.closure_type.clone()
            }
            None => {
                // Try the inherited unboxed closure type map.
                let unboxed_closures = vcx.unboxed_closures.borrow();
                match unboxed_closures.find(&closure_def_id) {
                    Some(closure) => {
                        if closure.kind != kind {
                            continue
                        }
                        closure.closure_type.clone()
                    }
                    None => {
                        tcx.sess.span_bug(span,
                                          "didn't find unboxed closure type \
                                           in tcx map or inh map")
                    }
                }
            }
        };

        // FIXME(pcwalton): This is a bogus thing to do, but
        // it'll do for now until we get the new trait-bound
        // region skolemization working.
        let (_, new_signature) =
            regionmanip::replace_late_bound_regions_in_fn_sig(
                tcx,
                &closure_type.sig,
                |br| {
                    vcx.infcx.next_region_var(infer::LateBoundRegion(span,
                                                                     br))
                });

        let arguments_tuple = *new_signature.inputs.get(0);
        let corresponding_trait_ref = Rc::new(ty::TraitRef {
            def_id: trait_ref.def_id,
            substs: subst::Substs::new_trait(
                vec![arguments_tuple, new_signature.output],
                Vec::new(),
                ty)
        });

        relate_trait_refs(vcx, span, corresponding_trait_ref, trait_ref);
        return Some(vtable_unboxed_closure(closure_def_id))
    }

    None
}

fn search_for_vtable(vcx: &VtableContext,
                     span: Span,
                     ty: ty::t,
                     trait_ref: Rc<ty::TraitRef>,
                     is_early: bool)
                     -> Option<vtable_origin> {
    let tcx = vcx.tcx();

    // First, check to see whether this is a call to the `call` method of an
    // unboxed closure. If so, and the arguments match, we're done.
    match search_for_unboxed_closure_vtable(vcx,
                                            span,
                                            ty,
                                            trait_ref.clone()) {
        Some(vtable_origin) => return Some(vtable_origin),
        None => {}
    }

    // Nope. Continue.

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
        let TypeAndSubsts {
            substs: substs,
            ty: for_ty
        } = impl_self_ty(vcx, span, impl_did);
        match infer::mk_eqty(vcx.infcx,
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
               for_ty.repr(tcx),
               substs.types.repr(tcx));

        // Next, we unify trait_ref -- the type that we want to cast
        // to -- with of_trait_ref -- the trait that im implements. At
        // this point, we require that they be unifiable with each
        // other -- that's what relate_trait_refs does.
        //
        // For example, in the above example, of_trait_ref would be
        // some_trait<T>, so we would be unifying trait_ref<U> (for
        // some value of U) with some_trait<T>. This would fail if T
        // and U weren't compatible.

        let of_trait_ref = of_trait_ref.subst(tcx, &substs);

        debug!("(checking vtable) num 2 relating trait \
                ty {} to of_trait_ref {}",
               vcx.infcx.trait_ref_to_string(&*trait_ref),
               vcx.infcx.trait_ref_to_string(&*of_trait_ref));

        relate_trait_refs(vcx, span, of_trait_ref, trait_ref.clone());


        // Recall that trait_ref -- the trait type we're casting to --
        // is the trait with id trait_ref.def_id applied to the substs
        // trait_ref.substs.

        // Resolve any sub bounds. Note that there still may be free
        // type variables in substs. This might still be OK: the
        // process of looking up bounds might constrain some of them.
        //
        // This does not check built-in traits because those are handled
        // later in the kind checking pass.
        let im_generics =
            ty::lookup_item_type(tcx, impl_did).generics;
        let subres = lookup_vtables(vcx,
                                    span,
                                    &im_generics.types,
                                    &substs,
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
                return Some(vtable_error);
            }
        };

        debug!("The fixed-up substs are {} - \
                they will be unified with the bounds for \
                the target ty, {}",
               substs_f.types.repr(tcx),
               trait_ref.repr(tcx));

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
        found.push(vtable_static(impl_did, substs_f, subres));
    }

    match found.len() {
        0 => { return None }
        1 => return Some(found.get(0).clone()),
        _ => {
            if !is_early {
                span_err!(vcx.tcx().sess, span, E0096,
                          "multiple applicable methods in scope");
            }
            return Some(found.get(0).clone());
        }
    }
}


fn fixup_substs(vcx: &VtableContext,
                span: Span,
                id: ast::DefId,
                substs: subst::Substs,
                is_early: bool)
                -> Option<subst::Substs> {
    let tcx = vcx.tcx();
    // use a dummy type just to package up the substs that need fixing up
    let t = ty::mk_trait(tcx,
                         id, substs,
                         ty::empty_builtin_bounds());
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
    match resolve_type(vcx.infcx, Some(span), ty, resolve_and_force_all_but_regions) {
        Ok(new_type) => Some(new_type),
        Err(e) if !is_early => {
            tcx.sess.span_err(span,
                format!("cannot determine a type for this bounded type \
                         parameter: {}",
                        fixup_err_to_string(e)).as_slice());
            Some(ty::mk_err())
        }
        Err(_) => {
            None
        }
    }
}

fn connect_trait_tps(vcx: &VtableContext,
                     span: Span,
                     impl_substs: &subst::Substs,
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
    debug!("insert_vtables(vtable_key={}, vtables={})",
           vtable_key, vtables.repr(fcx.tcx()));
    fcx.inh.vtable_map.borrow_mut().insert(vtable_key, vtables);
}

pub fn early_resolve_expr(ex: &ast::Expr, fcx: &FnCtxt, is_early: bool) {
    fn mutability_allowed(a_mutbl: ast::Mutability,
                          b_mutbl: ast::Mutability) -> bool {
        a_mutbl == b_mutbl ||
        (a_mutbl == ast::MutMutable && b_mutbl == ast::MutImmutable)
    }

    debug!("vtable: early_resolve_expr() ex with id {:?} (early: {}): {}",
           ex.id, is_early, expr_to_string(ex));
    let _indent = indenter();

    let cx = fcx.ccx;
    let check_object_cast = |src_ty: ty::t, target_ty: ty::t| {
      // Check that a cast is of correct types.
      match (&ty::get(target_ty).sty, &ty::get(src_ty).sty) {
          (&ty::ty_rptr(_, ty::mt{ty, mutbl}), &ty::ty_rptr(_, mt))
            if !mutability_allowed(mt.mutbl, mutbl) => {
              match ty::get(ty).sty {
                  ty::ty_trait(..) => {
                      span_err!(fcx.tcx().sess, ex.span, E0097, "types differ in mutability");
                  }
                  _ => {}
              }
          }
          (&ty::ty_uniq(..), &ty::ty_uniq(..) ) => {}
          (&ty::ty_rptr(r_t, _), &ty::ty_rptr(r_s, _)) => {
              infer::mk_subr(fcx.infcx(),
                             false,
                             infer::RelateObjectBound(ex.span),
                             r_t,
                             r_s);
          }
          (&ty::ty_uniq(ty), _) => {
              match ty::get(ty).sty {
                  ty::ty_trait(..) => {
                      span_err!(fcx.ccx.tcx.sess, ex.span, E0098,
                                "can only cast an boxed pointer to a boxed object, not a {}",
                                ty::ty_sort_string(fcx.tcx(), src_ty));
                  }
                  _ => {}
              }

          }
          (&ty::ty_rptr(_, ty::mt{ty, ..}), _) => {
              match ty::get(ty).sty {
                  ty::ty_trait(..) => {
                      span_err!(fcx.ccx.tcx.sess, ex.span, E0099,
                                "can only cast an &-pointer to an &-object, not a {}",
                                ty::ty_sort_string(fcx.tcx(), src_ty));
                  }
                  _ => {}
              }
          }
          _ => {}
      }
    };
    let resolve_object_cast = |src_ty: ty::t, target_ty: ty::t, key: MethodCall| {
      // Look up vtables for the type we're casting to,
      // passing in the source and target type.  The source
      // must be a pointer type suitable to the object sigil,
      // e.g.: `&x as &Trait` or `box x as Box<Trait>`
      // Bounds of type's contents are not checked here, but in kind.rs.
      match ty::get(target_ty).sty {
          ty::ty_trait(box ty::TyTrait {
              def_id: target_def_id, substs: ref target_substs, ..
          }) => {
              let vcx = fcx.vtable_context();

              // Take the type parameters from the object
              // type, but set the Self type (which is
              // unknown, for the object type) to be the type
              // we are casting from.
              let mut target_types = target_substs.types.clone();
              assert!(target_types.get_self().is_none());
              target_types.push(subst::SelfSpace, src_ty);

              let target_trait_ref = Rc::new(ty::TraitRef {
                  def_id: target_def_id,
                  substs: subst::Substs {
                      regions: target_substs.regions.clone(),
                      types: target_types
                  }
              });

              let param_bounds = ty::ParamBounds {
                  builtin_bounds: ty::empty_builtin_bounds(),
                  trait_bounds: vec!(target_trait_ref)
              };
              let vtables =
                    lookup_vtables_for_param(&vcx,
                                             ex.span,
                                             None,
                                             &param_bounds,
                                             src_ty,
                                             is_early);

              if !is_early {
                  let mut r = VecPerParamSpace::empty();
                  r.push(subst::SelfSpace, vtables);
                  insert_vtables(fcx, key, r);
              }
          }
          _ => {}
      }
    };
    match ex.node {
      ast::ExprPath(..) => {
        fcx.opt_node_ty_substs(ex.id, |item_substs| {
            debug!("vtable resolution on parameter bounds for expr {}",
                   ex.repr(fcx.tcx()));
            let def = cx.tcx.def_map.borrow().get_copy(&ex.id);
            let did = def.def_id();
            let item_ty = ty::lookup_item_type(cx.tcx, did);
            debug!("early resolve expr: def {:?} {:?}, {:?}, {}", ex.id, did, def,
                   fcx.infcx().ty_to_string(item_ty.ty));
            debug!("early_resolve_expr: looking up vtables for type params {}",
                   item_ty.generics.types.repr(fcx.tcx()));
            let vcx = fcx.vtable_context();
            let vtbls = lookup_vtables(&vcx, ex.span,
                                       &item_ty.generics.types,
                                       &item_substs.substs, is_early);
            if !is_early {
                insert_vtables(fcx, MethodCall::expr(ex.id), vtbls);
            }
        });
      }

      // Must resolve bounds on methods with bounded params
      ast::ExprBinary(_, _, _) |
      ast::ExprUnary(_, _) |
      ast::ExprAssignOp(_, _, _) |
      ast::ExprIndex(_, _) |
      ast::ExprMethodCall(_, _, _) |
      ast::ExprForLoop(..) |
      ast::ExprCall(..) => {
        match fcx.inh.method_map.borrow().find(&MethodCall::expr(ex.id)) {
          Some(method) => {
              debug!("vtable resolution on parameter bounds for method call {}",
                     ex.repr(fcx.tcx()));
              let type_param_defs =
                  ty::method_call_type_param_defs(fcx, method.origin);
              let substs = fcx.method_ty_substs(ex.id);
              let vcx = fcx.vtable_context();
              let vtbls = lookup_vtables(&vcx, ex.span,
                                         &type_param_defs,
                                         &substs, is_early);
              if !is_early {
                  insert_vtables(fcx, MethodCall::expr(ex.id), vtbls);
              }
          }
          None => {}
        }
      }
      ast::ExprCast(ref src, _) => {
          debug!("vtable resolution on expr {}", ex.repr(fcx.tcx()));
          let target_ty = fcx.expr_ty(ex);
          let src_ty = structurally_resolved_type(fcx, ex.span,
                                                  fcx.expr_ty(&**src));
          check_object_cast(src_ty, target_ty);
          match (ty::deref(src_ty, false), ty::deref(target_ty, false)) {
              (Some(s), Some(t)) => {
                  let key = MethodCall::expr(ex.id);
                  resolve_object_cast(s.ty, t.ty, key)
              }
              _ => {}
          }
      }
      _ => ()
    }

    // Search for auto-adjustments to find trait coercions
    match fcx.inh.adjustments.borrow().find(&ex.id) {
        Some(adjustment) => {
            match *adjustment {
                _ if ty::adjust_is_object(adjustment) => {
                    let src_ty = structurally_resolved_type(fcx, ex.span,
                                                            fcx.expr_ty(ex));
                    match ty::type_of_adjust(fcx.tcx(), adjustment) {
                        Some(target_ty) => {
                            check_object_cast(src_ty, target_ty)
                        }
                        None => {}
                    }

                    match trait_cast_types(fcx, adjustment, src_ty, ex.span) {
                        Some((s, t)) => {
                            let key = MethodCall::autoobject(ex.id);
                            resolve_object_cast(s, t, key)
                        }
                        None => fail!("Couldn't extract types from adjustment")
                    }
                }
                AutoDerefRef(ref adj) => {
                    for autoderef in range(0, adj.autoderefs) {
                        let method_call = MethodCall::autoderef(ex.id, autoderef);
                        match fcx.inh.method_map.borrow().find(&method_call) {
                            Some(method) => {
                                debug!("vtable resolution on parameter bounds for autoderef {}",
                                       ex.repr(fcx.tcx()));
                                let type_param_defs =
                                    ty::method_call_type_param_defs(cx.tcx, method.origin);
                                let vcx = fcx.vtable_context();
                                let vtbls = lookup_vtables(&vcx, ex.span,
                                                           &type_param_defs,
                                                           &method.substs, is_early);
                                if !is_early {
                                    insert_vtables(fcx, method_call, vtbls);
                                }
                            }
                            None => {}
                        }
                    }
                }
                _ => {}
            }
        }
        None => {}
    }
}

// When we coerce (possibly implicitly) from a concrete type to a trait type, this
// function returns the concrete type and trait. This might happen arbitrarily
// deep in the adjustment. This function will fail if the adjustment does not
// match the source type.
// This function will always return types if ty::adjust_is_object is true for the
// adjustment
fn trait_cast_types(fcx: &FnCtxt,
                    adj: &ty::AutoAdjustment,
                    src_ty: ty::t,
                    sp: Span)
                    -> Option<(ty::t, ty::t)> {
    fn trait_cast_types_autoref(fcx: &FnCtxt,
                                autoref: &ty::AutoRef,
                                src_ty: ty::t,
                                sp: Span)
                                -> Option<(ty::t, ty::t)> {
        fn trait_cast_types_unsize(fcx: &FnCtxt,
                                   k: &ty::UnsizeKind,
                                   src_ty: ty::t,
                                   sp: Span)
                                   -> Option<(ty::t, ty::t)> {
            match k {
                &ty::UnsizeVtable(bounds, def_id, ref substs) => {
                    Some((src_ty, ty::mk_trait(fcx.tcx(), def_id, substs.clone(), bounds)))
                }
                &ty::UnsizeStruct(box ref k, tp_index) => match ty::get(src_ty).sty {
                    ty::ty_struct(_, ref substs) => {
                        let ty_substs = substs.types.get_slice(subst::TypeSpace);
                        let field_ty = structurally_resolved_type(fcx, sp, ty_substs[tp_index]);
                        trait_cast_types_unsize(fcx, k, field_ty, sp)
                    }
                    _ => fail!("Failed to find a ty_struct to correspond with \
                                UnsizeStruct whilst walking adjustment. Found {}",
                                ppaux::ty_to_string(fcx.tcx(), src_ty))
                },
                _ => None
            }
        }

        match autoref {
            &ty::AutoUnsize(ref k) |
            &ty::AutoUnsizeUniq(ref k) => trait_cast_types_unsize(fcx, k, src_ty, sp),
            &ty::AutoPtr(_, _, Some(box ref autoref)) => {
                trait_cast_types_autoref(fcx, autoref, src_ty, sp)
            }
            _ => None
        }
    }

    match adj {
        &ty::AutoDerefRef(AutoDerefRef{autoref: Some(ref autoref), autoderefs}) => {
            let mut derefed_type = src_ty;
            for _ in range(0, autoderefs) {
                derefed_type = ty::deref(derefed_type, false).unwrap().ty;
                derefed_type = structurally_resolved_type(fcx, sp, derefed_type)
            }
            trait_cast_types_autoref(fcx, autoref, derefed_type, sp)
        }
        _ => None
    }
}

pub fn resolve_impl(tcx: &ty::ctxt,
                    impl_item: &ast::Item,
                    impl_generics: &ty::Generics,
                    impl_trait_ref: &ty::TraitRef) {
    /*!
     * The situation is as follows. We have some trait like:
     *
     *    trait Foo<A:Clone> : Bar {
     *        fn method() { ... }
     *    }
     *
     * and an impl like:
     *
     *    impl<B:Clone> Foo<B> for int { ... }
     *
     * We want to validate that the various requirements of the trait
     * are met:
     *
     *    A:Clone, Self:Bar
     *
     * But of course after substituting the types from the impl:
     *
     *    B:Clone, int:Bar
     *
     * We store these results away as the "impl_res" for use by the
     * default methods.
     */

    debug!("resolve_impl(impl_item.id={})",
           impl_item.id);

    let param_env = ty::construct_parameter_environment(tcx,
                                                        impl_generics,
                                                        impl_item.id);

    // The impl_trait_ref in our example above would be
    //     `Foo<B> for int`
    let impl_trait_ref = impl_trait_ref.subst(tcx, &param_env.free_substs);
    debug!("impl_trait_ref={}", impl_trait_ref.repr(tcx));

    let infcx = &infer::new_infer_ctxt(tcx);
    let unboxed_closures = RefCell::new(DefIdMap::new());
    let vcx = VtableContext {
        infcx: infcx,
        param_env: &param_env,
        unboxed_closures: &unboxed_closures,
    };

    // Resolve the vtables for the trait reference on the impl.  This
    // serves many purposes, best explained by example. Imagine we have:
    //
    //    trait A<T:B> : C { fn x(&self) { ... } }
    //
    // and
    //
    //    impl A<int> for uint { ... }
    //
    // In that case, the trait ref will be `A<int> for uint`. Resolving
    // this will first check that the various types meet their requirements:
    //
    // 1. Because of T:B, int must implement the trait B
    // 2. Because of the supertrait C, uint must implement the trait C.
    //
    // Simultaneously, the result of this resolution (`vtbls`), is precisely
    // the set of vtable information needed to compile the default method
    // `x()` adapted to the impl. (After all, a default method is basically
    // the same as:
    //
    //     fn default_x<T:B, Self:A>(...) { .. .})

    let trait_def = ty::lookup_trait_def(tcx, impl_trait_ref.def_id);
    let vtbls = lookup_vtables(&vcx,
                                   impl_item.span,
                                   &trait_def.generics.types,
                                   &impl_trait_ref.substs,
                                   false);

    infcx.resolve_regions_and_report_errors();

    let vtbls = writeback::resolve_impl_res(infcx, impl_item.span, &vtbls);
    let impl_def_id = ast_util::local_def(impl_item.id);

    debug!("impl_vtables for {} are {}",
           impl_def_id.repr(tcx),
           vtbls.repr(tcx));

    tcx.impl_vtables.borrow_mut().insert(impl_def_id, vtbls);
}

/// Resolve vtables for a method call after typeck has finished.
/// Used by trans to monomorphize artificial method callees (e.g. drop).
pub fn trans_resolve_method(tcx: &ty::ctxt, id: ast::NodeId,
                            substs: &subst::Substs) -> vtable_res {
    let generics = ty::lookup_item_type(tcx, ast_util::local_def(id)).generics;
    let unboxed_closures = RefCell::new(DefIdMap::new());
    let vcx = VtableContext {
        infcx: &infer::new_infer_ctxt(tcx),
        param_env: &ty::construct_parameter_environment(tcx, &ty::Generics::empty(), id),
        unboxed_closures: &unboxed_closures,
    };

    lookup_vtables(&vcx,
                   tcx.map.span(id),
                   &generics.types,
                   substs,
                   false)
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

/// Used in the kind checker after typechecking has finished. Calls
/// `any_missing` if any bounds were missing.
pub fn check_param_bounds(tcx: &ty::ctxt,
                          span: Span,
                          parameter_environment: &ty::ParameterEnvironment,
                          type_param_defs:
                            &VecPerParamSpace<ty::TypeParameterDef>,
                          substs: &subst::Substs,
                          any_missing: |&ty::TraitRef|) {
    let unboxed_closures = RefCell::new(DefIdMap::new());
    let vcx = VtableContext {
        infcx: &infer::new_infer_ctxt(tcx),
        param_env: parameter_environment,
        unboxed_closures: &unboxed_closures,
    };
    let vtable_param_results =
        lookup_vtables(&vcx, span, type_param_defs, substs, false);
    for (vtable_param_result, type_param_def) in
            vtable_param_results.iter().zip(type_param_defs.iter()) {
        for (vtable_result, trait_ref) in
                vtable_param_result.iter()
                                   .zip(type_param_def.bounds
                                                      .trait_bounds
                                                      .iter()) {
            match *vtable_result {
                vtable_error => any_missing(&**trait_ref),
                vtable_static(..) |
                vtable_param(..) |
                vtable_unboxed_closure(..) => {}
            }
        }
    }
}

