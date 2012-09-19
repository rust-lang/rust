use check::{fn_ctxt, impl_self_ty};
use infer::{resolve_type, resolve_and_force_all_but_regions,
               fixup_err_to_str};
use syntax::print::pprust;
use result::{Result, Ok, Err};
use util::common::indenter;

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

fn has_trait_bounds(tps: ~[ty::param_bounds]) -> bool {
    vec::any(tps, |bs| {
        vec::any(*bs, |b| {
            match b { ty::bound_trait(_) => true, _ => false }
        })
    })
}

fn lookup_vtables(fcx: @fn_ctxt,
                  expr: @ast::expr,
                  bounds: @~[ty::param_bounds],
                  substs: &ty::substs,
                  allow_unsafe: bool,
                  is_early: bool) -> vtable_res
{
    debug!("lookup_vtables(expr=%?/%s, \
            # bounds=%?, \
            substs=%s",
           expr.id, fcx.expr_to_str(expr),
           bounds.len(),
           ty::substs_to_str(fcx.tcx(), substs));
    let _i = indenter();

    let tcx = fcx.ccx.tcx;
    let mut result = ~[], i = 0u;
    for substs.tps.each |ty| {
        for vec::each(*bounds[i]) |bound| {
            match *bound {
              ty::bound_trait(i_ty) => {
                let i_ty = ty::subst(tcx, substs, i_ty);
                vec::push(result, lookup_vtable(fcx, expr, *ty, i_ty,
                                                allow_unsafe, is_early));
              }
              _ => ()
            }
        }
        i += 1u;
    }
    @result
}

fn fixup_substs(fcx: @fn_ctxt, expr: @ast::expr,
                id: ast::def_id, substs: ty::substs,
                is_early: bool) -> Option<ty::substs> {
    let tcx = fcx.ccx.tcx;
    // use a dummy type just to package up the substs that need fixing up
    let t = ty::mk_trait(tcx, id, substs, ty::vstore_slice(ty::re_static));
    do fixup_ty(fcx, expr, t, is_early).map |t_f| {
        match ty::get(t_f).sty {
          ty::ty_trait(_, substs_f, _) => substs_f,
          _ => fail ~"t_f should be a trait"
        }
    }
}

fn relate_trait_tys(fcx: @fn_ctxt, expr: @ast::expr,
                    exp_trait_ty: ty::t, act_trait_ty: ty::t) {
    demand::suptype(fcx, expr.span, exp_trait_ty, act_trait_ty)
}

/*
Look up the vtable to use when treating an item of type <t>
as if it has type <trait_ty>
*/
fn lookup_vtable(fcx: @fn_ctxt,
                 expr: @ast::expr,
                 ty: ty::t,
                 trait_ty: ty::t,
                 allow_unsafe: bool,
                 is_early: bool)
    -> vtable_origin
{

    debug!("lookup_vtable(ty=%s, trait_ty=%s)",
           fcx.infcx().ty_to_str(ty), fcx.inh.infcx.ty_to_str(trait_ty));
    let _i = indenter();

    let tcx = fcx.ccx.tcx;
    let (trait_id, trait_substs) = match ty::get(trait_ty).sty {
        ty::ty_trait(did, substs, _) => (did, substs),
        _ => tcx.sess.impossible_case(expr.span,
                                      "lookup_vtable: \
                                       don't know how to handle a non-trait")
    };
    let ty = match fixup_ty(fcx, expr, ty, is_early) {
        Some(ty) => ty,
        None => {
            // fixup_ty can only fail if this is early resolution
            assert is_early;
            // The type has unconstrained type variables in it, so we can't
            // do early resolution on it. Return some completely bogus vtable
            // information: we aren't storing it anyways.
            return vtable_param(0, 0);
        }
    };

    match ty::get(ty).sty {
        ty::ty_param({idx: n, def_id: did}) => {
            let mut n_bound = 0;
            for vec::each(*tcx.ty_param_bounds.get(did.node)) |bound| {
                match *bound {
                    ty::bound_send | ty::bound_copy | ty::bound_const |
                    ty::bound_owned => {
                        /* ignore */
                    }
                    ty::bound_trait(ity) => {
                        match ty::get(ity).sty {
                            ty::ty_trait(idid, _, _) => {
                                if trait_id == idid {
                                    debug!("(checking vtable) @0 relating \
                                            ty to trait ty with did %?",
                                           idid);
                                    relate_trait_tys(fcx, expr,
                                                     trait_ty, ity);
                                    return vtable_param(n, n_bound);
                                }
                            }
                            _ => tcx.sess.impossible_case(
                                expr.span,
                                "lookup_vtable: in loop, \
                                 don't know how to handle a non-trait ity")
                        }
                        n_bound += 1u;
                    }
                }
            }
        }

        ty::ty_trait(did, substs, _) if trait_id == did => {
            debug!("(checking vtable) @1 relating ty to trait ty with did %?",
                   did);

            relate_trait_tys(fcx, expr, trait_ty, ty);
            if !allow_unsafe && !is_early {
                for vec::each(*ty::trait_methods(tcx, did)) |m| {
                    if ty::type_has_self(ty::mk_fn(tcx, m.fty)) {
                        tcx.sess.span_err(
                            expr.span,
                            ~"a boxed trait with self types may not be \
                              passed as a bounded type");
                    } else if (*m.tps).len() > 0u {
                        tcx.sess.span_err(
                            expr.span,
                            ~"a boxed trait with generic methods may not \
                              be passed as a bounded type");

                    }
                }
            }
            return vtable_trait(did, substs.tps);
        }

        _ => {
            let mut found = ~[];

            let mut impls_seen = HashMap();

            match fcx.ccx.coherence_info.extension_methods.find(trait_id) {
                None => {
                    // Nothing found. Continue.
                }
                Some(implementations) => {
                    // implementations is the list of all impls in scope for
                    // trait_ty. (Usually, there's just one.)
                    for uint::range(0, implementations.len()) |i| {
                        let im = implementations[i];

                        // im is one specific impl of trait_ty.

                        // First, ensure we haven't processed this impl yet.
                        if impls_seen.contains_key(im.did) {
                            loop;
                        }
                        impls_seen.insert(im.did, ());

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
                        for vec::each(ty::impl_traits(tcx, im.did)) |of_ty| {
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
                            let {substs: substs, ty: for_ty} =
                                impl_self_ty(fcx, expr, im.did);
                            let im_bs = ty::lookup_item_type(tcx,
                                                             im.did).bounds;
                            match fcx.mk_subty(false, expr.span, ty, for_ty) {
                                result::Err(_) => loop,
                                result::Ok(()) => ()
                            }

                            // Now, in the previous example, for_ty is
                            // bound to the type self_ty, and substs
                            // is bound to [T].
                            debug!("The self ty is %s and its substs are %s",
                                   fcx.infcx().ty_to_str(for_ty),
                                   tys_to_str(fcx.ccx.tcx, substs.tps));

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
                                   fcx.infcx().ty_to_str(trait_ty),
                                   fcx.infcx().ty_to_str(*of_ty));
                            let of_ty = ty::subst(tcx, &substs, *of_ty);
                            relate_trait_tys(fcx, expr, trait_ty, of_ty);

                            // Recall that trait_ty -- the trait type
                            // we're casting to -- is the trait with
                            // id trait_id applied to the substs
                            // trait_substs. Now we extract out the
                            // types themselves from trait_substs.

                            let trait_tps = trait_substs.tps;

                            debug!("Casting to a trait ty whose substs \
                                    (trait_tps) are %s",
                                   tys_to_str(fcx.ccx.tcx, trait_tps));

                            // Recall that substs is the impl self
                            // type's list of substitutions. That is,
                            // if this is an impl of some trait for
                            // foo<T, U>, then substs is [T,
                            // U]. substs might contain type
                            // variables, so we call fixup_substs to
                            // resolve them.

                            let substs_f = match fixup_substs(fcx,
                                                              expr,
                                                              trait_id,
                                                              substs,
                                                              is_early) {
                                Some(substs) => substs,
                                None => {
                                    assert is_early;
                                    // Bail out with a bogus answer
                                    return vtable_param(0, 0);
                                }
                            };

                            debug!("The fixed-up substs are %s - \
                                    they will be unified with the bounds for \
                                    the target ty, %s",
                                   tys_to_str(fcx.ccx.tcx, substs_f.tps),
                                   tys_to_str(fcx.ccx.tcx, trait_tps));

                            // Next, we unify the fixed-up
                            // substitutions for the impl self ty with
                            // the substitutions from the trait type
                            // that we're trying to cast
                            // to. connect_trait_tps requires these
                            // lists of types to unify pairwise.

                            connect_trait_tps(fcx, expr, substs_f.tps,
                                              trait_tps, im.did);
                            let subres = lookup_vtables(
                                fcx, expr, im_bs, &substs_f,
                                false, is_early);

                            // Finally, we register that we found a
                            // matching impl, and record the def ID of
                            // the impl as well as the resolved list
                            // of type substitutions for the target
                            // trait.
                            vec::push(found,
                                      vtable_static(im.did, substs_f.tps,
                                                    subres));
                        }
                    }
                }
            }

            match found.len() {
                0 => { /* fallthrough */ }
                1 => { return found[0]; }
                _ => {
                    if !is_early {
                        fcx.ccx.tcx.sess.span_err(
                            expr.span,
                            ~"multiple applicable methods in scope");
                    }
                    return found[0];
                }
            }
        }
    }

    tcx.sess.span_fatal(
        expr.span,
        fmt!("failed to find an implementation of trait %s for %s",
             ty_to_str(tcx, trait_ty), ty_to_str(tcx, ty)));
}

fn fixup_ty(fcx: @fn_ctxt,
            expr: @ast::expr,
            ty: ty::t,
            is_early: bool) -> Option<ty::t>
{
    let tcx = fcx.ccx.tcx;
    match resolve_type(fcx.infcx(), ty, resolve_and_force_all_but_regions) {
        Ok(new_type) => Some(new_type),
        Err(e) if !is_early => {
            tcx.sess.span_fatal(
                expr.span,
                fmt!("cannot determine a type \
                      for this bounded type parameter: %s",
                     fixup_err_to_str(e)))
        }
        Err(_) => {
            None
        }
    }
}

fn connect_trait_tps(fcx: @fn_ctxt, expr: @ast::expr, impl_tys: ~[ty::t],
                     trait_tys: ~[ty::t], impl_did: ast::def_id) {
    let tcx = fcx.ccx.tcx;

    // XXX: This should work for multiple traits.
    let ity = ty::impl_traits(tcx, impl_did)[0];
    let trait_ty = ty::subst_tps(tcx, impl_tys, ity);
    debug!("(connect trait tps) trait type is %?, impl did is %?",
           ty::get(trait_ty).sty, impl_did);
    match ty::get(trait_ty).sty {
     ty::ty_trait(_, substs, _) => {
        vec::iter2(substs.tps, trait_tys,
                   |a, b| demand::suptype(fcx, expr.span, a, b));
      }
     _ => tcx.sess.impossible_case(expr.span, "connect_trait_tps: \
            don't know how to handle a non-trait ty")
    }
}

fn insert_vtables(ccx: @crate_ctxt, callee_id: ast::node_id,
                  vtables: vtable_res) {
    debug!("insert_vtables(callee_id=%d, vtables=%?)",
           callee_id, vtables.map(|v| v.to_str(ccx.tcx)));
    ccx.vtable_map.insert(callee_id, vtables);
}

fn early_resolve_expr(ex: @ast::expr, &&fcx: @fn_ctxt, is_early: bool) {
    debug!("vtable: early_resolve_expr() ex with id %? (early: %b): %s",
           ex.id, is_early, expr_to_str(ex, fcx.tcx().sess.intr()));
    let _indent = indenter();

    let cx = fcx.ccx;
    match ex.node {
      ast::expr_path(*) => {
        match fcx.opt_node_ty_substs(ex.id) {
          Some(ref substs) => {
            let did = ast_util::def_id_of_def(cx.tcx.def_map.get(ex.id));
            let item_ty = ty::lookup_item_type(cx.tcx, did);
            if has_trait_bounds(*item_ty.bounds) {
                let vtbls = lookup_vtables(fcx, ex, item_ty.bounds,
                                           substs, false, is_early);
                if !is_early { cx.vtable_map.insert(ex.id, vtbls); }
            }
          }
          _ => ()
        }
      }
      // Must resolve bounds on methods with bounded params
      ast::expr_field(*) | ast::expr_binary(*) |
      ast::expr_unary(*) | ast::expr_assign_op(*) |
      ast::expr_index(*) => {
        match ty::method_call_bounds(cx.tcx, cx.method_map, ex.id) {
          Some(bounds) => {
            if has_trait_bounds(*bounds) {
                let callee_id = match ex.node {
                  ast::expr_field(_, _, _) => ex.id,
                  _ => ex.callee_id
                };
                let substs = fcx.node_ty_substs(callee_id);
                let vtbls = lookup_vtables(fcx, ex, bounds,
                                           &substs, false, is_early);
                if !is_early {
                    insert_vtables(cx, callee_id, vtbls);
                }
            }
          }
          None => ()
        }
      }
      ast::expr_cast(src, _) => {
        let target_ty = fcx.expr_ty(ex);
        match ty::get(target_ty).sty {
          ty::ty_trait(*) => {
            /*
            Look up vtables for the type we're casting to,
            passing in the source and target type
            */
            let vtable = lookup_vtable(fcx, ex, fcx.expr_ty(src),
                                       target_ty, true, is_early);
            /*
            Map this expression to that vtable (that is: "ex has
            vtable <vtable>")
            */
            if !is_early { cx.vtable_map.insert(ex.id, @~[vtable]); }
          }
          _ => ()
        }
      }
      _ => ()
    }
}

fn resolve_expr(ex: @ast::expr, &&fcx: @fn_ctxt, v: visit::vt<@fn_ctxt>) {
    early_resolve_expr(ex, fcx, false);
    visit::visit_expr(ex, fcx, v);
}

// Detect points where a trait-bounded type parameter is
// instantiated, resolve the impls for the parameters.
fn resolve_in_block(fcx: @fn_ctxt, bl: ast::blk) {
    visit::visit_block(bl, fcx, visit::mk_vt(@{
        visit_expr: resolve_expr,
        visit_item: fn@(_i: @ast::item, &&_e: @fn_ctxt,
                        _v: visit::vt<@fn_ctxt>) {},
        .. *visit::default_visitor()
    }));
}


