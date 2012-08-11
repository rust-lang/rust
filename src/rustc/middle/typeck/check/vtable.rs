import check::{fn_ctxt, impl_self_ty};
import infer::{resolve_type, resolve_all, force_all, fixup_err_to_str};
import ast_util::new_def_hash;

fn has_trait_bounds(tps: ~[ty::param_bounds]) -> bool {
    vec::any(tps, |bs| {
        vec::any(*bs, |b| {
            match b { ty::bound_trait(_) => true, _ => false }
        })
    })
}

fn lookup_vtables(fcx: @fn_ctxt,
                  sp: span,
                  bounds: @~[ty::param_bounds],
                  substs: &ty::substs,
                  allow_unsafe: bool) -> vtable_res {
    let tcx = fcx.ccx.tcx;
    let mut result = ~[], i = 0u;
    for substs.tps.each |ty| {
        for vec::each(*bounds[i]) |bound| {
            match bound {
              ty::bound_trait(i_ty) => {
                let i_ty = ty::subst(tcx, substs, i_ty);
                vec::push(result, lookup_vtable(fcx, sp, ty, i_ty,
                                                allow_unsafe));
              }
              _ => ()
            }
        }
        i += 1u;
    }
    @result
}

fn fixup_substs(fcx: @fn_ctxt, sp: span,
                id: ast::def_id, substs: ty::substs) -> ty::substs {
    let tcx = fcx.ccx.tcx;
    // use a dummy type just to package up the substs that need fixing up
    let t = ty::mk_trait(tcx, id, substs);
    let t_f = fixup_ty(fcx, sp, t);
    match check ty::get(t_f).struct {
      ty::ty_trait(_, substs_f) => substs_f,
    }
}

fn relate_trait_tys(fcx: @fn_ctxt, sp: span,
                    exp_trait_ty: ty::t, act_trait_ty: ty::t) {
    demand::suptype(fcx, sp, exp_trait_ty, act_trait_ty)
}

/*
Look up the vtable to use when treating an item of type <t>
as if it has type <trait_ty>
*/
fn lookup_vtable(fcx: @fn_ctxt, sp: span, ty: ty::t, trait_ty: ty::t,
                 allow_unsafe: bool)
              -> vtable_origin {

    debug!{"lookup_vtable(ty=%s, trait_ty=%s)",
           fcx.infcx.ty_to_str(ty), fcx.infcx.ty_to_str(trait_ty)};
    let _i = indenter();

    let tcx = fcx.ccx.tcx;
    let (trait_id, trait_substs) = match check ty::get(trait_ty).struct {
      ty::ty_trait(did, substs) => (did, substs)
    };
    let ty = fixup_ty(fcx, sp, ty);
    match ty::get(ty).struct {
      ty::ty_param({idx: n, def_id: did}) => {
        let mut n_bound = 0u;
        for vec::each(*tcx.ty_param_bounds.get(did.node)) |bound| {
            match bound {
              ty::bound_send | ty::bound_copy | ty::bound_const |
              ty::bound_owned => {
                /* ignore */
              }
              ty::bound_trait(ity) => {
                match check ty::get(ity).struct {
                  ty::ty_trait(idid, substs) => {
                    if trait_id == idid {
                        debug!{"(checking vtable) @0 relating ty to trait ty
                                with did %?", idid};
                        relate_trait_tys(fcx, sp, trait_ty, ity);
                        return vtable_param(n, n_bound);
                    }
                  }
                }
                n_bound += 1u;
              }
            }
        }
      }

      ty::ty_trait(did, substs) if trait_id == did => {
        debug!{"(checking vtable) @1 relating ty to trait ty with did %?",
               did};

        relate_trait_tys(fcx, sp, trait_ty, ty);
        if !allow_unsafe {
            for vec::each(*ty::trait_methods(tcx, did)) |m| {
                if ty::type_has_self(ty::mk_fn(tcx, m.fty)) {
                    tcx.sess.span_err(
                        sp, ~"a boxed trait with self types may not be \
                             passed as a bounded type");
                } else if (*m.tps).len() > 0u {
                    tcx.sess.span_err(
                        sp, ~"a boxed trait with generic methods may not \
                             be passed as a bounded type");

                }
            }
        }
        return vtable_trait(did, substs.tps);
      }

      _ => {
        let mut found = ~[];

        let mut impls_seen = new_def_hash();

        match fcx.ccx.coherence_info.extension_methods.find(trait_id) {
            none => {
                // Nothing found. Continue.
            }
            some(implementations) => {
                for uint::range(0, implementations.len()) |i| {
                    let im = implementations[i];

                    // im = one specific impl

                    // First, ensure that we haven't processed this impl yet.
                    if impls_seen.contains_key(im.did) {
                        again;
                    }
                    impls_seen.insert(im.did, ());

                    // find the trait that im implements (if any)
                    for vec::each(ty::impl_traits(tcx, im.did)) |of_ty| {
                        // it must have the same id as the expected one
                        match ty::get(of_ty).struct {
                          ty::ty_trait(id, _) if id != trait_id => again,
                          _ => { /* ok */ }
                        }

                        // check whether the type unifies with the type
                        // that the impl is for, and continue if not
                        let {substs: substs, ty: for_ty} =
                            impl_self_ty(fcx, im.did);
                        let im_bs = ty::lookup_item_type(tcx, im.did).bounds;
                        match fcx.mk_subty(ty, for_ty) {
                          result::err(_) => again,
                          result::ok(()) => ()
                        }

                        // check that desired trait type unifies
                        debug!{"(checking vtable) @2 relating trait ty %s to \
                                of_ty %s",
                               fcx.infcx.ty_to_str(trait_ty),
                               fcx.infcx.ty_to_str(of_ty)};
                        let of_ty = ty::subst(tcx, &substs, of_ty);
                        relate_trait_tys(fcx, sp, trait_ty, of_ty);

                        // recursively process the bounds
                        let trait_tps = trait_substs.tps;
                        let substs_f = fixup_substs(fcx, sp, trait_id,
                                                    substs);
                        connect_trait_tps(fcx, sp, substs_f.tps,
                                          trait_tps, im.did);
                        let subres = lookup_vtables(fcx, sp, im_bs, &substs_f,
                                                    false);
                        vec::push(found,
                                  vtable_static(im.did, substs_f.tps,
                                                subres));
                    }
                }
            }
        }

        match found.len() {
          0u => { /* fallthrough */ }
          1u => { return found[0]; }
          _ => {
            fcx.ccx.tcx.sess.span_err(
                sp, ~"multiple applicable methods in scope");
            return found[0];
          }
        }
      }
    }

    tcx.sess.span_fatal(
        sp, ~"failed to find an implementation of trait " +
        ty_to_str(tcx, trait_ty) + ~" for " +
        ty_to_str(tcx, ty));
}

fn fixup_ty(fcx: @fn_ctxt, sp: span, ty: ty::t) -> ty::t {
    let tcx = fcx.ccx.tcx;
    match resolve_type(fcx.infcx, ty, resolve_all | force_all) {
      result::ok(new_type) => new_type,
      result::err(e) => {
        tcx.sess.span_fatal(
            sp,
            fmt!{"cannot determine a type \
                  for this bounded type parameter: %s",
                 fixup_err_to_str(e)})
      }
    }
}

fn connect_trait_tps(fcx: @fn_ctxt, sp: span, impl_tys: ~[ty::t],
                     trait_tys: ~[ty::t], impl_did: ast::def_id) {
    let tcx = fcx.ccx.tcx;

    // XXX: This should work for multiple traits.
    let ity = ty::impl_traits(tcx, impl_did)[0];
    let trait_ty = ty::subst_tps(tcx, impl_tys, ity);
    debug!{"(connect trait tps) trait type is %?, impl did is %?",
           ty::get(trait_ty).struct, impl_did};
    match check ty::get(trait_ty).struct {
      ty::ty_trait(_, substs) => {
        vec::iter2(substs.tps, trait_tys,
                   |a, b| demand::suptype(fcx, sp, a, b));
      }
    }
}

fn resolve_expr(ex: @ast::expr, &&fcx: @fn_ctxt, v: visit::vt<@fn_ctxt>) {
    let cx = fcx.ccx;
    match ex.node {
      ast::expr_path(*) => {
        debug!("(vtable - resolving expr) resolving path expr");
        match fcx.opt_node_ty_substs(ex.id) {
          some(ref substs) => {
            let did = ast_util::def_id_of_def(cx.tcx.def_map.get(ex.id));
            let item_ty = ty::lookup_item_type(cx.tcx, did);
            if has_trait_bounds(*item_ty.bounds) {
                cx.vtable_map.insert(ex.id, lookup_vtables(fcx,
                                                           ex.span,
                                                           item_ty.bounds,
                                                           substs,
                                                           false));
            }
          }
          _ => ()
        }
      }
      // Must resolve bounds on methods with bounded params
      ast::expr_field(*) | ast::expr_binary(*) |
      ast::expr_unary(*) | ast::expr_assign_op(*) |
      ast::expr_index(*) => {
        debug!("(vtable - resolving expr) resolving field/binary/unary/\
                assign/index expr");
        match cx.method_map.find(ex.id) {
          some({origin: method_static(did), _}) => {
            let bounds = ty::lookup_item_type(cx.tcx, did).bounds;
            if has_trait_bounds(*bounds) {
                let callee_id = match ex.node {
                  ast::expr_field(_, _, _) => ex.id,
                  _ => ex.callee_id
                };
                let substs = fcx.node_ty_substs(callee_id);
                cx.vtable_map.insert(callee_id, lookup_vtables(fcx,
                                                               ex.span,
                                                               bounds,
                                                               &substs,
                                                               false));
            }
          }
          _ => ()
        }
      }
      ast::expr_cast(src, _) => {
        debug!("(vtable - resolving expr) resolving cast expr");
        let target_ty = fcx.expr_ty(ex);
        match ty::get(target_ty).struct {
          ty::ty_trait(*) => {
            /*
            Look up vtables for the type we're casting to,
            passing in the source and target type
            */
            let vtable = lookup_vtable(fcx, ex.span, fcx.expr_ty(src),
                                       target_ty, true);
            /*
            Map this expression to that vtable (that is: "ex has
            vtable <vtable>")
            */
            cx.vtable_map.insert(ex.id, @~[vtable]);
          }
          _ => ()
        }
      }
      _ => ()
    }
    visit::visit_expr(ex, fcx, v);
}

// Detect points where a trait-bounded type parameter is
// instantiated, resolve the impls for the parameters.
fn resolve_in_block(fcx: @fn_ctxt, bl: ast::blk) {
    visit::visit_block(bl, fcx, visit::mk_vt(@{
        visit_expr: resolve_expr,
        visit_item: fn@(_i: @ast::item, &&_e: @fn_ctxt,
                        _v: visit::vt<@fn_ctxt>) {}
        with *visit::default_visitor()
    }));
}


