import check::{fn_ctxt, impl_self_ty, methods};

fn has_iface_bounds(tps: [ty::param_bounds]) -> bool {
    vec::any(tps, {|bs|
        vec::any(*bs, {|b|
            alt b { ty::bound_iface(_) { true } _ { false } }
        })
    })
}

fn lookup_vtables(fcx: @fn_ctxt, isc: resolve::iscopes, sp: span,
                  bounds: @[ty::param_bounds], substs: ty::substs,
                  allow_unsafe: bool) -> vtable_res {
    let tcx = fcx.ccx.tcx;
    let mut result = [], i = 0u;
    for substs.tps.each {|ty|
        for vec::each(*bounds[i]) {|bound|
            alt bound {
              ty::bound_iface(i_ty) {
                let i_ty = ty::subst(tcx, substs, i_ty);
                result += [lookup_vtable(fcx, isc, sp, ty, i_ty,
                                         allow_unsafe)];
              }
              _ {}
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
    let t = ty::mk_iface(tcx, id, substs);
    let t_f = fixup_ty(fcx, sp, t);
    alt check ty::get(t_f).struct {
      ty::ty_iface(_, substs_f) { substs_f }
    }
}

fn relate_iface_tys(fcx: @fn_ctxt, sp: span,
                    exp_iface_ty: ty::t, act_iface_ty: ty::t) {
    demand::suptype(fcx, sp, exp_iface_ty, act_iface_ty)
}

/*
Look up the vtable to use when treating an item of type <t>
as if it has type <iface_ty>
*/
fn lookup_vtable(fcx: @fn_ctxt, isc: resolve::iscopes, sp: span,
                 ty: ty::t, iface_ty: ty::t, allow_unsafe: bool)
    -> vtable_origin {

    #debug["lookup_vtable(ty=%s, iface_ty=%s)",
           fcx.ty_to_str(ty), fcx.ty_to_str(iface_ty)];
    let _i = indenter();

    let tcx = fcx.ccx.tcx;
    let (iface_id, iface_substs) = alt check ty::get(iface_ty).struct {
      ty::ty_iface(did, substs) { (did, substs) }
    };
    let ty = fixup_ty(fcx, sp, ty);
    alt ty::get(ty).struct {
      ty::ty_param(n, did) {
        let mut n_bound = 0u;
        for vec::each(*tcx.ty_param_bounds.get(did.node)) { |bound|
            alt bound {
              ty::bound_send | ty::bound_copy { /* ignore */ }
              ty::bound_iface(ity) {
                alt check ty::get(ity).struct {
                  ty::ty_iface(idid, substs) {
                    if iface_id == idid {
                        relate_iface_tys(fcx, sp, iface_ty, ity);
                        ret vtable_param(n, n_bound);
                    }
                  }
                }
                n_bound += 1u;
              }
            }
        }
      }

      ty::ty_iface(did, substs) if iface_id == did {
        relate_iface_tys(fcx, sp, iface_ty, ty);
        if !allow_unsafe {
            for vec::each(*ty::iface_methods(tcx, did)) {|m|
                if ty::type_has_self(ty::mk_fn(tcx, m.fty)) {
                    tcx.sess.span_err(
                        sp, "a boxed iface with self types may not be \
                             passed as a bounded type");
                } else if (*m.tps).len() > 0u {
                    tcx.sess.span_err(
                        sp, "a boxed iface with generic methods may not \
                             be passed as a bounded type");

                }
            }
        }
        ret vtable_iface(did, substs.tps);
      }

      _ {
        let mut found = [];

        for list::each(isc) {|impls|
            /* For each impl in scope... */
                for vec::each(*impls) {|im|
                    // im = one specific impl
                    // find the iface that im implements (if any)
                    let of_ty = alt ty::impl_iface(tcx, im.did) {
                      some(of_ty) { of_ty }
                      _ { cont; }
                    };

                    // it must have the same id as the expected one
                    alt ty::get(of_ty).struct {
                      ty::ty_iface(id, _) if id != iface_id { cont; }
                      _ { /* ok */ }
                    }

                    // check whether the type unifies with the type
                    // that the impl is for, and continue if not
                    let {substs: substs, ty: for_ty} =
                        impl_self_ty(fcx, im.did);
                    let im_bs = ty::lookup_item_type(tcx, im.did).bounds;
                    alt fcx.mk_subty(ty, for_ty) {
                      result::err(_) { cont; }
                      result::ok(()) { }
                    }

                    // check that desired iface type unifies
                    let of_ty = ty::subst(tcx, substs, of_ty);
                    relate_iface_tys(fcx, sp, iface_ty, of_ty);

                    // recursively process the bounds
                    let iface_tps = iface_substs.tps;
                    let substs_f = fixup_substs(fcx, sp, iface_id, substs);
                    connect_iface_tps(fcx, sp, substs_f.tps,
                                      iface_tps, im.did);
                    let subres = lookup_vtables(fcx, isc, sp,
                                                im_bs, substs_f, false);
                    found += [vtable_static(im.did, substs_f.tps, subres)];
                }

                alt found.len() {
                  0u { /* fallthrough */ }
                  1u { ret found[0]; }
                  _ {
                    fcx.ccx.tcx.sess.span_err(
                        sp, "multiple applicable methods in scope");
                    ret found[0];
                  }
                }
            }
          }
        }

        tcx.sess.span_fatal(
            sp, "failed to find an implementation of interface " +
            ty_to_str(tcx, iface_ty) + " for " +
            ty_to_str(tcx, ty));
    }

fn fixup_ty(fcx: @fn_ctxt, sp: span, ty: ty::t) -> ty::t {
    let tcx = fcx.ccx.tcx;
    alt infer::resolve_deep(fcx.infcx, ty, true) {
      result::ok(new_type) { new_type }
      result::err(e) {
        tcx.sess.span_fatal(
            sp,
            #fmt["cannot determine a type \
                  for this bounded type parameter: %s",
                 infer::fixup_err_to_str(e)])
      }
    }
}

fn connect_iface_tps(fcx: @fn_ctxt, sp: span, impl_tys: [ty::t],
                     iface_tys: [ty::t], impl_did: ast::def_id) {
    let tcx = fcx.ccx.tcx;
    let ity = option::get(ty::impl_iface(tcx, impl_did));
    let iface_ty = ty::subst_tps(tcx, impl_tys, ity);
    alt check ty::get(iface_ty).struct {
      ty::ty_iface(_, substs) {
        vec::iter2(substs.tps, iface_tys,
                   {|a, b| demand::suptype(fcx, sp, a, b);});
      }
    }
}

fn resolve_expr(ex: @ast::expr, &&fcx: @fn_ctxt, v: visit::vt<@fn_ctxt>) {
    let cx = fcx.ccx;
    alt ex.node {
      ast::expr_path(*) {
        alt fcx.opt_node_ty_substs(ex.id) {
          some(substs) {
            let did = ast_util::def_id_of_def(cx.tcx.def_map.get(ex.id));
            let item_ty = ty::lookup_item_type(cx.tcx, did);
            if has_iface_bounds(*item_ty.bounds) {
                let impls = cx.impl_map.get(ex.id);
                cx.vtable_map.insert(ex.id, lookup_vtables(
                    fcx, impls, ex.span,
                    item_ty.bounds, substs, false));
            }
          }
          _ {}
        }
      }
      // Must resolve bounds on methods with bounded params
      ast::expr_field(*) | ast::expr_binary(*) |
      ast::expr_unary(*) | ast::expr_assign_op(*) |
      ast::expr_index(*) {
        alt cx.method_map.find(ex.id) {
          some(method_static(did)) {
            let bounds = ty::lookup_item_type(cx.tcx, did).bounds;
            if has_iface_bounds(*bounds) {
                let callee_id = alt ex.node {
                  ast::expr_field(_, _, _) { ex.id }
                  _ { ast_util::op_expr_callee_id(ex) }
                };
                let substs = fcx.node_ty_substs(callee_id);
                let iscs = cx.impl_map.get(ex.id);
                cx.vtable_map.insert(callee_id, lookup_vtables(
                    fcx, iscs, ex.span, bounds, substs, false));
            }
          }
          _ {}
        }
      }
      ast::expr_cast(src, _) {
        let target_ty = fcx.expr_ty(ex);
        alt ty::get(target_ty).struct {
          ty::ty_iface(*) {
            /* Casting to an interface type.
            Look up all impls for the cast expr...
            */
            let impls = cx.impl_map.get(ex.id);
            /*
            Look up vtables for the type we're casting to,
            passing in the source and target type
            */
            let vtable = lookup_vtable(fcx, impls, ex.span,
                                       fcx.expr_ty(src), target_ty,
                                       true);
            /*
            Map this expression to that vtable (that is: "ex has
            vtable <vtable>")
            */
            cx.vtable_map.insert(ex.id, @[vtable]);
          }
          _ {}
        }
      }
      _ {}
    }
    visit::visit_expr(ex, fcx, v);
}

// Detect points where an interface-bounded type parameter is
// instantiated, resolve the impls for the parameters.
fn resolve_in_block(fcx: @fn_ctxt, bl: ast::blk) {
    visit::visit_block(bl, fcx, visit::mk_vt(@{
        visit_expr: resolve_expr,
        visit_item: fn@(_i: @ast::item, &&_e: @fn_ctxt,
                        _v: visit::vt<@fn_ctxt>) {}
        with *visit::default_visitor()
    }));
}


