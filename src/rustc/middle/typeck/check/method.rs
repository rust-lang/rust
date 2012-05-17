/* Code to handle method lookups (which can be quite complex) */

import regionmanip::universally_quantify_regions;
import middle::typeck::infer::{ty_and_region_var_methods};

enum lookup = {
    fcx: @fn_ctxt,
    expr: @ast::expr, // expr for a.b in a.b()
    node_id: ast::node_id, // node id of call (not always expr.id)
    m_name: ast::ident, // b in a.b(...)
    self_ty: ty::t, // type of a in a.b(...)
    supplied_tps: [ty::t], // Xs in a.b::<Xs>(...)
    include_private: bool
};

impl methods for lookup {
    // Entrypoint:
    fn method() -> option<method_origin> {
        // First, see whether this is an interface-bounded parameter
        let pass1 = alt ty::get(self.self_ty).struct {
          ty::ty_param(n, did) {
            self.method_from_param(n, did)
          }
          ty::ty_iface(did, substs) {
            self.method_from_iface(did, substs)
          }
          ty::ty_class(did, substs) {
            self.method_from_class(did, substs)
          }
          _ {
            none
          }
        };

        alt pass1 {
          some(r) { some(r) }
          none { self.method_from_scope() }
        }
    }

    fn tcx() -> ty::ctxt { self.fcx.ccx.tcx }

    fn method_from_param(n: uint, did: ast::def_id) -> option<method_origin> {
        let tcx = self.tcx();
        let mut iface_bnd_idx = 0u; // count only iface bounds
        let bounds = tcx.ty_param_bounds.get(did.node);
        for vec::each(*bounds) {|bound|
            let (iid, bound_substs) = alt bound {
              ty::bound_copy | ty::bound_send { cont; /* ok */ }
              ty::bound_iface(bound_t) {
                alt check ty::get(bound_t).struct {
                  ty::ty_iface(i, substs) { (i, substs) }
                }
              }
            };

            let ifce_methods = ty::iface_methods(tcx, iid);
            alt vec::position(*ifce_methods, {|m| m.ident == self.m_name}) {
              none {
                /* check next bound */
                iface_bnd_idx += 1u;
              }

              some(pos) {
                // Replace any appearance of `self` with the type of the
                // generic parameter itself.  Note that this is the only case
                // where this replacement is necessary: in all other cases, we
                // are either invoking a method directly from an impl or class
                // (where the self type is not permitted), or from a iface
                // type (in which case methods that refer to self are not
                // permitted).
                let substs = {self_ty: some(self.self_ty)
                              with bound_substs};

                ret some(self.write_mty_from_m(
                    substs, ifce_methods[pos],
                    method_param(iid, pos, n, iface_bnd_idx)));
              }
            }
        }
        ret none;
    }

    fn method_from_iface(
        did: ast::def_id, iface_substs: ty::substs) -> option<method_origin> {

        let ms = *ty::iface_methods(self.tcx(), did);
        for ms.eachi {|i, m|
            if m.ident != self.m_name { cont; }

            let m_fty = ty::mk_fn(self.tcx(), m.fty);

            if ty::type_has_self(m_fty) {
                self.tcx().sess.span_err(
                    self.expr.span,
                    "can not call a method that contains a \
                     self type through a boxed iface");
            }

            if (*m.tps).len() > 0u {
                self.tcx().sess.span_err(
                    self.expr.span,
                    "can not call a generic method through a \
                     boxed iface");
            }

            // Note: although it is illegal to invoke a method that uses self
            // through a iface instance, we use a dummy subst here so that we
            // can soldier on with the compilation.
            let substs = {self_ty: some(self.self_ty)
                          with iface_substs};

            ret some(self.write_mty_from_m(
                substs, m, method_iface(did, i)));
        }

        ret none;
    }

    fn method_from_class(did: ast::def_id, class_substs: ty::substs)
        -> option<method_origin> {

        let ms = *ty::iface_methods(self.tcx(), did);

        for ms.each {|m|
            if m.ident != self.m_name { cont; }

            if m.vis == ast::private && !self.include_private {
                self.tcx().sess.span_fatal(
                    self.expr.span,
                    "Call to private method not allowed outside \
                     its defining class");
            }

            // look up method named <name>.
            let m_declared = ty::lookup_class_method_by_name(
                self.tcx(), did, self.m_name, self.expr.span);

            ret some(self.write_mty_from_m(
                class_substs, m,
                method_static(m_declared)));
        }

        ret none;
    }

    fn ty_from_did(did: ast::def_id) -> ty::t {
        alt check ty::get(ty::lookup_item_type(self.tcx(), did).ty).struct {
          ty::ty_fn(fty) {
            ty::mk_fn(self.tcx(), {proto: ast::proto_box with fty})
          }
        }
        /*
        if did.crate == ast::local_crate {
            alt check self.tcx().items.get(did.node) {
              ast_map::node_method(m, _, _) {
                // NDM iface/impl regions
                let mt = ty_of_method(self.fcx.ccx, m, ast::rp_none);
                ty::mk_fn(self.tcx(), {proto: ast::proto_box with mt.fty})
              }
            }
        } else {
            alt check ty::get(csearch::get_type(self.tcx(), did).ty).struct {
              ty::ty_fn(fty) {
                ty::mk_fn(self.tcx(), {proto: ast::proto_box with fty})
              }
            }
        }
        */
    }

    fn method_from_scope() -> option<method_origin> {
        let impls_vecs = self.fcx.ccx.impl_map.get(self.expr.id);

        for list::each(impls_vecs) {|impls|
            let mut results = [];
            for vec::each(*impls) {|im|
                // Check whether this impl has a method with the right name.
                for im.methods.find({|m| m.ident == self.m_name}).each {|m|

                    // determine the `self` with fresh variables for
                    // each parameter:
                    let {substs: self_substs, ty: self_ty} =
                        impl_self_ty(self.fcx, im.did);

                    // Here "self" refers to the callee side...
                    let self_ty =
                        universally_quantify_regions(
                            self.fcx, self.expr.span, self_ty);

                    // ... and "ty" refers to the caller side.
                    let ty =
                        universally_quantify_regions(
                            self.fcx, self.expr.span, self.self_ty);

                    // if we can assign the caller to the callee, that's a
                    // potential match.  Collect those in the vector.
                    alt self.fcx.mk_subty(ty, self_ty) {
                      result::err(_) { /* keep looking */ }
                      result::ok(_) {
                        results += [(self_substs, m.n_tps, m.did)];
                      }
                    }
                }
            }

            if results.len() >= 1u {
                if results.len() > 1u {
                    self.tcx().sess.span_err(
                        self.expr.span,
                        "multiple applicable methods in scope");

                    // I would like to print out how each impl was imported,
                    // but I cannot for the life of me figure out how to
                    // annotate resolve to preserve this information.
                    for results.eachi { |i, result|
                        let (_, _, did) = result;
                        let span = if did.crate == ast::local_crate {
                            alt check self.tcx().items.get(did.node) {
                              ast_map::node_method(m, _, _) { m.span }
                            }
                        } else {
                            self.expr.span
                        };
                        self.tcx().sess.span_note(
                            span,
                            #fmt["candidate #%u is %s",
                                 (i+1u),
                                 ty::item_path_str(self.tcx(), did)]);
                    }
                }

                let (self_substs, n_tps, did) = results[0];
                let fty = self.ty_from_did(did);
                ret some(self.write_mty_from_fty(
                    self_substs, n_tps, fty,
                    method_static(did)));
            }
        }

        ret none;
    }

    fn write_mty_from_m(self_substs: ty::substs,
                        m: ty::method,
                        origin: method_origin) -> method_origin {
        let tcx = self.fcx.ccx.tcx;

        // a bit hokey, but the method unbound has a bare protocol, whereas
        // a.b has a protocol like fn@() (perhaps eventually fn&()):
        let fty = ty::mk_fn(tcx, {proto: ast::proto_box with m.fty});

        ret self.write_mty_from_fty(self_substs, (*m.tps).len(),
                                    fty, origin);
    }

    fn write_mty_from_fty(self_substs: ty::substs,
                          n_tps_m: uint,
                          fty: ty::t,
                          origin: method_origin) -> method_origin {

        let tcx = self.fcx.ccx.tcx;

        // Here I will use the "c_" prefix to refer to the method's
        // owner.  You can read it as class, but it may also be an iface.

        let n_tps_supplied = self.supplied_tps.len();
        let m_substs = {
            if n_tps_supplied == 0u {
                self.fcx.infcx.next_ty_vars(n_tps_m)
            } else if n_tps_m == 0u {
                tcx.sess.span_err(
                    self.expr.span,
                    "this method does not take type parameters");
                self.fcx.infcx.next_ty_vars(n_tps_m)
            } else if n_tps_supplied != n_tps_m {
                tcx.sess.span_err(
                    self.expr.span,
                    "incorrect number of type \
                     parameters given for this method");
                self.fcx.infcx.next_ty_vars(n_tps_m)
            } else {
                self.supplied_tps
            }
        };

        let all_substs = {tps: self_substs.tps + m_substs
                          with self_substs};

        self.fcx.write_ty_substs(self.node_id, fty, all_substs);

        ret origin;
    }
}

