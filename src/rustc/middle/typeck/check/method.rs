/* Code to handle method lookups (which can be quite complex) */

import syntax::ast_map;
import middle::typeck::infer::methods; // next_ty_vars
import dvec::{dvec, extensions};

type candidate = {
    self_ty: ty::t,          // type of a in a.b()
    self_substs: ty::substs, // values for any tvars def'd on the class
    rcvr_ty: ty::t,          // type of receiver in the method def
    n_tps_m: uint,           // number of tvars defined on the method
    fty: ty::t,              // type of the method
    entry: method_map_entry
};

class lookup {
    let fcx: @fn_ctxt;
    let expr: @ast::expr;
    let self_expr: @ast::expr;
    let borrow_scope: ast::node_id;
    let node_id: ast::node_id;
    let m_name: ast::ident;
    let mut self_ty: ty::t;
    let mut derefs: uint;
    let candidates: dvec<candidate>;
    let supplied_tps: [ty::t];
    let include_private: bool;

    new(fcx: @fn_ctxt,
        expr: @ast::expr,           //expr for a.b in a.b()
        self_expr: @ast::expr,      //a in a.b(...)
        borrow_scope: ast::node_id, //scope to borrow the expr for
        node_id: ast::node_id,      //node id where to store type of fn
        m_name: ast::ident,         //b in a.b(...)
        self_ty: ty::t,             //type of a in a.b(...)
        supplied_tps: [ty::t],      //Xs in a.b::<Xs>(...)
        include_private: bool) {

        self.fcx = fcx;
        self.expr = expr;
        self.self_expr = self_expr;
        self.borrow_scope = borrow_scope;
        self.node_id = node_id;
        self.m_name = m_name;
        self.self_ty = self_ty;
        self.derefs = 0u;
        self.candidates = dvec();
        self.supplied_tps = supplied_tps;
        self.include_private = include_private;
    }

    // Entrypoint:
    fn method() -> option<method_map_entry> {
        #debug["method lookup(m_name=%s, self_ty=%s)",
               *self.m_name, self.fcx.infcx.ty_to_str(self.self_ty)];

        loop {
            // First, see whether this is an interface-bounded parameter
            alt ty::get(self.self_ty).struct {
              ty::ty_param(n, did) {
                self.add_candidates_from_param(n, did);
              }
              ty::ty_iface(did, substs) {
                self.add_candidates_from_iface(did, substs);
              }
              ty::ty_class(did, substs) {
                self.add_candidates_from_class(did, substs);
              }
              _ { }
            }

            // if we found anything, stop now.  otherwise continue to
            // loop for impls in scope.  Note: I don't love these
            // semantics, but that's what we had so I am preserving
            // it.
            if self.candidates.len() > 0u {
                break;
            }

            self.add_candidates_from_scope();

            // if we found anything, stop before attempting auto-deref.
            if self.candidates.len() > 0u {
                break;
            }

            // check whether we can autoderef and if so loop around again.
            alt ty::deref(self.tcx(), self.self_ty, false) {
              none { break; }
              some(mt) {
                self.self_ty = mt.ty;
                self.derefs += 1u;
              }
            }
        }

        if self.candidates.len() == 0u { ret none; }

        if self.candidates.len() > 1u {
            self.tcx().sess.span_err(
                self.expr.span,
                "multiple applicable methods in scope");

            for self.candidates.eachi { |i, candidate|
                alt candidate.entry.origin {
                  method_static(did) {
                    self.report_static_candidate(i, did);
                  }
                  method_param(p) {
                    self.report_param_candidate(i, p.iface_id);
                  }
                  method_iface(did, _) {
                    self.report_iface_candidate(i, did);
                  }
                }
            }
        }

        some(self.write_mty_from_candidate(self.candidates[0u]))
    }

    fn tcx() -> ty::ctxt { self.fcx.ccx.tcx }

    fn report_static_candidate(idx: uint, did: ast::def_id) {
        let span = if did.crate == ast::local_crate {
            alt check self.tcx().items.get(did.node) {
              ast_map::node_method(m, _, _) { m.span }
            }
        } else {
            self.expr.span
        };
        self.tcx().sess.span_note(
            span,
            #fmt["candidate #%u is `%s`",
                 (idx+1u),
                 ty::item_path_str(self.tcx(), did)]);
    }

    fn report_param_candidate(idx: uint, did: ast::def_id) {
        self.tcx().sess.span_note(
            self.expr.span,
            #fmt["candidate #%u derives from the bound `%s`",
                 (idx+1u),
                 ty::item_path_str(self.tcx(), did)]);
    }

    fn report_iface_candidate(idx: uint, did: ast::def_id) {
        self.tcx().sess.span_note(
            self.expr.span,
            #fmt["candidate #%u derives from the type of the receiver, \
                  which is the iface `%s`",
                 (idx+1u),
                 ty::item_path_str(self.tcx(), did)]);
    }

    fn add_candidates_from_param(n: uint, did: ast::def_id) {

        let tcx = self.tcx();
        let mut iface_bnd_idx = 0u; // count only iface bounds
        let bounds = tcx.ty_param_bounds.get(did.node);
        for vec::each(*bounds) {|bound|
            let (iid, bound_substs) = alt bound {
              ty::bound_copy | ty::bound_send | ty::bound_const {
                cont; /* ok */
              }
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

                self.add_candidates_from_m(
                    substs, ifce_methods[pos],
                    method_param({iface_id:iid,
                                  method_num:pos,
                                  param_num:n,
                                  bound_num:iface_bnd_idx}));
              }
            }
        }

    }

    fn add_candidates_from_iface(did: ast::def_id, iface_substs: ty::substs) {

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

            self.add_candidates_from_m(
                substs, m, method_iface(did, i));
        }
    }

    fn add_candidates_from_class(did: ast::def_id, class_substs: ty::substs) {

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

            self.add_candidates_from_m(
                class_substs, m, method_static(m_declared));
        }
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

    fn add_candidates_from_scope() {
        let impls_vecs = self.fcx.ccx.impl_map.get(self.expr.id);
        let mut added_any = false;

        for list::each(impls_vecs) {|impls|
            for vec::each(*impls) {|im|
                // Check whether this impl has a method with the right name.
                for im.methods.find({|m| m.ident == self.m_name}).each {|m|

                    // determine the `self` of the impl with fresh
                    // variables for each parameter:
                    let {substs: impl_substs, ty: impl_ty} =
                        impl_self_ty(self.fcx, im.did);

                    // if we can assign the caller to the callee, that's a
                    // potential match.  Collect those in the vector.
                    alt self.fcx.can_mk_assignty(
                        self.self_expr, self.borrow_scope,
                        self.self_ty, impl_ty) {
                      result::err(_) { /* keep looking */ }
                      result::ok(_) {
                        let fty = self.ty_from_did(m.did);
                        self.candidates.push(
                            {self_ty: self.self_ty,
                             self_substs: impl_substs,
                             rcvr_ty: impl_ty,
                             n_tps_m: m.n_tps,
                             fty: fty,
                             entry: {derefs: self.derefs,
                                     origin: method_static(m.did)}});
                        added_any = true;
                      }
                    }
                }
            }

            // we want to find the innermost scope that has any
            // matches and then ignore outer scopes
            if added_any {ret;}
        }
    }

    fn add_candidates_from_m(self_substs: ty::substs,
                             m: ty::method,
                             origin: method_origin) {
        let tcx = self.fcx.ccx.tcx;

        // a bit hokey, but the method unbound has a bare protocol, whereas
        // a.b has a protocol like fn@() (perhaps eventually fn&()):
        let fty = ty::mk_fn(tcx, {proto: ast::proto_box with m.fty});

        self.candidates.push(
            {self_ty: self.self_ty,
             self_substs: self_substs,
             rcvr_ty: self.self_ty,
             n_tps_m: (*m.tps).len(),
             fty: fty,
             entry: {derefs: self.derefs, origin: origin}});
    }

    fn write_mty_from_candidate(cand: candidate) -> method_map_entry {
        let tcx = self.fcx.ccx.tcx;

        #debug["write_mty_from_candidate(n_tps_m=%u, fty=%s, entry=%?)",
               cand.n_tps_m,
               self.fcx.infcx.ty_to_str(cand.fty),
               cand.entry];

        // Make the actual receiver type (cand.self_ty) assignable to the
        // required receiver type (cand.rcvr_ty).  If this method is not
        // from an impl, this'll basically be a no-nop.
        alt self.fcx.mk_assignty(self.self_expr, self.borrow_scope,
                                 cand.self_ty, cand.rcvr_ty) {
          result::ok(_) {}
          result::err(_) {
            self.tcx().sess.span_bug(
                self.expr.span,
                #fmt["%s was assignable to %s but now is not?",
                     self.fcx.infcx.ty_to_str(cand.self_ty),
                     self.fcx.infcx.ty_to_str(cand.rcvr_ty)]);
          }
        }

        // Construct the full set of type parameters for the method,
        // which is equal to the class tps + the method tps.
        let n_tps_supplied = self.supplied_tps.len();
        let n_tps_m = cand.n_tps_m;
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

        let all_substs = {tps: cand.self_substs.tps + m_substs
                          with cand.self_substs};

        self.fcx.write_ty_substs(self.node_id, cand.fty, all_substs);

        ret cand.entry;
    }
}

