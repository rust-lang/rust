/* Code to handle method lookups (which can be quite complex) */

import coherence::get_base_type_def_id;
import middle::resolve3::{Impl, MethodInfo};
import middle::ty::{mk_box, mk_rptr, mk_uniq};
import middle::typeck::infer::methods; // next_ty_vars
import syntax::ast::{def_id, sty_box, sty_by_ref, sty_region, sty_uniq};
import syntax::ast::{sty_value};
import syntax::ast_map;
import syntax::ast_map::node_id_to_str;
import syntax::ast_util::{dummy_sp, new_def_hash};
import dvec::{dvec, extensions};

type candidate = {
    self_ty: ty::t,          // type of a in a.b()
    self_substs: ty::substs, // values for any tvars def'd on the class
    rcvr_ty: ty::t,          // type of receiver in the method def
    n_tps_m: uint,           // number of tvars defined on the method
    fty: ty::t,              // type of the method
    entry: method_map_entry
};

fn transform_self_type_for_method(fcx: @fn_ctxt,
                                  impl_ty: ty::t,
                                  method_info: MethodInfo)
                               -> ty::t {
    match method_info.self_type {
      sty_by_ref | sty_value => {
        impl_ty
      }
      sty_region(r, mutability) => {
        // XXX: dummy_sp is unfortunate here.
        let region = ast_region_to_region(fcx, fcx, dummy_sp(), r);
        mk_rptr(fcx.ccx.tcx, region, { ty: impl_ty, mutbl: mutability })
      }
      sty_box(mutability) => {
        mk_box(fcx.ccx.tcx, { ty: impl_ty, mutbl: mutability })
      }
      sty_uniq(mutability) => {
        mk_uniq(fcx.ccx.tcx, { ty: impl_ty, mutbl: mutability })
      }
    }
}

class lookup {
    let fcx: @fn_ctxt;
    let expr: @ast::expr;
    let self_expr: @ast::expr;
    let borrow_lb: ast::node_id;
    let node_id: ast::node_id;
    let m_name: ast::ident;
    let mut self_ty: ty::t;
    let mut derefs: uint;
    let candidates: dvec<candidate>;
    let candidate_impls: hashmap<def_id, ()>;
    let supplied_tps: ~[ty::t];
    let include_private: bool;

    new(fcx: @fn_ctxt,

        // In a call `a.b::<X, Y, ...>(...)`:
        expr: @ast::expr,        // The expression `a.b`.
        self_expr: @ast::expr,   // The expression `a`.
        borrow_lb: ast::node_id, // Scope to borrow the expression `a` for.
        node_id: ast::node_id,   // The node_id in which to store the type of
                                 // `a.b`.
        m_name: ast::ident,      // The ident `b`.
        self_ty: ty::t,          // The type of `a`.
        supplied_tps: ~[ty::t],  // The list of types X, Y, ... .
        include_private: bool) {

        self.fcx = fcx;
        self.expr = expr;
        self.self_expr = self_expr;
        self.borrow_lb = borrow_lb;
        self.node_id = node_id;
        self.m_name = m_name;
        self.self_ty = self_ty;
        self.derefs = 0u;
        self.candidates = dvec();
        self.candidate_impls = new_def_hash();
        self.supplied_tps = supplied_tps;
        self.include_private = include_private;
    }

    // Entrypoint:
    fn method() -> option<method_map_entry> {
        debug!{"method lookup(m_name=%s, self_ty=%s, %?)",
               *self.m_name, self.fcx.infcx.ty_to_str(self.self_ty),
               ty::get(self.self_ty).struct};

        // Determine if there are any inherent methods we can call.
        // (An inherent method is one that belongs to no trait, but is
        // inherent to a class or impl.)
        let optional_inherent_methods;
        match get_base_type_def_id(self.fcx.infcx,
                                 self.self_expr.span,
                                 self.self_ty) {
          none => {
            optional_inherent_methods = none;
          }
          some(base_type_def_id) => {
            debug!{"(checking method) found base type"};
            optional_inherent_methods =
                self.fcx.ccx.coherence_info.inherent_methods.find
                (base_type_def_id);

            if optional_inherent_methods.is_none() {
                debug!{"(checking method) ... no inherent methods found"};
            } else {
                debug!{"(checking method) ... inherent methods found"};
            }
          }
        }

        loop {
            // First, see whether this is a bounded parameter.
            match ty::get(self.self_ty).struct {
              ty::ty_param(p) => {
                self.add_candidates_from_param(p.idx, p.def_id);
              }
              ty::ty_trait(did, substs) => {
                self.add_candidates_from_trait(did, substs);
              }
              ty::ty_class(did, substs) => {
                self.add_candidates_from_class(did, substs);
              }
              _ => ()
            }

            // if we found anything, stop now.  otherwise continue to
            // loop for impls in scope.  Note: I don't love these
            // semantics, but that's what we had so I am preserving
            // it.
            if self.candidates.len() > 0u { break; }

            // now look for impls in scope, but don't look for impls that
            // would require doing an implicit borrow of the lhs.
            self.add_candidates_from_scope(false);

            // Look for inherent methods.
            self.add_inherent_and_extension_candidates
                (optional_inherent_methods, false);

            // if we found anything, stop before trying borrows
            if self.candidates.len() > 0u { break; }

            // now look for impls in scope that might require a borrow
            self.add_candidates_from_scope(true);

            // Again, look for inherent methods.
            self.add_inherent_and_extension_candidates
                (optional_inherent_methods, true);

            // if we found anything, stop before attempting auto-deref.
            if self.candidates.len() > 0u { break; }

            // check whether we can autoderef and if so loop around again.
            match ty::deref(self.tcx(), self.self_ty, false) {
              none => break,
              some(mt) => {
                self.self_ty = mt.ty;
                self.derefs += 1u;
              }
            }
        }

        if self.candidates.len() == 0u { return none; }

        if self.candidates.len() > 1u {
            self.tcx().sess.span_err(
                self.expr.span,
                ~"multiple applicable methods in scope");

            for self.candidates.eachi |i, candidate| {
                match candidate.entry.origin {
                  method_static(did) => {
                    self.report_static_candidate(i, did);
                  }
                  method_param(p) => {
                    self.report_param_candidate(i, p.trait_id);
                  }
                  method_trait(did, _) => {
                    self.report_trait_candidate(i, did);
                  }
                }
            }
        }

        some(self.write_mty_from_candidate(self.candidates[0u]))
    }

    fn tcx() -> ty::ctxt { self.fcx.ccx.tcx }

    fn report_static_candidate(idx: uint, did: ast::def_id) {
        let span = if did.crate == ast::local_crate {
            match check self.tcx().items.get(did.node) {
              ast_map::node_method(m, _, _) => m.span,
            }
        } else {
            self.expr.span
        };
        self.tcx().sess.span_note(
            span,
            fmt!{"candidate #%u is `%s`",
                 (idx+1u),
                 ty::item_path_str(self.tcx(), did)});
    }

    fn report_param_candidate(idx: uint, did: ast::def_id) {
        self.tcx().sess.span_note(
            self.expr.span,
            fmt!{"candidate #%u derives from the bound `%s`",
                 (idx+1u),
                 ty::item_path_str(self.tcx(), did)});
    }

    fn report_trait_candidate(idx: uint, did: ast::def_id) {
        self.tcx().sess.span_note(
            self.expr.span,
            fmt!{"candidate #%u derives from the type of the receiver, \
                  which is the trait `%s`",
                 (idx+1u),
                 ty::item_path_str(self.tcx(), did)});
    }

    fn add_candidates_from_param(n: uint, did: ast::def_id) {
        debug!{"candidates_from_param"};

        let tcx = self.tcx();
        let mut trait_bnd_idx = 0u; // count only trait bounds
        let bounds = tcx.ty_param_bounds.get(did.node);
        for vec::each(*bounds) |bound| {
            let (iid, bound_substs) = match bound {
              ty::bound_copy | ty::bound_send | ty::bound_const |
              ty::bound_owned => {
                again; /* ok */
              }
              ty::bound_trait(bound_t) => {
                match check ty::get(bound_t).struct {
                  ty::ty_trait(i, substs) => (i, substs)
                }
              }
            };

            let trt_methods = ty::trait_methods(tcx, iid);
            match vec::position(*trt_methods, |m| m.ident == self.m_name) {
              none => {
                /* check next bound */
                trait_bnd_idx += 1u;
              }

              some(pos) => {
                // Replace any appearance of `self` with the type of the
                // generic parameter itself.  Note that this is the only case
                // where this replacement is necessary: in all other cases, we
                // are either invoking a method directly from an impl or class
                // (where the self type is not permitted), or from a trait
                // type (in which case methods that refer to self are not
                // permitted).
                let substs = {self_ty: some(self.self_ty)
                              with bound_substs};

                self.add_candidates_from_m(
                    substs, trt_methods[pos],
                    method_param({trait_id:iid,
                                  method_num:pos,
                                  param_num:n,
                                  bound_num:trait_bnd_idx}));
              }
            }
        }

    }

    fn add_candidates_from_trait(did: ast::def_id, trait_substs: ty::substs) {

        debug!{"method_from_trait"};

        let ms = *ty::trait_methods(self.tcx(), did);
        for ms.eachi |i, m| {
            if m.ident != self.m_name { again; }

            let m_fty = ty::mk_fn(self.tcx(), m.fty);

            if ty::type_has_self(m_fty) {
                self.tcx().sess.span_err(
                    self.expr.span,
                    ~"cannot call a method whose type contains a \
                     self-type through a boxed trait");
            }

            if (*m.tps).len() > 0u {
                self.tcx().sess.span_err(
                    self.expr.span,
                    ~"cannot call a generic method through a \
                     boxed trait");
            }

            // Note: although it is illegal to invoke a method that uses self
            // through a trait instance, we use a dummy subst here so that we
            // can soldier on with the compilation.
            let substs = {self_ty: some(self.self_ty)
                          with trait_substs};

            self.add_candidates_from_m(
                substs, m, method_trait(did, i));
        }
    }

    fn add_candidates_from_class(did: ast::def_id, class_substs: ty::substs) {

        debug!{"method_from_class"};

        let ms = *ty::trait_methods(self.tcx(), did);

        for ms.each |m| {
            if m.ident != self.m_name { again; }

            if m.vis == ast::private && !self.include_private {
                self.tcx().sess.span_fatal(
                    self.expr.span,
                    ~"call to private method not allowed outside \
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
        match check ty::get(ty::lookup_item_type(self.tcx(), did).ty).struct {
          ty::ty_fn(fty) => {
            ty::mk_fn(self.tcx(), {proto: ast::proto_box with fty})
          }
        }
        /*
        if did.crate == ast::local_crate {
            match check self.tcx().items.get(did.node) {
              ast_map::node_method(m, _, _) {
                // NDM trait/impl regions
                let mt = ty_of_method(self.fcx.ccx, m, ast::rp_none);
                ty::mk_fn(self.tcx(), {proto: ast::proto_box with mt.fty})
              }
            }
        } else {
            match check ty::get(csearch::get_type(self.tcx(), did).ty)
              .struct {

              ty::ty_fn(fty) {
                ty::mk_fn(self.tcx(), {proto: ast::proto_box with fty})
              }
            }
        }
        */
    }

    fn add_candidates_from_scope(use_assignability: bool) {
        // If we're using coherence and this is one of the method invocation
        // forms it supports, don't use this method; it'll result in lots of
        // multiple-methods-in-scope errors.

        if self.fcx.ccx.trait_map.contains_key(self.expr.id) {
            return;
        }

        let impls_vecs = self.fcx.ccx.impl_map.get(self.expr.id);
        let mut added_any = false;

        debug!{"method_from_scope"};

        for list::each(impls_vecs) |impls| {
            for vec::each(*impls) |im| {
                if self.add_candidates_from_impl(im, use_assignability) {
                    added_any = true;
                }
            }

            // we want to find the innermost scope that has any
            // matches and then ignore outer scopes
            if added_any {return;}
        }
    }

    // Returns true if any were added and false otherwise.
    fn add_candidates_from_impl(im: @resolve3::Impl,
                                use_assignability: bool) -> bool {

        let mut added_any = false;

        // Check whether this impl has a method with the right name.
        for im.methods.find(|m| m.ident == self.m_name).each |m| {

            // determine the `self` of the impl with fresh
            // variables for each parameter:
            let {substs: impl_substs, ty: impl_ty} =
                impl_self_ty(self.fcx, im.did);

            let impl_ty = transform_self_type_for_method(self.fcx,
                                                         impl_ty,
                                                         *m);

            // Depending on our argument, we find potential
            // matches either by checking subtypability or
            // type assignability. Collect the matches.
            let matches = if use_assignability {
                self.fcx.can_mk_assignty(self.self_expr, self.borrow_lb,
                                         self.self_ty, impl_ty)
            } else {
                self.fcx.can_mk_subty(self.self_ty, impl_ty)
            };
            debug!{"matches = %?", matches};
            match matches {
              result::err(_) => { /* keep looking */ }
              result::ok(_) => {
                if !self.candidate_impls.contains_key(im.did) {
                    let fty = self.ty_from_did(m.did);
                    self.candidates.push(
                        {self_ty: self.self_ty,
                         self_substs: impl_substs,
                         rcvr_ty: impl_ty,
                         n_tps_m: m.n_tps,
                         fty: fty,
                         entry: {derefs: self.derefs,
                                 origin: method_static(m.did)}});
                    self.candidate_impls.insert(im.did, ());
                    added_any = true;
                }
              }
            }
        }

        return added_any;
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

    fn add_inherent_and_extension_candidates(optional_inherent_methods:
                                                option<@dvec<@Impl>>,
                                             use_assignability: bool) {

        // Add inherent methods.
        match optional_inherent_methods {
          none => {
            // Continue.
          }
          some(inherent_methods) => {
            debug!{"(adding inherent and extension candidates) adding \
                    inherent candidates"};
            for inherent_methods.each |implementation| {
                debug!{"(adding inherent and extension candidates) \
                        adding candidates from impl: %s",
                        node_id_to_str(self.tcx().items,
                                       implementation.did.node)};
                self.add_candidates_from_impl(implementation,
                                              use_assignability);
            }
          }
        }

        // Add trait methods.
        match self.fcx.ccx.trait_map.find(self.expr.id) {
          none => {
            // Should only happen for placement new right now.
          }
          some(trait_ids) => {
            for (*trait_ids).each |trait_id| {
                debug!{"(adding inherent and extension candidates) \
                        trying trait: %s",
                        self.def_id_to_str(trait_id)};

                let coherence_info = self.fcx.ccx.coherence_info;
                match coherence_info.extension_methods.find(trait_id) {
                  none => {
                    // Do nothing.
                  }
                  some(extension_methods) => {
                    for extension_methods.each |implementation| {
                        debug!{"(adding inherent and extension \
                                candidates) adding impl %s",
                                self.def_id_to_str
                                (implementation.did)};
                        self.add_candidates_from_impl
                            (implementation, use_assignability);
                    }
                  }
                }
            }
          }
        }
    }

    fn def_id_to_str(def_id: ast::def_id) -> ~str {
        if def_id.crate == ast::local_crate {
            node_id_to_str(self.tcx().items, def_id.node)
        } else {
            ast_map::path_to_str(csearch::get_item_path(self.tcx(), def_id))
        }
    }

    fn write_mty_from_candidate(cand: candidate) -> method_map_entry {
        let tcx = self.fcx.ccx.tcx;

        debug!{"write_mty_from_candidate(n_tps_m=%u, fty=%s, entry=%?)",
               cand.n_tps_m,
               self.fcx.infcx.ty_to_str(cand.fty),
               cand.entry};

        // Make the actual receiver type (cand.self_ty) assignable to the
        // required receiver type (cand.rcvr_ty).  If this method is not
        // from an impl, this'll basically be a no-nop.
        match self.fcx.mk_assignty(self.self_expr, self.borrow_lb,
                                   cand.self_ty, cand.rcvr_ty) {
          result::ok(_) => (),
          result::err(_) => {
            self.tcx().sess.span_bug(
                self.expr.span,
                fmt!{"%s was assignable to %s but now is not?",
                     self.fcx.infcx.ty_to_str(cand.self_ty),
                     self.fcx.infcx.ty_to_str(cand.rcvr_ty)});
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
                    ~"this method does not take type parameters");
                self.fcx.infcx.next_ty_vars(n_tps_m)
            } else if n_tps_supplied != n_tps_m {
                tcx.sess.span_err(
                    self.expr.span,
                    ~"incorrect number of type \
                     parameters given for this method");
                self.fcx.infcx.next_ty_vars(n_tps_m)
            } else {
                self.supplied_tps
            }
        };

        let all_substs = {tps: vec::append(cand.self_substs.tps, m_substs)
                          with cand.self_substs};

         self.fcx.write_ty_substs(self.node_id, cand.fty, all_substs);

        return cand.entry;
    }
}

