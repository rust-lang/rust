/* Code to handle method lookups (which can be quite complex) */

import coherence::get_base_type_def_id;
import middle::resolve3::{Impl, MethodInfo};
import middle::ty::{mk_box, mk_rptr, mk_uniq};
import syntax::ast::{def_id,
                     sty_static, sty_box, sty_by_ref, sty_region, sty_uniq};
import syntax::ast::{sty_value, by_ref, by_copy};
import syntax::ast_map;
import syntax::ast_map::node_id_to_str;
import syntax::ast_util::{dummy_sp, new_def_hash};
import dvec::{DVec, dvec};

enum method_lookup_mode {
    subtyping_mode,
    assignability_mode,
    immutable_reference_mode,
    mutable_reference_mode
}

type candidate = {
    self_ty: ty::t,             // type of a in a.b()
    self_substs: ty::substs,    // values for any tvars def'd on the class
    rcvr_ty: ty::t,             // type of receiver in the method def
    n_tps_m: uint,              // number of tvars defined on the method
    fty: ty::t,                 // type of the method
    entry: method_map_entry,
    mode: method_lookup_mode    // the mode we used
};

fn transform_self_type_for_method
    (tcx: ty::ctxt,
     self_region: Option<ty::region>,
     impl_ty: ty::t,
     self_type: ast::self_ty_)
                               -> ty::t {
    match self_type {
      sty_static => {
        tcx.sess.bug(~"calling transform_self_type_for_method on \
                       static method");
      }
      sty_by_ref | sty_value => {
        impl_ty
      }
      sty_region(mutability) => {
        mk_rptr(tcx,
                self_region.expect(~"self region missing for &self param"),
                { ty: impl_ty, mutbl: mutability })
      }
      sty_box(mutability) => {
        mk_box(tcx, { ty: impl_ty, mutbl: mutability })
      }
      sty_uniq(mutability) => {
        mk_uniq(tcx, { ty: impl_ty, mutbl: mutability })
      }
    }
}

fn get_mode_from_self_type(self_type: ast::self_ty_) -> ast::rmode {
    match self_type {
      sty_value => by_copy,
      _ => by_ref
    }
}

struct lookup {
    let fcx: @fn_ctxt;
    let expr: @ast::expr;
    let self_expr: @ast::expr;
    let borrow_lb: ast::node_id;
    let node_id: ast::node_id;
    let m_name: ast::ident;
    let mut self_ty: ty::t;
    let mut derefs: uint;
    let candidates: DVec<candidate>;
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
    fn method() -> Option<method_map_entry> {
        debug!("method lookup(m_name=%s, self_ty=%s, %?)",
               self.fcx.tcx().sess.str_of(self.m_name),
               self.fcx.infcx.ty_to_str(self.self_ty),
               ty::get(self.self_ty).struct);

        // Determine if there are any inherent methods we can call.
        // (An inherent method is one that belongs to no trait, but is
        // inherent to a class or impl.)
        let optional_inherent_methods;
        match get_base_type_def_id(self.fcx.infcx,
                                 self.self_expr.span,
                                 self.self_ty) {
          None => {
            optional_inherent_methods = None;
          }
          Some(base_type_def_id) => {
            debug!("(checking method) found base type");
            optional_inherent_methods =
                self.fcx.ccx.coherence_info.inherent_methods.find
                (base_type_def_id);

            if optional_inherent_methods.is_none() {
                debug!("(checking method) ... no inherent methods found");
            } else {
                debug!("(checking method) ... inherent methods found");
            }
          }
        }

        let matching_modes =
            [subtyping_mode, assignability_mode,
             immutable_reference_mode, mutable_reference_mode];

        loop {
            // Try to find a method that is keyed directly off of the
            // type. This only happens for boxed traits, type params,
            // classes, and self. If we see some sort of pointer, then
            // we look at candidates for the pointed to type to match
            // them against methods that take explicit self parameters.
            // N.B.: this looking through boxes to match against
            // explicit self parameters is *not* the same as
            // autoderef.
            // Try each of the possible matching semantics in turn.
            for matching_modes.each |mode| {
                match ty::get(self.self_ty).struct {
                  ty::ty_box(mt) | ty::ty_uniq(mt) | ty::ty_rptr(_, mt) => {
                    self.add_candidates_from_type(mt.ty, mode);
                  }
                  _ => { self.add_candidates_from_type(self.self_ty, mode); }
                }
                if self.candidates.len() > 0u { break; }
            }

            // if we found anything, stop now.  otherwise continue to
            // loop for impls in scope.  Note: I don't love these
            // semantics, but that's what we had so I am preserving
            // it.
            if self.candidates.len() > 0u { break; }

            // Try each of the possible matching semantics in turn.
            for matching_modes.each |mode| {
                self.add_inherent_and_extension_candidates(
                    optional_inherent_methods, mode);
                // If we find anything, stop.
                if self.candidates.len() > 0u { break; }
            }
            // if we found anything, stop before attempting auto-deref.
            if self.candidates.len() > 0u {
                debug!("(checking method) found at least one inherent \
                        method; giving up looking now");
                break;
            }

            // check whether we can autoderef and if so loop around again.
            match ty::deref(self.tcx(), self.self_ty, false) {
              None => break,
              Some(mt) => {
                self.self_ty = mt.ty;
                self.derefs += 1u;
              }
            }
        }

        if self.candidates.len() == 0u {
            debug!("(checking method) couldn't find any candidate methods; \
                    returning none");
            return None;
        }

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

        Some(self.write_mty_from_candidate(self.candidates[0u]))
    }

    fn tcx() -> ty::ctxt { self.fcx.ccx.tcx }

    fn report_static_candidate(idx: uint, did: ast::def_id) {
        let span = if did.crate == ast::local_crate {
            match self.tcx().items.get(did.node) {
              ast_map::node_method(m, _, _) => m.span,
              _ => fail ~"report_static_candidate: bad item"
            }
        } else {
            self.expr.span
        };
        self.tcx().sess.span_note(
            span,
            fmt!("candidate #%u is `%s`",
                 (idx+1u),
                 ty::item_path_str(self.tcx(), did)));
    }

    fn report_param_candidate(idx: uint, did: ast::def_id) {
        self.tcx().sess.span_note(
            self.expr.span,
            fmt!("candidate #%u derives from the bound `%s`",
                 (idx+1u),
                 ty::item_path_str(self.tcx(), did)));
    }

    fn report_trait_candidate(idx: uint, did: ast::def_id) {
        self.tcx().sess.span_note(
            self.expr.span,
            fmt!("candidate #%u derives from the type of the receiver, \
                  which is the trait `%s`",
                 (idx+1u),
                 ty::item_path_str(self.tcx(), did)));
    }

    fn add_candidates_from_type(inner_ty: ty::t, mode: method_lookup_mode) {
        match ty::get(inner_ty).struct {
          // First, see whether this is a bounded parameter.
          ty::ty_param(p) => {
            self.add_candidates_from_param(inner_ty, mode, p.idx, p.def_id);
          }
          ty::ty_trait(did, substs, _) => {
            self.add_candidates_from_trait(inner_ty, mode, did, substs);
          }
          ty::ty_class(did, substs) => {
            self.add_candidates_from_class(inner_ty, mode, did, substs);
          }
          ty::ty_self => {
            // Call is of the form "self.foo()" and appears in one
            // of a trait's provided methods.
            let self_def_id = self.fcx.self_impl_def_id.expect(
                ~"unexpected `none` for self_impl_def_id");

            let substs = {
                self_r: None,
                self_ty: None,
                tps: ~[],
            };

            self.add_candidates_from_trait(inner_ty, mode,
                                           self_def_id, substs);
          }
          _ => ()
        }
    }

    fn add_candidates_from_param(inner_ty: ty::t, mode: method_lookup_mode,
                                 n: uint, did: ast::def_id) {
        debug!("add_candidates_from_param");

        let tcx = self.tcx();
        let mut trait_bnd_idx = 0u; // count only trait bounds
        let bounds = tcx.ty_param_bounds.get(did.node);
        for vec::each(*bounds) |bound| {
            let (trait_id, bound_substs) = match bound {
              ty::bound_copy | ty::bound_send | ty::bound_const |
              ty::bound_owned => {
                again; /* ok */
              }
              ty::bound_trait(bound_t) => {
                match ty::get(bound_t).struct {
                  ty::ty_trait(i, substs, _) => (i, substs),
                  _ => fail ~"add_candidates_from_param: non-trait bound"
                }
              }
            };

            let trt_methods = ty::trait_methods(tcx, trait_id);
            match vec::position(*trt_methods, |m| m.ident == self.m_name) {
              None => {
                /* check next bound */
                trait_bnd_idx += 1u;
              }

              Some(pos) => {
                // Replace any appearance of `self` with the type of the
                // generic parameter itself.  Note that this is the only case
                // where this replacement is necessary: in all other cases, we
                // are either invoking a method directly from an impl or class
                // (where the self type is not permitted), or from a trait
                // type (in which case methods that refer to self are not
                // permitted).
                let substs = {self_ty: Some(self.self_ty)
                              with bound_substs};

                self.add_candidates_from_m(
                    inner_ty,
                    mode,
                    substs, trt_methods[pos],
                    method_param({trait_id:trait_id,
                                  method_num:pos,
                                  param_num:n,
                                  bound_num:trait_bnd_idx}));
              }
            }
        }

    }

    fn add_candidates_from_trait(inner_ty: ty::t,
                                 mode: method_lookup_mode,
                                 did: ast::def_id,
                                 trait_substs: ty::substs) {

        debug!("add_candidates_from_trait");

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
            let substs = {self_ty: Some(self.self_ty)
                          with trait_substs};

            self.add_candidates_from_m(
                inner_ty, mode, substs, m, method_trait(did, i));
        }
    }

    fn add_candidates_from_class(inner_ty: ty::t,
                                 mode: method_lookup_mode,
                                 did: ast::def_id,
                                 class_substs: ty::substs) {

        debug!("add_candidates_from_class");

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
                inner_ty, mode, class_substs, m, method_static(m_declared));
        }
    }

    fn ty_from_did(did: ast::def_id) -> ty::t {
        match ty::get(ty::lookup_item_type(self.tcx(), did).ty).struct {
          ty::ty_fn(fty) => {
            ty::mk_fn(self.tcx(),
                      {proto: ty::proto_vstore(ty::vstore_box) with fty})
          }
          _ => fail ~"ty_from_did: not function ty"
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

    fn check_type_match(impl_ty: ty::t,
                        mode: method_lookup_mode)
        -> result<(), ty::type_err> {
        // Depending on our argument, we find potential matches by
        // checking subtypability, type assignability, or reference
        // subtypability. Collect the matches.
        let matches;
        match mode {
          subtyping_mode => {
            matches = self.fcx.can_mk_subty(self.self_ty, impl_ty);
          }
          assignability_mode => {
            matches = self.fcx.can_mk_assignty(self.self_expr,
                                               self.borrow_lb,
                                               self.self_ty,
                                               impl_ty);
          }
          immutable_reference_mode => {
            let region = self.fcx.infcx.next_region_var(
                self.self_expr.span,
                self.self_expr.id);
            let tm = { ty: self.self_ty, mutbl: ast::m_imm };
            let ref_ty = ty::mk_rptr(self.tcx(), region, tm);
            matches = self.fcx.can_mk_subty(ref_ty, impl_ty);
          }
          mutable_reference_mode => {
            let region = self.fcx.infcx.next_region_var(
                self.self_expr.span,
                self.self_expr.id);
            let tm = { ty: self.self_ty, mutbl: ast::m_mutbl };
            let ref_ty = ty::mk_rptr(self.tcx(), region, tm);
            matches = self.fcx.can_mk_subty(ref_ty, impl_ty);
          }
        }
        matches
    }

    // Returns true if any were added and false otherwise.
    fn add_candidates_from_impl(im: @resolve3::Impl, mode: method_lookup_mode)
                             -> bool {
        let mut added_any = false;

        // Check whether this impl has a method with the right name.
        for im.methods.find(|m| m.ident == self.m_name).each |m| {

            let need_rp = match m.self_type { ast::sty_region(_) => true,
                                              _ => false };

            // determine the `self` of the impl with fresh
            // variables for each parameter:
            let {substs: impl_substs, ty: impl_ty} =
                impl_self_ty(self.fcx, self.self_expr, im.did, need_rp);

            let impl_ty = transform_self_type_for_method(
                self.tcx(), impl_substs.self_r,
                impl_ty, m.self_type);

            let matches = self.check_type_match(impl_ty, mode);
            debug!("matches = %?", matches);
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
                                 self_mode: get_mode_from_self_type(
                                     m.self_type),
                                 origin: method_static(m.did)},
                         mode: mode});
                    self.candidate_impls.insert(im.did, ());
                    added_any = true;
                }
              }
            }
        }

        return added_any;
    }

    fn add_candidates_from_m(inner_ty: ty::t,
                             mode: method_lookup_mode,
                             self_substs: ty::substs,
                             m: ty::method,
                             origin: method_origin) {
        let tcx = self.fcx.ccx.tcx;

        // If we don't have a self region but have an region pointer
        // explicit self, we need to make up a new region.
        let self_r = match self_substs.self_r {
          None => {
            match m.self_ty {
              ast::sty_region(_) =>
                  Some(self.fcx.infcx.next_region_var(
                      self.self_expr.span,
                      self.self_expr.id)),
              _ => None
            }
          }
          Some(_) => self_substs.self_r
        };
        let self_substs = {self_r: self_r with self_substs};

        // Before we can be sure we succeeded we need to match the
        // self type against the impl type that we get when we apply
        // the explicit self parameter to whatever inner type we are
        // looking at (which may be something that the self_type
        // points to).
        let impl_ty = transform_self_type_for_method(
            self.tcx(), self_substs.self_r,
            inner_ty, m.self_ty);

        let matches = self.check_type_match(impl_ty, mode);
        debug!("matches = %?", matches);
        if matches.is_err() { return; }

        // a bit hokey, but the method unbound has a bare protocol, whereas
        // a.b has a protocol like fn@() (perhaps eventually fn&()):
        let fty = ty::mk_fn(tcx, {proto: ty::proto_vstore(ty::vstore_box)
                                  with m.fty});

        self.candidates.push(
            {self_ty: self.self_ty,
             self_substs: self_substs,
             rcvr_ty: self.self_ty,
             n_tps_m: (*m.tps).len(),
             fty: fty,
             entry: {derefs: self.derefs,
                     self_mode: get_mode_from_self_type(m.self_ty),
                     origin: origin},
             mode: mode});
    }

    fn add_inherent_and_extension_candidates(optional_inherent_methods:
                                                Option<@DVec<@Impl>>,
                                             mode: method_lookup_mode) {

        // Add inherent methods.
        match optional_inherent_methods {
          None => {
            // Continue.
          }
          Some(inherent_methods) => {
            debug!("(adding inherent and extension candidates) adding \
                    inherent candidates");
            for inherent_methods.each |implementation| {
                debug!("(adding inherent and extension candidates) \
                        adding candidates from impl: %s",
                        node_id_to_str(self.tcx().items,
                                       implementation.did.node,
                                       self.fcx.tcx().sess.parse_sess
                                           .interner));
                self.add_candidates_from_impl(implementation, mode);
            }
          }
        }

        // Add trait methods.
        match self.fcx.ccx.trait_map.find(self.expr.id) {
          None => {
            // Should only happen for placement new right now.
          }
          Some(trait_ids) => {
            for (*trait_ids).each |trait_id| {
                debug!("(adding inherent and extension candidates) \
                        trying trait: %s",
                        self.def_id_to_str(trait_id));

                let coherence_info = self.fcx.ccx.coherence_info;
                match coherence_info.extension_methods.find(trait_id) {
                  None => {
                    // Do nothing.
                  }
                  Some(extension_methods) => {
                    for extension_methods.each |implementation| {
                        debug!("(adding inherent and extension \
                                candidates) adding impl %s",
                                self.def_id_to_str
                                (implementation.did));
                        self.add_candidates_from_impl(implementation, mode);
                    }
                  }
                }
            }
          }
        }
    }

    fn def_id_to_str(def_id: ast::def_id) -> ~str {
        if def_id.crate == ast::local_crate {
            node_id_to_str(self.tcx().items, def_id.node,
                           self.fcx.tcx().sess.parse_sess.interner)
        } else {
            ast_map::path_to_str(csearch::get_item_path(self.tcx(), def_id),
                                 self.fcx.tcx().sess.parse_sess.interner)
        }
    }

    fn write_mty_from_candidate(cand: candidate) -> method_map_entry {
        let tcx = self.fcx.ccx.tcx;

        debug!("write_mty_from_candidate(n_tps_m=%u, fty=%s, entry=%?)",
               cand.n_tps_m,
               self.fcx.infcx.ty_to_str(cand.fty),
               cand.entry);

        match cand.mode {
            subtyping_mode | assignability_mode => {
                // Make the actual receiver type (cand.self_ty) assignable to
                // the required receiver type (cand.rcvr_ty).  If this method
                // is not from an impl, this'll basically be a no-nop.
                match self.fcx.mk_assignty(self.self_expr, self.borrow_lb,
                                           cand.self_ty, cand.rcvr_ty) {
                  result::ok(_) => (),
                  result::err(_) => {
                    self.tcx().sess.span_bug(
                        self.expr.span,
                        fmt!("%s was assignable to %s but now is not?",
                             self.fcx.infcx.ty_to_str(cand.self_ty),
                             self.fcx.infcx.ty_to_str(cand.rcvr_ty)));
                  }
                }
            }
            immutable_reference_mode => {
                // Borrow as an immutable reference.
                let region_var = self.fcx.infcx.next_region_var(
                    self.self_expr.span,
                    self.self_expr.id);
                self.fcx.infcx.borrowings.push({expr_id: self.self_expr.id,
                                                span: self.self_expr.span,
                                                scope: region_var,
                                                mutbl: ast::m_imm});
            }
            mutable_reference_mode => {
                // Borrow as a mutable reference.
                let region_var = self.fcx.infcx.next_region_var(
                    self.self_expr.span,
                    self.self_expr.id);
                self.fcx.infcx.borrowings.push({expr_id: self.self_expr.id,
                                                span: self.self_expr.span,
                                                scope: region_var,
                                                mutbl: ast::m_mutbl});
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

