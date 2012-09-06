use combine::*;
use unify::*;
use to_str::to_str;

enum Sub = combine_fields;  // "subtype", "subregion" etc

impl Sub: combine {
    fn infcx() -> infer_ctxt { self.infcx }
    fn tag() -> ~str { ~"sub" }
    fn a_is_expected() -> bool { self.a_is_expected }

    fn sub() -> Sub { Sub(*self) }
    fn lub() -> Lub { Lub(*self) }
    fn glb() -> Glb { Glb(*self) }

    fn contratys(a: ty::t, b: ty::t) -> cres<ty::t> {
        let opp = combine_fields {
            a_is_expected: !self.a_is_expected,.. *self
        };
        Sub(opp).tys(b, a)
    }

    fn contraregions(a: ty::region, b: ty::region) -> cres<ty::region> {
        let opp = combine_fields {
            a_is_expected: !self.a_is_expected,.. *self
        };
        Sub(opp).regions(b, a)
    }

    fn regions(a: ty::region, b: ty::region) -> cres<ty::region> {
        debug!("%s.regions(%s, %s)",
               self.tag(),
               a.to_str(self.infcx),
               b.to_str(self.infcx));
        do indent {
            match self.infcx.region_vars.make_subregion(self.span, a, b) {
              Ok(()) => Ok(a),
              Err(e) => Err(e)
            }
        }
    }

    fn mts(a: ty::mt, b: ty::mt) -> cres<ty::mt> {
        debug!("mts(%s <: %s)", a.to_str(self.infcx), b.to_str(self.infcx));

        if a.mutbl != b.mutbl && b.mutbl != m_const {
            return Err(ty::terr_mutability);
        }

        match b.mutbl {
          m_mutbl => {
            // If supertype is mut, subtype must match exactly
            // (i.e., invariant if mut):
            eq_tys(&self, a.ty, b.ty).then(|| Ok(a) )
          }
          m_imm | m_const => {
            // Otherwise we can be covariant:
            self.tys(a.ty, b.ty).chain(|_t| Ok(a) )
          }
        }
    }

    fn protos(a: ty::fn_proto, b: ty::fn_proto) -> cres<ty::fn_proto> {
        match (a, b) {
            (ty::proto_bare, _) =>
                Ok(ty::proto_bare),

            (ty::proto_vstore(ty::vstore_box),
             ty::proto_vstore(ty::vstore_slice(_))) =>
                Ok(ty::proto_vstore(ty::vstore_box)),

            (ty::proto_vstore(ty::vstore_uniq),
             ty::proto_vstore(ty::vstore_slice(_))) =>
                Ok(ty::proto_vstore(ty::vstore_uniq)),

            (_, ty::proto_bare) =>
                Err(ty::terr_proto_mismatch(expected_found(&self, a, b))),

            (ty::proto_vstore(vs_a), ty::proto_vstore(vs_b)) => {
                do self.vstores(ty::terr_fn, vs_a, vs_b).chain |vs_c| {
                    Ok(ty::proto_vstore(vs_c))
                }
            }
        }
    }

    fn purities(a: purity, b: purity) -> cres<purity> {
        self.lub().purities(a, b).compare(b, || {
            ty::terr_purity_mismatch(expected_found(&self, a, b))
        })
    }

    fn ret_styles(a: ret_style, b: ret_style) -> cres<ret_style> {
        self.lub().ret_styles(a, b).compare(b, || {
            ty::terr_ret_style_mismatch(expected_found(&self, a, b))
        })
    }

    fn tys(a: ty::t, b: ty::t) -> cres<ty::t> {
        debug!("%s.tys(%s, %s)", self.tag(),
               a.to_str(self.infcx), b.to_str(self.infcx));
        if a == b { return Ok(a); }
        do indent {
            match (ty::get(a).struct, ty::get(b).struct) {
              (ty::ty_bot, _) => {
                Ok(a)
              }
              (ty::ty_infer(TyVar(a_id)), ty::ty_infer(TyVar(b_id))) => {
                var_sub_var(&self, a_id, b_id).then(|| Ok(a) )
              }
              (ty::ty_infer(TyVar(a_id)), _) => {
                var_sub_t(&self, a_id, b).then(|| Ok(a) )
              }
              (_, ty::ty_infer(TyVar(b_id))) => {
                t_sub_var(&self, a, b_id).then(|| Ok(a) )
              }
              (_, ty::ty_bot) => {
                Err(ty::terr_sorts(expected_found(&self, a, b)))
              }
              _ => {
                super_tys(&self, a, b)
              }
            }
        }
    }

    fn fns(a: &ty::fn_ty, b: &ty::fn_ty) -> cres<ty::fn_ty> {
        // Rather than checking the subtype relationship between `a` and `b`
        // as-is, we need to do some extra work here in order to make sure
        // that function subtyping works correctly with respect to regions
        // (issue #2263).

        // First, we instantiate each bound region in the subtype with a fresh
        // region variable.
        let {fn_ty: a_fn_ty, _} = {
            do replace_bound_regions_in_fn_ty(self.infcx.tcx, @Nil,
                                              None, a) |br| {
                // N.B.: The name of the bound region doesn't have
                // anything to do with the region variable that's created
                // for it.  The only thing we're doing with `br` here is
                // using it in the debug message.
                //
                // NDM--we should not be used dummy_sp() here, but
                // rather passing in the span or something like that.
                let rvar = self.infcx.next_region_var_nb(dummy_sp());
                debug!("Bound region %s maps to %s",
                       bound_region_to_str(self.infcx.tcx, br),
                       region_to_str(self.infcx.tcx, rvar));
                rvar
            }
        };

        // Second, we instantiate each bound region in the supertype with a
        // fresh concrete region.
        let {fn_ty: b_fn_ty, _} = {
            do replace_bound_regions_in_fn_ty(self.infcx.tcx, @Nil,
                                              None, b) |br| {
                // FIXME: eventually re_skolemized (issue #2263)
                ty::re_bound(br)
            }
        };

        // Try to compare the supertype and subtype now that they've been
        // instantiated.
        super_fns(&self, &a_fn_ty, &b_fn_ty)
    }

    // Traits please (FIXME: #2794):

    fn flds(a: ty::field, b: ty::field) -> cres<ty::field> {
        super_flds(&self, a, b)
    }

    fn vstores(vk: ty::terr_vstore_kind,
               a: ty::vstore, b: ty::vstore) -> cres<ty::vstore> {
        super_vstores(&self, vk, a, b)
    }

    fn modes(a: ast::mode, b: ast::mode) -> cres<ast::mode> {
        super_modes(&self, a, b)
    }

    fn args(a: ty::arg, b: ty::arg) -> cres<ty::arg> {
        super_args(&self, a, b)
    }

    fn substs(did: ast::def_id,
              as: &ty::substs,
              bs: &ty::substs) -> cres<ty::substs> {
        super_substs(&self, did, as, bs)
    }

    fn tps(as: &[ty::t], bs: &[ty::t]) -> cres<~[ty::t]> {
        super_tps(&self, as, bs)
    }

    fn self_tys(a: Option<ty::t>, b: Option<ty::t>) -> cres<Option<ty::t>> {
        super_self_tys(&self, a, b)
    }
}

