iface lattice<T> {
    fn lub(T, T) -> cres<T>;
    fn glb(T, T) -> cres<T>;
}

iface lattice_op<T> {
    fn bnd<V:copy>(b: bounds<V>) -> option<V>;
    fn with_bnd<V:copy>(b: bounds<V>, v: V) -> bounds<V>;
}

iface pairwise {
    fn infcx() -> infer_ctxt;
    fn tag() -> str;

    fn c_tys(t1: ty::t, t2: ty::t) -> cres<ty::t>;
    fn c_flds(a: ty::field, b: ty::field) -> cres<ty::field>;
    fn c_bot(b: ty::t) -> cres<ty::t>;
    fn c_mts(a: ty::mt, b: ty::mt) -> cres<ty::mt>;
    fn c_contratys(t1: ty::t, t2: ty::t) -> cres<ty::t>;
    fn c_protos(p1: ast::proto, p2: ast::proto) -> cres<ast::proto>;
    fn c_ret_styles(r1: ret_style, r2: ret_style) -> cres<ret_style>;

    // Combining regions (along with some specific cases that are
    // different for LUB/GLB):
    fn c_regions(
        a: ty::region, b: ty::region) -> cres<ty::region>;
    fn c_regions_scope_scope(
        a: ty::region, a_id: ast::node_id,
        b: ty::region, b_id: ast::node_id) -> cres<ty::region>;
    fn c_regions_free_scope(
        a: ty::region, a_id: ast::node_id, a_br: ty::bound_region,
        b: ty::region, b_id: ast::node_id) -> cres<ty::region>;
}

fn c_vars<V:copy vid, PW:pairwise, T:copy to_str st>(
    self: PW, vb: vals_and_bindings<V, T>,
    a_t: T, a_vid: V, b_vid: V,
    c_ts: fn(T, T) -> cres<T>) -> cres<T> {

    // The comments in this function are written for LUB and types,
    // but they apply equally well to GLB and regions if you inverse
    // upper/lower/sub/super/etc.

    // Need to find a type that is a supertype of both a and b:
    let {root: a_vid, bounds: a_bounds} = self.infcx().get(vb, a_vid);
    let {root: b_vid, bounds: b_bounds} = self.infcx().get(vb, b_vid);

    #debug["%s.c_vars(%s=%s <: %s=%s)",
           self.tag(),
           a_vid.to_str(), a_bounds.to_str(self.infcx()),
           b_vid.to_str(), b_bounds.to_str(self.infcx())];

    if a_vid == b_vid {
        ret ok(a_t);
    }

    // If both A and B have an UB type, then we can just compute the
    // LUB of those types:
    let a_bnd = self.bnd(a_bounds), b_bnd = self.bnd(b_bounds);
    alt (a_bnd, b_bnd) {
      (some(a_ty), some(b_ty)) {
        alt self.infcx().try {|| c_ts(a_ty, b_ty) } {
            ok(t) { ret ok(t); }
            err(_) { /*fallthrough */ }
        }
      }
      _ {/*fallthrough*/}
    }

    // Otherwise, we need to merge A and B into one variable.  We can
    // then use either variable as an upper bound:
    self.infcx().vars(vb, a_vid, b_vid).then {||
        ok(a_t)
    }
}

fn c_var_t<V:copy vid, PW:pairwise, T:copy to_str st>(
    self: PW, vb: vals_and_bindings<V, T>,
    a_vid: V, b: T,
    c_ts: fn(T, T) -> cres<T>) -> cres<T> {

    let {root: a_id, bounds: a_bounds} = self.infcx().get(vb, a_vid);

    // The comments in this function are written for LUB, but they
    // apply equally well to GLB if you inverse upper/lower/sub/super/etc.

    #debug["%s.c_var_ty(%s=%s <: %s)",
           self.tag(),
           a_id.to_str(), a_bounds.to_str(self.infcx()),
           b.to_str(self.infcx())];

    alt self.bnd(a_bounds) {
      some(a_bnd) {
        // If a has an upper bound, return it.
        ret c_ts(a_bnd, b);
      }
      none {
        // If a does not have an upper bound, make b the upper bound of a
        // and then return b.
        let a_bounds = self.with_bnd(a_bounds, b);
        self.infcx().bnds(a_bounds.lb, a_bounds.ub).then {||
            self.infcx().set(vb, a_id, bounded(a_bounds));
            ok(b)
        }
      }
    }
}

fn c_tuptys<PW:pairwise>(self: PW, as: [ty::t], bs: [ty::t])
    -> cres<[ty::t]> {

    if check vec::same_length(as, bs) {
        map2(as, bs) {|a, b| self.c_tys(a, b) }
    } else {
        err(ty::terr_tuple_size(as.len(), bs.len()))
    }
}

fn c_tps<PW:pairwise>(self: PW, _did: ast::def_id, as: [ty::t], bs: [ty::t])
    -> cres<[ty::t]> {
    // FIXME #1973 lookup the declared variance of the type parameters
    // based on did
    if check vec::same_length(as, bs) {
        map2(as, bs) {|a,b| self.c_tys(a, b) }
    } else {
        err(ty::terr_ty_param_size(as.len(), bs.len()))
    }
}

fn c_fieldvecs<PW:pairwise>(
    self: PW, as: [ty::field], bs: [ty::field])
    -> cres<[ty::field]> {

    if check vec::same_length(as, bs) {
        map2(as, bs) {|a,b| self.c_flds(a, b) }
    } else {
        err(ty::terr_record_size(as.len(), bs.len()))
    }
}

fn c_flds<PW:pairwise>(
    self: PW, a: ty::field, b: ty::field) -> cres<ty::field> {

    if a.ident == b.ident {
        self.c_mts(a.mt, b.mt).chain {|mt|
            ok({ident: a.ident, mt: mt})
        }
    } else {
        err(ty::terr_record_fields(a.ident, b.ident))
    }
}

fn c_modes<PW:pairwise>(
    self: PW, a: ast::mode, b: ast::mode)
    -> cres<ast::mode> {

    let tcx = self.infcx().tcx;
    ty::unify_mode(tcx, a, b)
}

fn c_args<PW:pairwise>(
    self: PW, a: ty::arg, b: ty::arg)
    -> cres<ty::arg> {

    self.c_modes(a.mode, b.mode).chain {|m|
        // Note: contravariant
        self.c_contratys(b.ty, a.ty).chain {|t|
            ok({mode: m, ty: t})
        }
    }
}

fn c_argvecs<PW:pairwise>(
    self: PW, a_args: [ty::arg], b_args: [ty::arg]) -> cres<[ty::arg]> {

    if check vec::same_length(a_args, b_args) {
        map2(a_args, b_args) {|a, b| self.c_args(a, b) }
    } else {
        err(ty::terr_arg_count)
    }
}

fn c_fns<PW:pairwise>(
    self: PW, a_f: ty::fn_ty, b_f: ty::fn_ty) -> cres<ty::fn_ty> {

    self.c_protos(a_f.proto, b_f.proto).chain {|p|
        self.c_ret_styles(a_f.ret_style, b_f.ret_style).chain {|rs|
            self.c_argvecs(a_f.inputs, b_f.inputs).chain {|inputs|
                self.c_tys(a_f.output, b_f.output).chain {|output|
                    //FIXME self.infcx().constrvecs(a_f.constraints,
                    //FIXME                         b_f.constraints).then {||
                        ok({proto: p,
                            inputs: inputs,
                            output: output,
                            ret_style: rs,
                            constraints: a_f.constraints})
                    //FIXME }
                }
            }
        }
    }
}

fn c_tys<PW:pairwise>(
    self: PW, a: ty::t, b: ty::t) -> cres<ty::t> {

    let tcx = self.infcx().tcx;

    #debug("%s.c_tys(%s, %s)",
           self.tag(),
           ty_to_str(tcx, a),
           ty_to_str(tcx, b));

    // Fast path.
    if a == b { ret ok(a); }

    alt (ty::get(a).struct, ty::get(b).struct) {
      (ty::ty_bot, _) { self.c_bot(b) }
      (_, ty::ty_bot) { self.c_bot(b) }

      (ty::ty_var(a_id), ty::ty_var(b_id)) {
        self.c_vars(self.infcx().vb,
               a, a_id, b_id,
               {|x, y| self.c_tys(x, y) })
      }

      // Note that the LUB/GLB operations are commutative:
      (ty::ty_var(v_id), _) {
        self.c_var_t(self.infcx().vb,
                v_id, b,
                {|x, y| self.c_tys(x, y) })
      }
      (_, ty::ty_var(v_id)) {
        self.c_var_t(self.infcx().vb,
                v_id, a,
                {|x, y| self.c_tys(x, y) })
      }

      (ty::ty_nil, _) |
      (ty::ty_bool, _) |
      (ty::ty_int(_), _) |
      (ty::ty_uint(_), _) |
      (ty::ty_float(_), _) |
      (ty::ty_str, _) {
        let cfg = tcx.sess.targ_cfg;
        if ty::mach_sty(cfg, a) == ty::mach_sty(cfg, b) {
            ok(a)
        } else {
            err(ty::terr_mismatch)
        }
      }

      (ty::ty_param(a_n, _), ty::ty_param(b_n, _)) if a_n == b_n {
        ok(a)
      }

      (ty::ty_enum(a_id, a_tps), ty::ty_enum(b_id, b_tps))
      if a_id == b_id {
        self.c_tps(a_id, a_tps, b_tps).chain {|tps|
            ok(ty::mk_enum(tcx, a_id, tps))
        }
      }

      (ty::ty_iface(a_id, a_tps), ty::ty_iface(b_id, b_tps))
      if a_id == b_id {
        self.c_tps(a_id, a_tps, b_tps).chain {|tps|
            ok(ty::mk_iface(tcx, a_id, tps))
        }
      }

      (ty::ty_class(a_id, a_tps), ty::ty_class(b_id, b_tps))
      if a_id == b_id {
        // FIXME variance
        self.c_tps(a_id, a_tps, b_tps).chain {|tps|
            ok(ty::mk_class(tcx, a_id, tps))
        }
      }

      (ty::ty_box(a_mt), ty::ty_box(b_mt)) {
        self.c_mts(a_mt, b_mt).chain {|mt|
            ok(ty::mk_box(tcx, mt))
        }
      }

      (ty::ty_uniq(a_mt), ty::ty_uniq(b_mt)) {
        self.c_mts(a_mt, b_mt).chain {|mt|
            ok(ty::mk_uniq(tcx, mt))
        }
      }

      (ty::ty_vec(a_mt), ty::ty_vec(b_mt)) {
        self.c_mts(a_mt, b_mt).chain {|mt|
            ok(ty::mk_vec(tcx, mt))
        }
      }

      (ty::ty_ptr(a_mt), ty::ty_ptr(b_mt)) {
        self.c_mts(a_mt, b_mt).chain {|mt|
            ok(ty::mk_ptr(tcx, mt))
        }
      }

      (ty::ty_rptr(a_r, a_mt), ty::ty_rptr(b_r, b_mt)) {
        self.c_regions(a_r, b_r).chain {|r|
            self.c_mts(a_mt, b_mt).chain {|mt|
                ok(ty::mk_rptr(tcx, r, mt))
            }
        }
      }

      (ty::ty_res(a_id, a_t, a_tps), ty::ty_res(b_id, b_t, b_tps))
      if a_id == b_id {
        self.c_tys(a_t, b_t).chain {|t|
            self.c_tps(a_id, a_tps, b_tps).chain {|tps|
                ok(ty::mk_res(tcx, a_id, t, tps))
            }
        }
      }

      (ty::ty_rec(a_fields), ty::ty_rec(b_fields)) {
        self.c_fieldvecs(a_fields, b_fields).chain {|fs|
            ok(ty::mk_rec(tcx, fs))
        }
      }

      (ty::ty_tup(a_tys), ty::ty_tup(b_tys)) {
        self.c_tuptys(a_tys, b_tys).chain {|ts|
            ok(ty::mk_tup(tcx, ts))
        }
      }

      (ty::ty_fn(a_fty), ty::ty_fn(b_fty)) {
        self.c_fns(a_fty, b_fty).chain {|fty|
            ok(ty::mk_fn(tcx, fty))
        }
      }

      (ty::ty_constr(a_t, a_constrs), ty::ty_constr(b_t, b_constrs)) {
        self.c_tys(a_t, b_t).chain {|t|
            self.infcx().constrvecs(a_constrs, b_constrs).then {||
                ok(ty::mk_constr(tcx, t, a_constrs))
            }
        }
      }

      _ { err(ty::terr_mismatch) }
    }
}

fn c_regions<PW:pairwise>(
    self: PW, a: ty::region, b: ty::region) -> cres<ty::region> {

    #debug["%s.c_regions(%?, %?)",
           self.tag(),
           a.to_str(self.infcx()),
           b.to_str(self.infcx())];

    alt (a, b) {
      (ty::re_var(a_id), ty::re_var(b_id)) {
        self.c_vars(self.infcx().rb,
               a, a_id, b_id,
               {|x, y| self.c_regions(x, y) })
      }

      (ty::re_var(v_id), r) |
      (r, ty::re_var(v_id)) {
        self.c_var_t(self.infcx().rb,
                v_id, r,
                {|x, y| self.c_regions(x, y) })
      }

      (f @ ty::re_free(f_id, f_br), s @ ty::re_scope(s_id)) |
      (s @ ty::re_scope(s_id), f @ ty::re_free(f_id, f_br)) {
        self.c_regions_free_scope(f, f_id, f_br, s, s_id)
      }

      (ty::re_scope(a_id), ty::re_scope(b_id)) {
        self.c_regions_scope_scope(a, a_id, b, b_id)
      }

      // For these types, we cannot define any additional relationship:
      (ty::re_free(_, _), ty::re_free(_, _)) |
      (ty::re_bound(_), ty::re_bound(_)) |
      (ty::re_bound(_), ty::re_free(_, _)) |
      (ty::re_bound(_), ty::re_scope(_)) |
      (ty::re_free(_, _), ty::re_bound(_)) |
      (ty::re_scope(_), ty::re_bound(_)) {
        if a == b {
            #debug["... yes, %s == %s.",
                   a.to_str(self.infcx()),
                   b.to_str(self.infcx())];
            ok(a)
        } else {
            #debug["... no, %s != %s.",
                   a.to_str(self.infcx()),
                   b.to_str(self.infcx())];
            err(ty::terr_regions_differ(false, b, a))
        }
      }

      (ty::re_default, _) |
      (_, ty::re_default) {
        // actually a compiler bug, I think.
        err(ty::terr_regions_differ(false, b, a))
      }
    }
}
