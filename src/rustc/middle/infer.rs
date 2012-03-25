import std::smallintmap;
import std::smallintmap::smallintmap;
import std::smallintmap::map;
import middle::ty;
import syntax::ast;
import util::ppaux::{ty_to_str, mt_to_str};
import result::{result, chain, chain_err, ok, iter2};
import ty::type_is_bot;

export infer_ctxt;
export new_infer_ctxt;
export mk_subty;
export mk_eqty;
export resolve_type_structure;
export fixup_vars;
export resolve_var;
export compare_tys;

type bound = option<ty::t>;

type bounds = {lb: bound, ub: bound};

enum var_value {
    redirect(uint),
    bounded(bounds)
}

enum infer_ctxt = @{
    tcx: ty::ctxt,
    vals: smallintmap<var_value>,
    mut bindings: [(uint, var_value)]
};

type ures = result::result<(), ty::type_err>;
type fres<T> = result::result<T,int>;

fn new_infer_ctxt(tcx: ty::ctxt) -> infer_ctxt {
    infer_ctxt(@{tcx: tcx,
                 vals: smallintmap::mk(),
                 mut bindings: []})
}

fn mk_subty(cx: infer_ctxt, a: ty::t, b: ty::t) -> ures {
    #debug[">> mk_subty(%s <: %s)", cx.ty_to_str(a), cx.ty_to_str(b)];
    cx.commit {||
        cx.tys(a, b)
    }
}

fn mk_eqty(cx: infer_ctxt, a: ty::t, b: ty::t) -> ures {
    #debug["> mk_eqty(%s <: %s)", cx.ty_to_str(a), cx.ty_to_str(b)];
    cx.commit {||
        mk_subty(cx, a, b).then {||
            mk_subty(cx, b, a)
        }
    }
}

fn compare_tys(tcx: ty::ctxt, a: ty::t, b: ty::t) -> ures {
    let infcx = new_infer_ctxt(tcx);
    #debug["> compare_tys(%s == %s)", infcx.ty_to_str(a), infcx.ty_to_str(b)];
    infcx.commit {||
        mk_subty(infcx, a, b).then {||
            mk_subty(infcx, b, a)
        }
    }
}

fn resolve_type_structure(cx: infer_ctxt, a: ty::t) -> fres<ty::t> {
    cx.resolve_ty(a)
}

fn resolve_var(cx: infer_ctxt, vid: int) -> fres<ty::t> {
    cx.fixup_vars(ty::mk_var(cx.tcx, vid))
}

fn fixup_vars(cx: infer_ctxt, a: ty::t) -> fres<ty::t> {
    cx.fixup_vars(a)
}

impl methods for ures {
    fn then<T:copy>(f: fn() -> result<T,ty::type_err>)
        -> result<T,ty::type_err> {
        chain(self) {|_i| f() }
    }
}

// Most of these methods, like tys() and so forth, take two parameters
// a and b and they are tasked with "ensuring that a is a subtype of
// b".  They return success or failure.  They make changes in-place to
// the variable bindings: these changes are recorded in the `bindings`
// array, which then allows the changes to be rolled back if needed.
//
// The merge() and merge_bnds() methods are somewhat different in that
// they compute a new type range for a variable (generally a subset of
// the old range).  They therefore return a result.
impl unify_methods for infer_ctxt {
    fn uok() -> ures {
        #debug["Unification OK"];
        result::ok(())
    }

    fn uerr(e: ty::type_err) -> ures {
        #debug["Unification error: %?", e];
        result::err(e)
    }

    fn ty_to_str(t: ty::t) -> str {
        ty_to_str(self.tcx, t)
    }

    fn bound_to_str(b: bound) -> str {
        alt b {
          none { "none" }
          some(t) { self.ty_to_str(t) }
        }
    }

    fn bounds_to_str(v: bounds) -> str {
        #fmt["{%s <: X <: %s}",
             self.bound_to_str(v.lb),
             self.bound_to_str(v.ub)]
    }

    fn var_value_to_str(v: var_value) -> str {
        alt v {
          redirect(v) { #fmt["redirect(%u)", v] }
          bounded(b) { self.bounds_to_str(b) }
        }
    }

    fn set(vid: uint, +new_v: var_value) {
        let old_v = self.vals.get(vid);
        vec::push(self.bindings, (vid, old_v));

        #debug["Updating variable <T%u> from %s to %s",
               vid,
               self.var_value_to_str(old_v),
               self.var_value_to_str(new_v)];

        self.vals.insert(vid, new_v);
    }

    fn rollback_to(len: uint) {
        while self.bindings.len() != len {
            let (vid, old_v) = vec::pop(self.bindings);
            self.vals.insert(vid, old_v);
        }
    }

    fn commit<T:copy,E:copy>(f: fn() -> result<T,E>) -> result<T,E> {
        assert self.bindings.len() == 0u;
        let r = self.try(f);
        vec::clear(self.bindings);
        ret r;
    }

    fn try<T:copy,E:copy>(f: fn() -> result<T,E>) -> result<T,E> {
        let l = self.bindings.len();
        #debug["try(l=%u)", l];
        let r = f();
        alt r {
          result::ok(_) { #debug["try--ok"]; }
          result::err(_) { #debug["try--rollback"]; }
        }
        ret r;
    }

    fn get(vid: uint) -> {root: uint, bounds:bounds} {
        alt self.vals.find(vid) {
          none {
            let bnds = {lb: none, ub: none};
            self.vals.insert(vid, bounded(bnds));
            {root: vid, bounds: bnds}
          }
          some(redirect(vid)) {
            let {root, bounds} = self.get(vid);
            if root != vid {
                self.vals.insert(vid, redirect(root));
            }
            {root: root, bounds: bounds}
          }
          some(bounded(bounds)) {
            {root: vid, bounds: bounds}
          }
        }
    }

    // Combines the two bounds.  Returns a bounds r where (r.lb <:
    // a,b) and (a,b <: r.ub) (if such a bounds exists).
    fn merge_bnds(a: bound, b: bound) -> result<bounds, ty::type_err> {
        alt (a, b) {
          (none, none) {
            ok({lb: none, ub: none})
          }
          (some(_), none) {
            ok({lb: a, ub: a})
          }
          (none, some(_)) {
            ok({lb: b, ub: b})
          }
          (some(t_a), some(t_b)) {
            let r1 = self.try {||
                self.tys(t_a, t_b).then {||
                    ok({lb: a, ub: b})
                }
            };
            chain_err(r1) {|_e|
                self.tys(t_b, t_a).then {||
                    ok({lb: b, ub: a})
                }
            }
          }
        }
    }

    // Updates the bounds for the variable `v_id` to be the intersection
    // of `a` and `b`.  That is, the new bounds for `v_id` will be
    // a bounds c such that:
    //    c.ub <: a.ub
    //    c.ub <: b.ub
    //    a.lb <: c.lb
    //    b.lb <: c.lb
    // If this cannot be achieved, the result is failure.
    fn merge(v_id: uint, a: bounds, b: bounds) -> ures {
        // Think of the two diamonds, we want to find the
        // intersection.  There are basically four possibilities (you
        // can swap A/B in these pictures):
        //
        //       A         A
        //      / \       / \
        //     / B \     / B \
        //    / / \ \   / / \ \
        //   * *   * * * /   * *
        //    \ \ / /   \   / /
        //     \ B /   / \ / /
        //      \ /   *   \ /
        //       A     \ / A
        //              B

        #debug["merge(<T%u>,%s,%s)",
               v_id,
               self.bounds_to_str(a),
               self.bounds_to_str(b)];

        chain(self.merge_bnds(a.ub, b.ub)) {|ub|
            chain(self.merge_bnds(a.lb, b.lb)) {|lb|
                let bnds = {lb: lb.ub, ub: ub.lb};

                // the new bounds must themselves
                // be relatable:
                self.bnds(lb.ub, ub.lb).then {||
                    self.set(v_id, bounded(bnds));
                    self.uok()
                }
            }
        }
    }

    fn vars(a_id: uint, b_id: uint) -> ures {
        #debug["vars(<T%u> <: <T%u>)",
               a_id, b_id];

        // Need to make sub_id a subtype of sup_id.
        let {root: a_id, bounds: a_bounds} = self.get(a_id);
        let {root: b_id, bounds: b_bounds} = self.get(b_id);

        if a_id == b_id { ret self.uok(); }
        self.merge(a_id, a_bounds, b_bounds).then {||
            // For max perf, we should consider the rank here.
            self.set(b_id, redirect(a_id));
            self.uok()
        }
    }

    fn varty(a_id: uint, b: ty::t) -> ures {
        #debug["varty(<T%u> <: %s)",
               a_id, self.ty_to_str(b)];
        let {root: a_id, bounds: a_bounds} = self.get(a_id);
        let b_bounds = {lb: none, ub: some(b)};
        self.merge(a_id, a_bounds, b_bounds)
    }

    fn tyvar(a: ty::t, b_id: uint) -> ures {
        #debug["tyvar(%s <: <T%u>)",
               self.ty_to_str(a), b_id];
        let a_bounds = {lb: some(a), ub: none};
        let {root: b_id, bounds: b_bounds} = self.get(b_id);
        self.merge(b_id, a_bounds, b_bounds)
    }

    fn tyvecs(as: [ty::t], bs: [ty::t])
        : vec::same_length(as, bs) -> ures {
        iter2(as, bs) {|a,b| self.tys(a,b) }
    }

    fn regions(a: ty::region, b: ty::region) -> ures {
        alt (a, b) {
          (ty::re_var(_), _) | (_, ty::re_var(_)) {
            self.uok()  // FIXME: We need region variables!
          }
          (ty::re_inferred, _) | (_, ty::re_inferred) {
            fail "tried to unify inferred regions"
          }
          (ty::re_param(_), ty::re_param(_)) |
          (ty::re_self, ty::re_self) {
            if a == b {
                self.uok()
            } else {
                self.uerr(ty::terr_regions_differ(false, a, b))
            }
          }
          (ty::re_param(_), ty::re_block(_)) |
          (ty::re_self, ty::re_block(_)) {
            self.uok()
          }
          (ty::re_block(_), ty::re_param(_)) |
          (ty::re_block(_), ty::re_self) {
            self.uerr(ty::terr_regions_differ(false, a, b))
          }
          (ty::re_block(superblock), ty::re_block(subblock)) {
            // The region corresponding to an outer block is a subtype of the
            // region corresponding to an inner block.
            let rm = self.tcx.region_map;
            if region::scope_contains(rm, subblock, superblock) {
                self.uok()
            } else {
                self.uerr(ty::terr_regions_differ(false, a, b))
            }
          }
        }
    }

    fn mts(a: ty::mt, b: ty::mt) -> ures {
        #debug("mts(%s <: %s)",
               mt_to_str(self.tcx, a),
               mt_to_str(self.tcx, b));

        if a.mutbl != b.mutbl && b.mutbl != ast::m_const {
            ret self.uerr(ty::terr_mutability);
        }

        alt b.mutbl {
          ast::m_mutbl {
            // If supertype is mutable, subtype must mtach exactly
            // (i.e., invariant if mutable):
            self.tys(a.ty, b.ty).then {||
                self.tys(b.ty, a.ty)
            }
          }
          ast::m_imm | ast::m_const {
            // Otherwise we can be covariant:
            self.tys(a.ty, b.ty)
          }
        }
    }

    fn flds(a: ty::field, b: ty::field) -> ures {
        if a.ident != b.ident {
            ret self.uerr(ty::terr_record_fields(a.ident, b.ident));
        }
        self.mts(a.mt, b.mt)
    }

    fn tps(as: [ty::t], bs: [ty::t]) -> ures {
        if check vec::same_length(as, bs) {
            self.tyvecs(as, bs)
        } else {
            self.uerr(ty::terr_ty_param_size(as.len(), bs.len()))
        }
    }

    fn protos(a: ast::proto, b: ast::proto) -> ures {
        alt (a, b) {
          (_, ast::proto_any) { self.uok() }
          (ast::proto_bare, _) { self.uok() }
          (_, _) if a == b { self.uok() }
          _ { self.uerr(ty::terr_proto_mismatch(a, b)) }
        }
    }

    fn ret_styles(
        a_ret_style: ast::ret_style,
        b_ret_style: ast::ret_style) -> ures {

        if b_ret_style != ast::noreturn && b_ret_style != a_ret_style {
            /* even though typestate checking is mostly
               responsible for checking control flow annotations,
               this check is necessary to ensure that the
               annotation in an object method matches the
               declared object type */
            self.uerr(ty::terr_ret_style_mismatch(a_ret_style, b_ret_style))
        } else {
            self.uok()
        }
    }

    fn modes(a: ast::mode, b: ast::mode) -> ures {
        alt ty::unify_mode(self.tcx, a, b) {
          result::ok(_) { self.uok() }
          result::err(e) { self.uerr(e) }
        }
    }

    fn args(a: ty::arg, b: ty::arg) -> ures {
        self.modes(a.mode, b.mode).then {||
            self.tys(b.ty, a.ty) // Note: contravariant
        }
    }

    fn argvecs(
        a_args: [ty::arg],
        b_args: [ty::arg]) -> ures {

        if check vec::same_length(a_args, b_args) {
            iter2(a_args, b_args) {|a, b| self.args(a, b) }
        } else {
            ret self.uerr(ty::terr_arg_count);
        }
    }

    fn fns(a_f: ty::fn_ty, b_f: ty::fn_ty) -> ures {
        self.protos(a_f.proto, b_f.proto).then {||
            self.ret_styles(a_f.ret_style, b_f.ret_style).then {||
                self.argvecs(a_f.inputs, b_f.inputs).then {||
                    self.tys(a_f.output, b_f.output).then {||
                        // FIXME---constraints
                        self.uok()
                    }
                }
            }
        }
    }

    fn constrs(
        expected: @ty::type_constr,
        actual_constr: @ty::type_constr) -> ures {

        let err_res =
            self.uerr(ty::terr_constr_mismatch(expected, actual_constr));

        if expected.node.id != actual_constr.node.id { ret err_res; }
        let expected_arg_len = vec::len(expected.node.args);
        let actual_arg_len = vec::len(actual_constr.node.args);
        if expected_arg_len != actual_arg_len { ret err_res; }
        let mut i = 0u;
        for a in expected.node.args {
            let actual = actual_constr.node.args[i];
            alt a.node {
              ast::carg_base {
                alt actual.node {
                  ast::carg_base { }
                  _ { ret err_res; }
                }
              }
              ast::carg_lit(l) {
                alt actual.node {
                  ast::carg_lit(m) {
                    if l != m { ret err_res; }
                  }
                  _ { ret err_res; }
                }
              }
              ast::carg_ident(p) {
                alt actual.node {
                  ast::carg_ident(q) {
                    if p.node != q.node { ret err_res; }
                  }
                  _ { ret err_res; }
                }
              }
            }
            i += 1u;
        }
        ret self.uok();
    }

    fn bnds(a: bound, b: bound) -> ures {
        #debug("bnds(%s <: %s)",
               self.bound_to_str(a),
               self.bound_to_str(b));

        alt (a, b) {
          (none, none) |
          (some(_), none) |
          (none, some(_)) { self.uok() }
          (some(t_a), some(t_b)) { self.tys(t_a, t_b) }
        }
    }

    fn tys(a: ty::t, b: ty::t) -> ures {
        #debug("tys(%s <: %s)",
               ty_to_str(self.tcx, a),
               ty_to_str(self.tcx, b));

        // Fast path.
        if a == b { ret self.uok(); }

        alt (ty::get(a).struct, ty::get(b).struct) {
          (ty::ty_var(a_id), ty::ty_var(b_id)) {
            self.vars(a_id as uint, b_id as uint)
          }
          (ty::ty_var(a_id), _) {
            self.varty(a_id as uint, b)
          }
          (_, ty::ty_var(b_id)) {
            self.tyvar(a, b_id as uint)
          }

          (_, ty::ty_bot) { self.uok() }
          (ty::ty_bot, _) { self.uok() }

          (ty::ty_nil, _) |
          (ty::ty_bool, _) |
          (ty::ty_int(_), _) |
          (ty::ty_uint(_), _) |
          (ty::ty_float(_), _) |
          (ty::ty_str, _) {
            let cfg = self.tcx.sess.targ_cfg;
            if ty::mach_sty(cfg, a) == ty::mach_sty(cfg, b) {
                self.uok()
            } else {
                self.uerr(ty::terr_mismatch)
            }
          }

          (ty::ty_param(a_n, _), ty::ty_param(b_n, _))
          if a_n == b_n {
            self.uok()
          }

          (ty::ty_enum(a_id, a_tps), ty::ty_enum(b_id, b_tps)) |
          (ty::ty_iface(a_id, a_tps), ty::ty_iface(b_id, b_tps)) |
          (ty::ty_class(a_id, a_tps), ty::ty_class(b_id, b_tps))
          if a_id == b_id {
            self.tps(a_tps, b_tps)
          }

          (ty::ty_box(a_mt), ty::ty_box(b_mt)) |
          (ty::ty_uniq(a_mt), ty::ty_uniq(b_mt)) |
          (ty::ty_vec(a_mt), ty::ty_vec(b_mt)) |
          (ty::ty_ptr(a_mt), ty::ty_ptr(b_mt)) {
            self.mts(a_mt, b_mt)
          }

          (ty::ty_rptr(a_r, a_mt), ty::ty_rptr(b_r, b_mt)) {
            self.mts(a_mt, b_mt).then {||
                self.regions(a_r, b_r)
            }
          }

          (ty::ty_res(a_id, a_t, a_tps), ty::ty_res(b_id, b_t, b_tps))
          if a_id == b_id {
            self.tys(a_t, b_t).then {||
                self.tps(a_tps, b_tps)
            }
          }

          (ty::ty_rec(a_fields), ty::ty_rec(b_fields)) {
            if check vec::same_length(a_fields, b_fields) {
                iter2(a_fields, b_fields) {|a,b|
                    self.flds(a, b)
                }
            } else {
                ret self.uerr(ty::terr_record_size(a_fields.len(),
                                             b_fields.len()));
            }
          }

          (ty::ty_tup(a_tys), ty::ty_tup(b_tys)) {
            if check vec::same_length(a_tys, b_tys) {
                self.tyvecs(a_tys, b_tys)
            } else {
                self.uerr(ty::terr_tuple_size(a_tys.len(), b_tys.len()))
            }
          }

          (ty::ty_fn(a_fty), ty::ty_fn(b_fty)) {
            self.fns(a_fty, b_fty)
          }

          (ty::ty_constr(a_t, a_constrs), ty::ty_constr(b_t, b_constrs)) {
            self.tys(a_t, b_t).then {||
                if check vec::same_length(a_constrs, b_constrs) {
                    iter2(a_constrs, b_constrs) {|a,b|
                        self.constrs(a, b)
                    }
                } else {
                    ret self.uerr(ty::terr_constr_len(a_constrs.len(),
                                                b_constrs.len()));
                }
            }
          }

          _ { self.uerr(ty::terr_mismatch) }
        }
    }
}

impl resolve_methods for infer_ctxt {
    fn rok(t: ty::t) -> fres<ty::t> {
        #debug["Resolve OK: %s", self.ty_to_str(t)];
        result::ok(t)
    }

    fn rerr(v: int) -> fres<ty::t> {
        #debug["Resolve error: %?", v];
        result::err(v)
    }

    fn resolve_var(vid: int) -> fres<ty::t> {
        let {root:_, bounds} = self.get(vid as uint);

        // Nonobvious: prefer the most specific type
        // (i.e., the lower bound) to the more general
        // one.  More general types in Rust (e.g., fn())
        // tend to carry more restrictions or higher
        // perf. penalties, so it pays to know more.

        alt bounds {
          { ub:_, lb:some(t) } if !type_is_bot(t) { self.rok(t) }
          { ub:some(t), lb:_ } { self.rok(t) }
          { ub:_, lb:some(t) } { self.rok(t) }
          { ub:none, lb:none } { self.rerr(vid) }
        }
    }

    fn resolve_ty(typ: ty::t) -> fres<ty::t> {
        alt ty::get(typ).struct {
          ty::ty_var(vid) { self.resolve_var(vid) }
          _ { self.rok(typ) }
        }
    }

    fn subst_vars(unresolved: @mutable option<int>,
                  vars_seen: std::list::list<int>,
                  vid: int) -> ty::t {
        // Should really return a fixup_result instead of a t, but fold_ty
        // doesn't allow returning anything but a t.
        alt self.resolve_var(vid) {
          result::err(vid) {
            *unresolved = some(vid);
            ret ty::mk_var(self.tcx, vid);
          }
          result::ok(rt) {
            let mut give_up = false;
            std::list::iter(vars_seen) {|v|
                if v == vid {
                    *unresolved = some(-1); // hack: communicate inf ty
                    give_up = true;
                }
            }

            // Return the type unchanged, so we can error out
            // downstream
            if give_up { ret rt; }
            ret ty::fold_ty(self.tcx,
                            ty::fm_var(
                                self.subst_vars(
                                    unresolved,
                                    std::list::cons(vid, @vars_seen),
                                    _)),
                            rt);
          }
        }
    }

    fn fixup_vars(typ: ty::t) -> fres<ty::t> {
        let unresolved = @mutable none::<int>;
        let rty =
            ty::fold_ty(self.tcx,
                        ty::fm_var(
                            self.subst_vars(
                                unresolved,
                                std::list::nil,
                                _)),
                        typ);

        let ur = *unresolved;
        alt ur {
          none { ret self.rok(rty); }
          some(var_id) { ret self.rerr(var_id); }
        }
    }
}
