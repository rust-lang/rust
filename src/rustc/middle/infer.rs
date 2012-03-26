import std::smallintmap;
import std::smallintmap::smallintmap;
import std::smallintmap::map;
import middle::ty;
import syntax::ast;
import syntax::ast::{ret_style};
import util::ppaux::{ty_to_str, mt_to_str};
import result::{result, methods, chain, chain_err, ok, err, map, map2, iter2};
import ty::type_is_bot;

export infer_ctxt;
export new_infer_ctxt;
export mk_subty;
export mk_eqty;
export resolve_type_structure;
export fixup_vars;
export resolve_var;
export compare_tys;

type bound<T:copy> = option<T>;

type bounds<T:copy> = {lb: bound<T>, ub: bound<T>};

enum var_value<T:copy> {
    redirect(uint),
    bounded(bounds<T>)
}

type vals_and_bindings<T:copy> = {
    vals: smallintmap<var_value<T>>,
    mut bindings: [(uint, var_value<T>)]
};

enum infer_ctxt = @{
    tcx: ty::ctxt,
    vb: vals_and_bindings<ty::t>,
    rb: vals_and_bindings<ty::region>,
};

type ures = result::result<(), ty::type_err>;
type fres<T> = result::result<T,int>;

fn new_infer_ctxt(tcx: ty::ctxt) -> infer_ctxt {
    infer_ctxt(@{tcx: tcx,
                 vb: {vals: smallintmap::mk(), mut bindings: []},
                 rb: {vals: smallintmap::mk(), mut bindings: []}})
}

fn mk_subty(cx: infer_ctxt, a: ty::t, b: ty::t) -> ures {
    #debug[">> mk_subty(%s <: %s)", a.to_str(cx), b.to_str(cx)];
    cx.commit {||
        cx.tys(a, b)
    }
}

fn mk_eqty(cx: infer_ctxt, a: ty::t, b: ty::t) -> ures {
    #debug[">> mk_eqty(%s <: %s)", a.to_str(cx), b.to_str(cx)];
    cx.commit {||
        cx.eq_tys(a, b)
    }
}

fn compare_tys(tcx: ty::ctxt, a: ty::t, b: ty::t) -> ures {
    let infcx = new_infer_ctxt(tcx);
    #debug[">> compare_tys(%s == %s)", a.to_str(infcx), b.to_str(infcx)];
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

iface to_str {
    fn to_str(cx: infer_ctxt) -> str;
}

impl of to_str for ty::t {
    fn to_str(cx: infer_ctxt) -> str {
        ty_to_str(cx.tcx, self)
    }
}

impl of to_str for ty::mt {
    fn to_str(cx: infer_ctxt) -> str {
        mt_to_str(cx.tcx, self)
    }
}

impl of to_str for ty::region {
    fn to_str(cx: infer_ctxt) -> str {
        util::ppaux::region_to_str(cx.tcx, self)
    }
}

impl<V:copy to_str> of to_str for bound<V> {
    fn to_str(cx: infer_ctxt) -> str {
        alt self {
          some(v) { v.to_str(cx) }
          none { "none " }
        }
    }
}

impl<V:copy to_str> of to_str for bounds<V> {
    fn to_str(cx: infer_ctxt) -> str {
        #fmt["{%s <: %s}",
             self.lb.to_str(cx),
             self.ub.to_str(cx)]
    }
}

impl<V:copy to_str> of to_str for var_value<V> {
    fn to_str(cx: infer_ctxt) -> str {
        alt self {
          redirect(id) { #fmt("redirect(%u)", id) }
          bounded(bnds) { #fmt("bounded(%s)", bnds.to_str(cx)) }
        }
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
        ok(())
    }

    fn uerr(e: ty::type_err) -> ures {
        #debug["Unification error: %?", e];
        err(e)
    }

    fn set<T:copy to_str>(
        vb: vals_and_bindings<T>, vid: uint,
        +new_v: var_value<T>) {

        let old_v = vb.vals.get(vid);
        vec::push(vb.bindings, (vid, old_v));
        vb.vals.insert(vid, new_v);

        #debug["Updating variable <%u> from %s to %s",
               vid, old_v.to_str(self), new_v.to_str(self)];
    }

    fn set_ty(vid: uint, +new_v: var_value<ty::t>) {
        let old_v = self.vb.vals.get(vid);
        self.set(self.vb, vid, new_v);

        #debug["Updating variable <T%u> from %s to %s",
               vid, old_v.to_str(self), new_v.to_str(self)];
    }

    fn rollback_to<T:copy>(vb: vals_and_bindings<T>, len: uint) {
        while vb.bindings.len() != len {
            let (vid, old_v) = vec::pop(vb.bindings);
            vb.vals.insert(vid, old_v);
        }
    }

    fn commit<T:copy,E:copy>(f: fn() -> result<T,E>) -> result<T,E> {

        assert self.vb.bindings.len() == 0u;
        assert self.rb.bindings.len() == 0u;

        let r = self.try(f);

        // TODO---could use a vec::clear() that ran destructors but kept
        // the vec at its currently allocated length
        self.vb.bindings = [];
        self.rb.bindings = [];

        #debug[">> Commit result: %?", r];

        ret r;
    }

    fn try<T:copy,E:copy>(f: fn() -> result<T,E>) -> result<T,E> {

        fn rollback_to<T:copy>(vb: vals_and_bindings<T>, len: uint) {
            while vb.bindings.len() != len {
                let (vid, old_v) = vec::pop(vb.bindings);
                vb.vals.insert(vid, old_v);
            }
        }

        let vbl = self.vb.bindings.len();
        let rbl = self.rb.bindings.len();
        #debug["try(vbl=%u, rbl=%u)", vbl, rbl];
        let r = f();
        alt r {
          result::ok(_) { #debug["try--ok"]; }
          result::err(_) {
            #debug["try--rollback"];
            rollback_to(self.vb, vbl);
            rollback_to(self.rb, rbl);
          }
        }
        ret r;
    }

    fn get<T:copy>(vb: vals_and_bindings<T>, vid: uint)
        -> {root: uint, bounds:bounds<T>} {

        alt vb.vals.find(vid) {
          none {
            let bnds = {lb: none, ub: none};
            vb.vals.insert(vid, bounded(bnds));
            {root: vid, bounds: bnds}
          }
          some(redirect(vid)) {
            let {root, bounds} = self.get(vb, vid);
            if root != vid {
                vb.vals.insert(vid, redirect(root));
            }
            {root: root, bounds: bounds}
          }
          some(bounded(bounds)) {
            {root: vid, bounds: bounds}
          }
        }
    }

    fn get_var(vid: uint) -> {root: uint, bounds:bounds<ty::t>} {
        ret self.get(self.vb, vid);
    }

    fn get_region(rid: uint) -> {root: uint, bounds:bounds<ty::region>} {
        ret self.get(self.rb, rid);
    }

    // Combines the two bounds into a more general bound.
    fn merge_bnd<V:copy to_str>(
        a: bound<V>, b: bound<V>,
        merge_op: fn(V,V) -> cres<V>) -> cres<bound<V>> {

        alt (a, b) {
          (none, none) {
            ok(none)
          }
          (some(_), none) {
            ok(a)
          }
          (none, some(_)) {
            ok(b)
          }
          (some(v_a), some(v_b)) {
            merge_op(v_a, v_b).chain {|v|
                ok(some(v))
            }
          }
        }
    }

    fn merge_bnds<V:copy to_str>(
        a: bounds<V>, b: bounds<V>,
        lub: fn(V,V) -> cres<V>,
        glb: fn(V,V) -> cres<V>) -> cres<bounds<V>> {

        self.merge_bnd(a.ub, b.ub, glb).chain {|ub|
            #debug["glb of ubs %s and %s is %s",
                   a.ub.to_str(self), b.ub.to_str(self),
                   ub.to_str(self)];
            self.merge_bnd(a.lb, b.lb, lub).chain {|lb|
                #debug["lub of lbs %s and %s is %s",
                       a.lb.to_str(self), b.lb.to_str(self),
                       lb.to_str(self)];
                ok({lb: lb, ub: ub})
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
    fn set_ty_var_to_merged_bounds(
        v_id: uint, a: bounds<ty::t>, b: bounds<ty::t>) -> ures {

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
               a.to_str(self),
               b.to_str(self)];

        // First, relate the lower/upper bounds of A and B.
        // Note that these relations *must* hold for us to
        // to be able to merge A and B at all, and relating
        // them explicitly gives the type inferencer more
        // information and helps to produce tighter bounds
        // when necessary.
        self.bnds(a.lb, b.ub).then {||
        self.bnds(b.lb, a.ub).then {||
        self.merge_bnds(
            a, b,
            {|a_ty, b_ty| lub(self).c_tys(a_ty, b_ty) },
            {|a_ty, b_ty| glb(self).c_tys(a_ty, b_ty) }).chain {|bnds|

            #debug["merge(<T%u>): bnds=%s",
                   v_id,
                   bnds.to_str(self)];

            // the new bounds must themselves
            // be relatable:
            self.bnds(bnds.lb, bnds.ub).then {||
            self.set_ty(v_id, bounded(bnds));
            self.uok()
            }
        }}}
    }

    // TODO: Generalize to regions.
    fn vars(a_id: uint, b_id: uint) -> ures {
        // Need to make sub_id a subtype of sup_id.
        let {root: a_id, bounds: a_bounds} = self.get(self.vb, a_id);
        let {root: b_id, bounds: b_bounds} = self.get(self.vb, b_id);

        #debug["vars(<T%u>=%s <: <T%u>=%s)",
               a_id, a_bounds.to_str(self),
               b_id, b_bounds.to_str(self)];

        if a_id == b_id { ret self.uok(); }

        // If both A's UB and B's LB have already been bound to types,
        // see if we can make those types subtypes.
        alt (a_bounds.ub, b_bounds.lb) {
          (some(a_ub), some(b_lb)) {
            let r = self.try {|| self.tys(a_ub, b_lb) };
            alt r {
              ok(()) { ret result::ok(()); }
              err(_) { /*fallthrough */ }
            }
          }
          _ { /*fallthrough*/ }
        }

        // For max perf, we should consider the rank here.  But for now,
        // we always make b redirect to a.
        self.set_ty(b_id, redirect(a_id));

        // Otherwise, we need to merge A and B so as to guarantee that
        // A remains a subtype of B.  Actually, there are other options,
        // but that's the route we choose to take.
        self.set_ty_var_to_merged_bounds(a_id, a_bounds, b_bounds).then {||
            self.uok()
        }
    }

    fn varty(a_id: uint, b: ty::t) -> ures {
        let {root: a_id, bounds: a_bounds} = self.get(self.vb, a_id);
        #debug["varty(<T%u>=%s <: %s)",
               a_id, a_bounds.to_str(self),
               b.to_str(self)];
        let b_bounds = {lb: none, ub: some(b)};
        self.set_ty_var_to_merged_bounds(a_id, a_bounds, b_bounds)
    }

    fn tyvar(a: ty::t, b_id: uint) -> ures {
        let a_bounds = {lb: some(a), ub: none};
        let {root: b_id, bounds: b_bounds} = self.get(self.vb, b_id);
        #debug["tyvar(%s <: <T%u>=%s)",
               a.to_str(self),
               b_id, b_bounds.to_str(self)];
        self.set_ty_var_to_merged_bounds(b_id, a_bounds, b_bounds)
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
        #debug("mts(%s <: %s)", a.to_str(self), b.to_str(self));

        if a.mutbl != b.mutbl && b.mutbl != ast::m_const {
            ret self.uerr(ty::terr_mutability);
        }

        alt b.mutbl {
          ast::m_mutbl {
            // If supertype is mutable, subtype must match exactly
            // (i.e., invariant if mutable):
            self.eq_tys(a.ty, b.ty)
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
            iter2(as, bs) {|a, b| self.tys(a, b) }
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
        a_ret_style: ret_style,
        b_ret_style: ret_style) -> ures {

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
          ok(_) { self.uok() }
          err(e) { self.uerr(e) }
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
                        //TODO self.constrvecs(a_f.constraints,
                        //TODO                 b_f.constraints).then {||
                            self.uok()
                        //TODO }
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

    // TODO: Generalize this.
    fn bnds(a: bound<ty::t>, b: bound<ty::t>) -> ures {
        #debug("bnds(%s <: %s)",
               a.to_str(self),
               b.to_str(self));

        alt (a, b) {
          (none, none) |
          (some(_), none) |
          (none, some(_)) {
            self.uok()
          }
          (some(t_a), some(t_b)) {
            self.tys(t_a, t_b)
          }
        }
    }

    fn constrvecs(
        as: [@ty::type_constr], bs: [@ty::type_constr]) -> ures {

        if check vec::same_length(as, bs) {
            iter2(as, bs) {|a,b|
                self.constrs(a, b)
            }
        } else {
            self.uerr(ty::terr_constr_len(as.len(), bs.len()))
        }
    }

    fn eq_tys(a: ty::t, b: ty::t) -> ures {
        self.tys(a, b).then {||
            self.tys(b, a)
        }
    }

    fn tys(a: ty::t, b: ty::t) -> ures {
        #debug("tys(%s <: %s)",
               ty_to_str(self.tcx, a),
               ty_to_str(self.tcx, b));

        // Fast path.
        if a == b { ret self.uok(); }

        alt (ty::get(a).struct, ty::get(b).struct) {
          (ty::ty_bot, _) { self.uok() }

          (ty::ty_var(a_id), ty::ty_var(b_id)) {
            self.vars(a_id as uint, b_id as uint)
          }
          (ty::ty_var(a_id), _) {
            self.varty(a_id as uint, b)
          }
          (_, ty::ty_var(b_id)) {
            self.tyvar(a, b_id as uint)
          }

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
                iter2(a_tys, b_tys) {|a,b| self.tys(a,b) }
            } else {
                self.uerr(ty::terr_tuple_size(a_tys.len(), b_tys.len()))
            }
          }

          (ty::ty_fn(a_fty), ty::ty_fn(b_fty)) {
            self.fns(a_fty, b_fty)
          }

          (ty::ty_constr(a_t, a_constrs), ty::ty_constr(b_t, b_constrs)) {
            self.tys(a_t, b_t).then {||
                self.constrvecs(a_constrs, b_constrs)
            }
          }

          _ { self.uerr(ty::terr_mismatch) }
        }
    }
}

impl resolve_methods for infer_ctxt {
    fn rok(t: ty::t) -> fres<ty::t> {
        #debug["Resolve OK: %s", t.to_str(self)];
        ok(t)
    }

    fn rerr<T>(v: int) -> fres<T> {
        #debug["Resolve error: %?", v];
        err(v)
    }

    fn resolve_var<T:copy to_str>(
        vb: vals_and_bindings<T>, bot_guard: fn(T)->bool,
        vid: int) -> fres<T> {

        let {root:_, bounds} = self.get(vb, vid as uint);

        #debug["resolve_var(%d) bounds=%s",
               vid, bounds.to_str(self)];

        // Nonobvious: prefer the most specific type
        // (i.e., the lower bound) to the more general
        // one.  More general types in Rust (e.g., fn())
        // tend to carry more restrictions or higher
        // perf. penalties, so it pays to know more.

        alt bounds {
          { ub:_, lb:some(t) } if !bot_guard(t) { ok(t) }
          { ub:some(t), lb:_ } { ok(t) }
          { ub:_, lb:some(t) } { ok(t) }
          { ub:none, lb:none } { self.rerr(vid) }
        }
    }

    fn resolve_ty_var(vid: int) -> fres<ty::t> {
        ret self.resolve_var(self.vb, {|t| type_is_bot(t)}, vid);
    }

    fn resolve_region_var(rid: int) -> fres<ty::region> {
        ret self.resolve_var(self.rb, {|_t| false}, rid);
    }

    fn resolve_ty(typ: ty::t) -> fres<ty::t> {
        alt ty::get(typ).struct {
          ty::ty_var(vid) { self.resolve_ty_var(vid) }
          ty::ty_rptr(ty::re_var(rid), base_ty) {
            alt self.resolve_region_var(rid as int) {
              err(terr)  { err(terr) }
              ok(region) {
                self.rok(ty::mk_rptr(self.tcx, region, base_ty))
              }
            }
          }
          _ { self.rok(typ) }
        }
    }

    fn subst_vars(unresolved: @mutable option<int>,
                  vars_seen: std::list::list<int>,
                  vid: int) -> ty::t {
        // Should really return a fixup_result instead of a t, but fold_ty
        // doesn't allow returning anything but a t.
        alt self.resolve_ty_var(vid) {
          err(vid) {
            *unresolved = some(vid);
            ret ty::mk_var(self.tcx, vid);
          }
          ok(rt) {
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

    // FIXME: These should be integrated with the two functions above instead
    // of being such blatant lazy duplicates.

    fn subst_regions(unresolved: @mutable option<int>,
                     regions_seen: std::list::list<int>,
                     rid: int) -> ty::region {
        // Should really return a fixup_result instead of a t, but fold_ty
        // doesn't allow returning anything but a t.
        alt self.resolve_region_var(rid) {
          err(rid) {
            *unresolved = some(rid);
            ret ty::re_var(rid as uint);
          }
          ok(rr) {
            let mut give_up = false;
            std::list::iter(regions_seen) {|r|
                if r == rid {
                    *unresolved = some(-1); // hack: communicate inf region
                    give_up = true;
                }
            }
            ret rr;
          }
        }
    }

    fn fixup_regions(typ: ty::t) -> fres<ty::t> {
        let unresolved = @mutable none::<int>;
        let rty = ty::fold_ty(self.tcx, ty::fm_rptr({ |region, _under_rptr|
            alt region {
              ty::re_var(rid) {
                self.subst_regions(unresolved, std::list::nil, rid as int)
              }
              _ { region }
            }
        }), typ);

        let ur = *unresolved;
        alt ur {
          none { ret self.rok(rty); }
          some(var_id) { ret self.rerr(var_id); }
        }
    }
}

// ______________________________________________________________________
// Type combining
//
// There are two type combiners, lub and gub.  The first computes the
// Least Upper Bound of two types `a` and `b`---that is, a mutual
// supertype type `c` where `a <: c` and `a <: c`.  As the name
// implies, it tries to pick the most precise `c` possible.  `glb`
// computes the greatest lower bound---that is, it computes a mutual
// subtype, aiming for the most general such type possible.  Both
// computations may fail.
//
// There is a lot of common code for these operations, which is
// abstracted out into functions named `c_X()` which take a combiner
// instance as the first parameter.  This would be better implemented
// using traits.
//
// In principle, the subtyping relation computed above could be built
// on the combine framework---this would result in less code but would
// be less efficient.  There is a significant performance gain from
// not recreating types unless we need to.  Even so, we could write
// the routines with a few more generics in there to mask type
// construction (which is, after all, the significant expense) but I
// haven't gotten around to it.

type cres<T> = result<T,ty::type_err>;

iface combine {
    fn infcx() -> infer_ctxt;
    fn tag() -> str;
    fn bnd<V:copy>(b: bounds<V>) -> option<V>;
    fn with_bnd<V:copy>(b: bounds<V>, v: V) -> bounds<V>;
    fn c_bot(b: ty::t) -> cres<ty::t>;
    fn c_regions(a: ty::region, b: ty::region) -> cres<ty::region>;
    fn c_mts(a: ty::mt, b: ty::mt) -> cres<ty::mt>;
    fn c_contratys(t1: ty::t, t2: ty::t) -> cres<ty::t>;
    fn c_tys(t1: ty::t, t2: ty::t) -> cres<ty::t>;
    fn c_protos(p1: ast::proto, p2: ast::proto) -> cres<ast::proto>;
    fn c_ret_styles(r1: ret_style, r2: ret_style) -> cres<ret_style>;
}

enum lub = infer_ctxt;
enum glb = infer_ctxt;

fn c_ty_vars<C:combine>(self: C, a_id: uint, b_id: uint) -> cres<ty::t> {
    // Need to find a type that is a supertype of both a and b:
    let {root: a_id, bounds: a_bounds} = self.infcx().get_var(a_id);
    let {root: b_id, bounds: b_bounds} = self.infcx().get_var(b_id);

    #debug["%s.c_ty_vars(<T%u>=%s <: <T%u>=%s)",
           self.tag(),
           a_id, a_bounds.to_str(self.infcx()),
           b_id, b_bounds.to_str(self.infcx())];

    let tcx = self.infcx().tcx;

    if a_id == b_id {
        ret ok(ty::mk_var(tcx, a_id as int));
    }

    // The comments in this function are written for LUB, but they
    // apply equally well to GLB if you inverse upper/lower/sub/super/etc.

    // If both A and B have an UB type, then we can just compute the
    // LUB of those types:
    let a_bnd = self.bnd(a_bounds), b_bnd = self.bnd(b_bounds);
    alt (a_bnd, b_bnd) {
      (some(a_ty), some(b_ty)) {
        alt self.infcx().try {|| self.c_tys(a_ty, b_ty) } {
            ok(t) { ret ok(t); }
            err(_) { /*fallthrough */ }
        }
      }
      _ {/*fallthrough*/}
    }

    // Otherwise, we need to merge A and B into one variable.  We can
    // then use either variable as an upper bound:
    self.infcx().vars(a_id, b_id).then {||
        ok(ty::mk_var(tcx, a_id as int))
    }
}

fn c_ty_var_ty<C:combine>(self: C, a_id: uint, b: ty::t) -> cres<ty::t> {
    let {root: a_id, bounds: a_bounds} = self.infcx().get_var(a_id);

    // The comments in this function are written for LUB, but they
    // apply equally well to GLB if you inverse upper/lower/sub/super/etc.

    #debug["%s.c_ty_var_ty(<T%u>=%s <: %s)",
           self.tag(),
           a_id, a_bounds.to_str(self.infcx()),
           b.to_str(self.infcx())];

    alt self.bnd(a_bounds) {
      some(a_ty) {
        // If a has an upper bound, return it.
        ret self.c_tys(a_ty, b);
      }
      none {
        // If a does not have an upper bound, make b the upper bound of a
        // and then return b.
        let a_bounds = self.with_bnd(a_bounds, b);
        self.infcx().bnds(a_bounds.lb, a_bounds.ub).then {||
            self.infcx().set_ty(a_id, bounded(a_bounds));
            ok(b)
        }
      }
    }
}

fn c_tuptys<C:combine>(self: C, as: [ty::t], bs: [ty::t])
    -> cres<[ty::t]> {

    if check vec::same_length(as, bs) {
        map2(as, bs) {|a, b| self.c_tys(a, b) }
    } else {
        err(ty::terr_tuple_size(as.len(), bs.len()))
    }
}

fn c_tps<C:combine>(self: C, _did: ast::def_id, as: [ty::t], bs: [ty::t])
    -> cres<[ty::t]> {
    // FIXME #1973 lookup the declared variance of the type parameters
    // based on did
    if check vec::same_length(as, bs) {
        map2(as, bs) {|a,b| self.c_tys(a, b) }
    } else {
        err(ty::terr_ty_param_size(as.len(), bs.len()))
    }
}

fn c_fieldvecs<C:combine>(self: C, as: [ty::field], bs: [ty::field])
    -> cres<[ty::field]> {

    if check vec::same_length(as, bs) {
        map2(as, bs) {|a,b| c_flds(self, a, b) }
    } else {
        err(ty::terr_record_size(as.len(), bs.len()))
    }
}

fn c_flds<C:combine>(self: C, a: ty::field, b: ty::field) -> cres<ty::field> {
    if a.ident == b.ident {
        self.c_mts(a.mt, b.mt).chain {|mt|
            ok({ident: a.ident, mt: mt})
        }
    } else {
        err(ty::terr_record_fields(a.ident, b.ident))
    }
}

fn c_modes<C:combine>(self: C, a: ast::mode, b: ast::mode)
    -> cres<ast::mode> {

    let tcx = self.infcx().tcx;
    ty::unify_mode(tcx, a, b)
}

fn c_args<C:combine>(self: C, a: ty::arg, b: ty::arg)
    -> cres<ty::arg> {

    c_modes(self, a.mode, b.mode).chain {|m|
        // Note: contravariant
        self.c_contratys(b.ty, a.ty).chain {|t|
            ok({mode: m, ty: t})
        }
    }
}

fn c_argvecs<C:combine>(
    self: C, a_args: [ty::arg], b_args: [ty::arg]) -> cres<[ty::arg]> {

    if check vec::same_length(a_args, b_args) {
        map2(a_args, b_args) {|a, b| c_args(self, a, b) }
    } else {
        err(ty::terr_arg_count)
    }
}

fn c_fns<C:combine>(
    self: C, a_f: ty::fn_ty, b_f: ty::fn_ty) -> cres<ty::fn_ty> {

    self.c_protos(a_f.proto, b_f.proto).chain {|p|
        self.c_ret_styles(a_f.ret_style, b_f.ret_style).chain {|rs|
            c_argvecs(self, a_f.inputs, b_f.inputs).chain {|inputs|
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

fn c_tys<C:combine>(
    self: C, a: ty::t, b: ty::t) -> cres<ty::t> {

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
        c_ty_vars(self, a_id as uint, b_id as uint)
      }

      // Note that the LUB/GLB operations are commutative:
      (ty::ty_var(a_id), _) {
        c_ty_var_ty(self, a_id as uint, b)
      }
      (_, ty::ty_var(b_id)) {
        c_ty_var_ty(self, b_id as uint, a)
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
        c_tps(self, a_id, a_tps, b_tps).chain {|tps|
            ok(ty::mk_enum(tcx, a_id, tps))
        }
      }

      (ty::ty_iface(a_id, a_tps), ty::ty_iface(b_id, b_tps))
      if a_id == b_id {
        c_tps(self, a_id, a_tps, b_tps).chain {|tps|
            ok(ty::mk_iface(tcx, a_id, tps))
        }
      }

      (ty::ty_class(a_id, a_tps), ty::ty_class(b_id, b_tps))
      if a_id == b_id {
        // FIXME variance
        c_tps(self, a_id, a_tps, b_tps).chain {|tps|
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
            c_tps(self, a_id, a_tps, b_tps).chain {|tps|
                ok(ty::mk_res(tcx, a_id, t, tps))
            }
        }
      }

      (ty::ty_rec(a_fields), ty::ty_rec(b_fields)) {
        c_fieldvecs(self, a_fields, b_fields).chain {|fs|
            ok(ty::mk_rec(tcx, fs))
        }
      }

      (ty::ty_tup(a_tys), ty::ty_tup(b_tys)) {
        c_tuptys(self, a_tys, b_tys).chain {|ts|
            ok(ty::mk_tup(tcx, ts))
        }
      }

      (ty::ty_fn(a_fty), ty::ty_fn(b_fty)) {
        c_fns(self, a_fty, b_fty).chain {|fty|
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

impl of combine for lub {
    fn infcx() -> infer_ctxt { *self }

    fn tag() -> str { "lub" }

    fn bnd<V:copy>(b: bounds<V>) -> option<V> {
        b.ub
    }

    fn with_bnd<V:copy>(b: bounds<V>, v: V) -> bounds<V> {
        assert b.ub == none;
        {ub: some(v) with b}
    }

    fn c_bot(b: ty::t) -> cres<ty::t> {
        ok(b)
    }

    fn c_regions(a: ty::region, _b: ty::region) -> cres<ty::region> {
        ok(a) // FIXME
    }

    fn c_mts(a: ty::mt, b: ty::mt) -> cres<ty::mt> {
        let tcx = self.infcx().tcx;

        #debug("%s.c_mts(%s, %s)",
               self.tag(),
               mt_to_str(tcx, a),
               mt_to_str(tcx, b));

        let m = if a.mutbl == b.mutbl {
            a.mutbl
        } else {
            ast::m_const
        };

        alt m {
          ast::m_imm | ast::m_const {
            self.c_tys(a.ty, b.ty).chain {|t|
                ok({ty: t, mutbl: m})
            }
          }

          ast::m_mutbl {
            self.infcx().try {||
                self.infcx().eq_tys(a.ty, b.ty).then {||
                    ok({ty: a.ty, mutbl: m})
                }
            }.chain_err {|_e|
                self.c_tys(a.ty, b.ty).chain {|t|
                    ok({ty: t, mutbl: ast::m_const})
                }
            }
          }
        }
    }

    fn c_contratys(a: ty::t, b: ty::t) -> cres<ty::t> {
        glb(self.infcx()).c_tys(a, b)
    }

    fn c_tys(a: ty::t, b: ty::t) -> cres<ty::t> {
        c_tys(self, a, b)
    }

    fn c_protos(p1: ast::proto, p2: ast::proto) -> cres<ast::proto> {
        if p1 == ast::proto_bare {
            ok(p2)
        } else if p2 == ast::proto_bare {
            ok(p1)
        } else if p1 == p2 {
            ok(p1)
        } else {
            ok(ast::proto_any)
        }
    }

    fn c_ret_styles(r1: ret_style, r2: ret_style) -> cres<ret_style> {
        alt (r1, r2) {
          (ast::return_val, _) |
          (_, ast::return_val) {
            ok(ast::return_val)
          }
          (ast::noreturn, ast::noreturn) {
            ok(ast::noreturn)
          }
        }
    }
}

impl of combine for glb {
    fn infcx() -> infer_ctxt { *self }

    fn tag() -> str { "glb" }

    fn bnd<V:copy>(b: bounds<V>) -> option<V> {
        b.lb
    }

    fn with_bnd<V:copy>(b: bounds<V>, v: V) -> bounds<V> {
        assert b.lb == none;
        {lb: some(v) with b}
    }

    fn c_bot(_b: ty::t) -> cres<ty::t> {
        ok(ty::mk_bot(self.infcx().tcx))
    }

    fn c_regions(a: ty::region, _b: ty::region) -> cres<ty::region> {
        ok(a) // FIXME
    }

    fn c_mts(a: ty::mt, b: ty::mt) -> cres<ty::mt> {
        let tcx = self.infcx().tcx;

        #debug("%s.c_mts(%s, %s)",
               self.tag(),
               mt_to_str(tcx, a),
               mt_to_str(tcx, b));

        alt (a.mutbl, b.mutbl) {
          // If one side or both is mutable, then the GLB must use
          // the precise type from the mutable side.
          (ast::m_mutbl, ast::m_const) {
            self.infcx().tys(a.ty, b.ty).then {||
                ok({ty: a.ty, mutbl: ast::m_mutbl})
            }
          }
          (ast::m_const, ast::m_mutbl) {
            self.infcx().tys(b.ty, a.ty).then {||
                ok({ty: b.ty, mutbl: ast::m_mutbl})
            }
          }
          (ast::m_mutbl, ast::m_mutbl) {
            self.infcx().eq_tys(a.ty, b.ty).then {||
                ok({ty: a.ty, mutbl: ast::m_mutbl})
            }
          }

          // If one side or both is immutable, we can use the GLB of
          // both sides but mutbl must be `m_imm`.
          (ast::m_imm, ast::m_const) |
          (ast::m_const, ast::m_imm) |
          (ast::m_imm, ast::m_imm) {
            self.c_tys(a.ty, b.ty).chain {|t|
                ok({ty: t, mutbl: ast::m_imm})
            }
          }

          // If both sides are const, then we can use GLB of both
          // sides and mutbl of only `m_const`.
          (ast::m_const, ast::m_const) {
            self.c_tys(a.ty, b.ty).chain {|t|
                ok({ty: t, mutbl: ast::m_const})
            }
          }

          // There is no mutual subtype of these combinations.
          (ast::m_mutbl, ast::m_imm) |
          (ast::m_imm, ast::m_mutbl) {
              err(ty::terr_mutability)
          }
        }
    }

    fn c_contratys(a: ty::t, b: ty::t) -> cres<ty::t> {
        lub(self.infcx()).c_tys(a, b)
    }

    fn c_tys(a: ty::t, b: ty::t) -> cres<ty::t> {
        c_tys(self, a, b)
    }

    fn c_protos(p1: ast::proto, p2: ast::proto) -> cres<ast::proto> {
        if p1 == ast::proto_any {
            ok(p2)
        } else if p2 == ast::proto_any {
            ok(p1)
        } else if p1 == p2 {
            ok(p1)
        } else {
            ok(ast::proto_bare)
        }
    }

    fn c_ret_styles(r1: ret_style, r2: ret_style) -> cres<ret_style> {
        alt (r1, r2) {
          (ast::return_val, ast::return_val) {
            ok(ast::return_val)
          }
          (ast::noreturn, _) |
          (_, ast::noreturn) {
            ok(ast::noreturn)
          }
        }
    }
}
