#[doc = "

Conversion from AST representation of types to the ty.rs
representation.  The main routine here is `ast_ty_to_ty()`: each use
is parameterized by an instance of `ast_conv` and a `region_scope`.

The parameterization of `ast_ty_to_ty()` is because it behaves
somewhat differently during the collect and check phases, particularly
with respect to looking up the types of top-level items.  In the
collect phase, the crate context is used as the `ast_conv` instance;
in this phase, the `get_item_ty()` function triggers a recursive call
to `ty_of_item()` (note that `ast_ty_to_ty()` will detect recursive
types and report an error).  In the check phase, when the @fn_ctxt is
used as the `ast_conv`, `get_item_ty()` just looks up the item type in
`tcx.tcache`.

The `region_scope` interface controls how region references are
handled.  It has two methods which are used to resolve anonymous
region references (e.g., `&T`) and named region references (e.g.,
`&a.T`).  There are numerous region scopes that can be used, but most
commonly you want either `empty_rscope`, which permits only the static
region, or `type_rscope`, which permits the self region if the type in
question is parameterized by a region.

Unlike the `ast_conv` iface, the region scope can change as we descend
the type.  This is to accommodate the fact that (a) fn types are binding
scopes and (b) the default region may change.  To understand case (a),
consider something like:

  type foo = { x: &a.int, y: fn(&a.int) }

The type of `x` is an error because there is no region `a` in scope.
In the type of `y`, however, region `a` is considered a bound region
as it does not already appear in scope.

Case (b) says that if you have a type:
  type foo/& = ...;
  type bar = fn(&foo, &a.foo)
The fully expanded version of type bar is:
  type bar = fn(&foo/&, &a.foo/&a)
Note that the self region for the `foo` defaulted to `&` in the first
case but `&a` in the second.  Basically, defaults that appear inside
an rptr (`&r.T`) use the region `r` that appears in the rptr.

"];

import check::fn_ctxt;
import rscope::{anon_rscope, binding_rscope, empty_rscope, in_anon_rscope};
import rscope::{in_binding_rscope, region_scope, type_rscope};

iface ast_conv {
    fn tcx() -> ty::ctxt;
    fn ccx() -> @crate_ctxt;
    fn get_item_ty(id: ast::def_id) -> ty::ty_param_bounds_and_ty;

    // what type should we use when a type is omitted?
    fn ty_infer(span: span) -> ty::t;
}

fn get_region_reporting_err(tcx: ty::ctxt,
                            span: span,
                            res: result<ty::region, str>) -> ty::region {

    alt res {
      result::ok(r) { r }
      result::err(e) {
        tcx.sess.span_err(span, e);
        ty::re_static
      }
    }
}

fn ast_region_to_region<AC: ast_conv, RS: region_scope>(
    self: AC, rscope: RS, span: span, a_r: @ast::region) -> ty::region {

    let res = alt a_r.node {
      ast::re_anon { rscope.anon_region() }
      ast::re_named(id) { rscope.named_region(id) }
    };

    get_region_reporting_err(self.tcx(), span, res)
}

fn ast_path_to_substs_and_ty<AC: ast_conv, RS: region_scope copy>(
    self: AC, rscope: RS, did: ast::def_id,
    path: @ast::path) -> ty_param_substs_and_ty {

    let tcx = self.tcx();
    let {bounds: decl_bounds, rp: decl_rp, ty: decl_ty} =
        self.get_item_ty(did);

    // If the type is parameterized by the self region, then replace self
    // region with the current anon region binding (in other words,
    // whatever & would get replaced with).
    let self_r = alt (decl_rp, path.rp) {
      (ast::rp_none, none) {
        none
      }
      (ast::rp_none, some(_)) {
        tcx.sess.span_err(
            path.span,
            #fmt["No region bound is permitted on %s, \
                  which is not declared as containing region pointers",
                 ty::item_path_str(tcx, did)]);
        none
      }
      (ast::rp_self, none) {
        let res = rscope.anon_region();
        let r = get_region_reporting_err(self.tcx(), path.span, res);
        some(r)
      }
      (ast::rp_self, some(r)) {
        some(ast_region_to_region(self, rscope, path.span, r))
      }
    };

    // Convert the type parameters supplied by the user.
    if !vec::same_length(*decl_bounds, path.types) {
        self.tcx().sess.span_fatal(
            path.span,
            #fmt["wrong number of type arguments, expected %u but found %u",
                 (*decl_bounds).len(), path.types.len()]);
    }
    let tps = path.types.map { |a_t| ast_ty_to_ty(self, rscope, a_t) };

    let substs = {self_r:self_r, self_ty:none, tps:tps};
    {substs: substs, ty: ty::subst(tcx, substs, decl_ty)}
}

fn ast_path_to_ty<AC: ast_conv, RS: region_scope copy>(
    self: AC,
    rscope: RS,
    did: ast::def_id,
    path: @ast::path,
    path_id: ast::node_id) -> ty_param_substs_and_ty {

    // Lookup the polytype of the item and then substitute the provided types
    // for any type/region parameters.
    let tcx = self.tcx();
    let {substs: substs, ty: ty} =
        ast_path_to_substs_and_ty(self, rscope, did, path);
    write_ty_to_tcx(tcx, path_id, ty);
    write_substs_to_tcx(tcx, path_id, substs.tps);
    ret {substs: substs, ty: ty};
}

const NO_REGIONS: uint = 1u;
const NO_TPS: uint = 2u;

// Parses the programmer's textual representation of a type into our
// internal notion of a type. `getter` is a function that returns the type
// corresponding to a definition ID:
fn ast_ty_to_ty<AC: ast_conv, RS: region_scope copy>(
    self: AC, rscope: RS, &&ast_ty: @ast::ty) -> ty::t {

    fn ast_mt_to_mt<AC: ast_conv, RS: region_scope copy>(
        self: AC, rscope: RS, mt: ast::mt) -> ty::mt {

        ret {ty: ast_ty_to_ty(self, rscope, mt.ty), mutbl: mt.mutbl};
    }

    fn mk_vstore<AC: ast_conv, RS: region_scope copy>(
        self: AC, rscope: RS, a_seq_ty: @ast::ty, vst: ty::vstore) -> ty::t {

        let tcx = self.tcx();
        let seq_ty = ast_ty_to_ty(self, rscope, a_seq_ty);

        alt ty::get(seq_ty).struct {
          ty::ty_vec(mt) {
            ret ty::mk_evec(tcx, mt, vst);
          }

          ty::ty_str {
            ret ty::mk_estr(tcx, vst);
          }

          _ {
            tcx.sess.span_err(
                a_seq_ty.span,
                #fmt["Bound not allowed on a %s.",
                     ty::ty_sort_str(tcx, seq_ty)]);
            ret seq_ty;
          }
        }
    }

    fn check_path_args(tcx: ty::ctxt,
                       path: @ast::path,
                       flags: uint) {
        if (flags & NO_TPS) != 0u {
            if path.types.len() > 0u {
                tcx.sess.span_err(
                    path.span,
                    "Type parameters are not allowed on this type.");
            }
        }

        if (flags & NO_REGIONS) != 0u {
            if path.rp.is_some() {
                tcx.sess.span_err(
                    path.span,
                    "Region parameters are not allowed on this type.");
            }
        }
    }

    let tcx = self.tcx();

    alt tcx.ast_ty_to_ty_cache.find(ast_ty) {
      some(ty::atttce_resolved(ty)) { ret ty; }
      some(ty::atttce_unresolved) {
        tcx.sess.span_fatal(ast_ty.span, "illegal recursive type. \
                                          insert a enum in the cycle, \
                                          if this is desired)");
      }
      none { /* go on */ }
    }

    tcx.ast_ty_to_ty_cache.insert(ast_ty, ty::atttce_unresolved);
    let typ = alt ast_ty.node {
      ast::ty_nil { ty::mk_nil(tcx) }
      ast::ty_bot { ty::mk_bot(tcx) }
      ast::ty_box(mt) {
        ty::mk_box(tcx, ast_mt_to_mt(self, rscope, mt))
      }
      ast::ty_uniq(mt) {
        ty::mk_uniq(tcx, ast_mt_to_mt(self, rscope, mt))
      }
      ast::ty_vec(mt) {
        ty::mk_vec(tcx, ast_mt_to_mt(self, rscope, mt))
      }
      ast::ty_ptr(mt) {
        ty::mk_ptr(tcx, ast_mt_to_mt(self, rscope, mt))
      }
      ast::ty_rptr(region, mt) {
        let r = ast_region_to_region(self, rscope, ast_ty.span, region);
        let mt = ast_mt_to_mt(self, in_anon_rscope(rscope, r), mt);
        ty::mk_rptr(tcx, r, mt)
      }
      ast::ty_tup(fields) {
        let flds = vec::map(fields) { |t| ast_ty_to_ty(self, rscope, t) };
        ty::mk_tup(tcx, flds)
      }
      ast::ty_rec(fields) {
        let flds = fields.map {|f|
            let tm = ast_mt_to_mt(self, rscope, f.node.mt);
            {ident: f.node.ident, mt: tm}
        };
        ty::mk_rec(tcx, flds)
      }
      ast::ty_fn(proto, decl) {
        ty::mk_fn(tcx, ty_of_fn_decl(self, rscope, proto, decl, none))
      }
      ast::ty_path(path, id) {
        let a_def = alt tcx.def_map.find(id) {
          none { tcx.sess.span_fatal(ast_ty.span, #fmt("unbound path %s",
                                                       path_to_str(path))); }
          some(d) { d }};
        alt a_def {
          ast::def_ty(did) | ast::def_class(did) {
            ast_path_to_ty(self, rscope, did, path, id).ty
          }
          ast::def_prim_ty(nty) {
            alt nty {
              ast::ty_bool {
                check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                ty::mk_bool(tcx)
              }
              ast::ty_int(it) {
                check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                ty::mk_mach_int(tcx, it)
              }
              ast::ty_uint(uit) {
                check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                ty::mk_mach_uint(tcx, uit)
              }
              ast::ty_float(ft) {
                check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                ty::mk_mach_float(tcx, ft)
              }
              ast::ty_str {
                check_path_args(tcx, path, NO_TPS);
                // This is a bit of a hack, but basically str/& needs to be
                // converted into a vstore:
                alt path.rp {
                  none {
                    ty::mk_str(tcx)
                  }
                  some(ast_r) {
                    let r = ast_region_to_region(self, rscope,
                                                 ast_ty.span, ast_r);
                    ty::mk_estr(tcx, ty::vstore_slice(r))
                  }
                }
              }
            }
          }
          ast::def_ty_param(id, n) {
            check_path_args(tcx, path, NO_TPS | NO_REGIONS);
            ty::mk_param(tcx, n, id)
          }
          ast::def_self(_) {
            // n.b.: resolve guarantees that the self type only appears in an
            // iface, which we rely upon in various places when creating
            // substs
            check_path_args(tcx, path, NO_TPS | NO_REGIONS);
            ty::mk_self(tcx)
          }
          _ {
            tcx.sess.span_fatal(ast_ty.span,
                                "found type name used as a variable");
          }
        }
      }
      ast::ty_vstore(a_t, ast::vstore_slice(a_r)) {
        let r = ast_region_to_region(self, rscope, ast_ty.span, a_r);
        mk_vstore(self, in_anon_rscope(rscope, r), a_t, ty::vstore_slice(r))
      }
      ast::ty_vstore(a_t, ast::vstore_uniq) {
        mk_vstore(self, rscope, a_t, ty::vstore_uniq)
      }
      ast::ty_vstore(a_t, ast::vstore_box) {
        mk_vstore(self, rscope, a_t, ty::vstore_box)
      }
      ast::ty_vstore(a_t, ast::vstore_fixed(some(u))) {
        mk_vstore(self, rscope, a_t, ty::vstore_fixed(u))
      }
      ast::ty_vstore(_, ast::vstore_fixed(none)) {
        tcx.sess.span_bug(
            ast_ty.span,
            "implied fixed length for bound");
      }
      ast::ty_constr(t, cs) {
        let mut out_cs = [];
        for cs.each {|constr|
            out_cs += [ty::ast_constr_to_constr(tcx, constr)];
        }
        ty::mk_constr(tcx, ast_ty_to_ty(self, rscope, t), out_cs)
      }
      ast::ty_infer {
        // ty_infer should only appear as the type of arguments or return
        // values in a fn_expr, or as the type of local variables.  Both of
        // these cases are handled specially and should not descend into this
        // routine.
        self.tcx().sess.span_bug(
            ast_ty.span,
            "found `ty_infer` in unexpected place");
      }
      ast::ty_mac(_) {
        tcx.sess.span_bug(ast_ty.span,
                          "found `ty_mac` in unexpected place");
      }
    };

    tcx.ast_ty_to_ty_cache.insert(ast_ty, ty::atttce_resolved(typ));
    ret typ;
}

fn ty_of_arg<AC: ast_conv, RS: region_scope copy>(
    self: AC, rscope: RS, a: ast::arg,
    expected_ty: option<ty::arg>) -> ty::arg {

    let ty = alt a.ty.node {
      ast::ty_infer if expected_ty.is_some() {expected_ty.get().ty}
      ast::ty_infer {self.ty_infer(a.ty.span)}
      _ {ast_ty_to_ty(self, rscope, a.ty)}
    };

    let mode = {
        alt a.mode {
          ast::infer(_) if expected_ty.is_some() {
            result::get(ty::unify_mode(self.tcx(), a.mode,
                                       expected_ty.get().mode))
          }
          ast::infer(_) {
            alt ty::get(ty).struct {
              // If the type is not specified, then this must be a fn expr.
              // Leave the mode as infer(_), it will get inferred based
              // on constraints elsewhere.
              ty::ty_var(_) {a.mode}

              // If the type is known, then use the default for that type.
              // Here we unify m and the default.  This should update the
              // tables in tcx but should never fail, because nothing else
              // will have been unified with m yet:
              _ {
                let m1 = ast::expl(ty::default_arg_mode_for_ty(ty));
                result::get(ty::unify_mode(self.tcx(), a.mode, m1))
              }
            }
          }
          ast::expl(_) {a.mode}
        }
    };

    {mode: mode, ty: ty}
}

type expected_tys = option<{inputs: [ty::arg],
                            output: ty::t}>;

fn ty_of_fn_decl<AC: ast_conv, RS: region_scope copy>(
    self: AC, rscope: RS,
    proto: ast::proto,
    decl: ast::fn_decl,
    expected_tys: expected_tys) -> ty::fn_ty {

    #debug["ty_of_fn_decl"];
    indent {||
        // new region names that appear inside of the fn decl are bound to
        // that function type
        let rb = in_binding_rscope(rscope);

        let input_tys = decl.inputs.mapi { |i, a|
            let expected_arg_ty = expected_tys.chain { |e|
                // no guarantee that the correct number of expected args
                // were supplied
                if i < e.inputs.len() {some(e.inputs[i])} else {none}
            };
            ty_of_arg(self, rb, a, expected_arg_ty)
        };

        let expected_ret_ty = expected_tys.map { |e| e.output };
        let output_ty = alt decl.output.node {
          ast::ty_infer if expected_ret_ty.is_some() {expected_ret_ty.get()}
          ast::ty_infer {self.ty_infer(decl.output.span)}
          _ {ast_ty_to_ty(self, rb, decl.output)}
        };

        let out_constrs = vec::map(decl.constraints) {|constr|
            ty::ast_constr_to_constr(self.tcx(), constr)
        };

        {purity: decl.purity, proto: proto, inputs: input_tys,
         output: output_ty, ret_style: decl.cf, constraints: out_constrs}
    }
}


