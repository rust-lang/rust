#[doc = "

Conversion from AST representation of types to the ty.rs representation.

The main routine here is `ast_ty_to_ty()`: each use is parameterized
by an instance of `ast_conv` and a `region_scope`.

The `ast_conv` interface is the conversion context.  It has two
implementations, one for the crate context and one for the function
context.  The main purpose is to provide the `get_item_ty()` hook
which looks up the type of an item by its def-id.  This can be done in
two ways: in the initial phase, when a crate context is provided, this
will potentially trigger a call to `ty_of_item()`.  Later, when a
function context is used, this will simply be a lookup.

The `region_scope` interface controls how region references are
handled.  It has two methods which are used to resolve anonymous
region references (e.g., `&T`) and named region references (e.g.,
`&a.T`).  There are numerous region scopes that can be used, but most
commonly you want either `empty_rscope`, which permits only the static
region, or `type_rscope`, which permits the self region if the type in
question is parameterized by a region.

"];

iface ast_conv {
    fn tcx() -> ty::ctxt;
    fn ccx() -> @crate_ctxt;
    fn get_item_ty(id: ast::def_id) -> ty::ty_param_bounds_and_ty;

    // what type should we use when a type is omitted?
    fn ty_infer(span: span) -> ty::t;
}

impl of ast_conv for @crate_ctxt {
    fn tcx() -> ty::ctxt { self.tcx }
    fn ccx() -> @crate_ctxt { self }

    fn get_item_ty(id: ast::def_id) -> ty::ty_param_bounds_and_ty {
        if id.crate != ast::local_crate {
            csearch::get_type(self.tcx, id)
        } else {
            alt self.tcx.items.find(id.node) {
              some(ast_map::node_item(item, _)) {
                ty_of_item(self, item)
              }
              some(ast_map::node_native_item(native_item, _, _)) {
                ty_of_native_item(self, native_item)
              }
              x {
                self.tcx.sess.bug(#fmt["unexpected sort of item \
                                        in get_item_ty(): %?", x]);
              }
            }
        }
    }

    fn ty_infer(span: span) -> ty::t {
        self.tcx.sess.span_bug(span,
                               "found `ty_infer` in unexpected place");
    }
}

impl of ast_conv for @fn_ctxt {
    fn tcx() -> ty::ctxt { self.ccx.tcx }
    fn ccx() -> @crate_ctxt { self.ccx }

    fn get_item_ty(id: ast::def_id) -> ty::ty_param_bounds_and_ty {
        ty::lookup_item_type(self.tcx(), id)
    }

    fn ty_infer(_span: span) -> ty::t {
        self.next_ty_var()
    }
}

iface region_scope {
    fn anon_region() -> result<ty::region, str>;
    fn named_region(id: str) -> result<ty::region, str>;
}

enum empty_rscope { empty_rscope }
impl of region_scope for empty_rscope {
    fn anon_region() -> result<ty::region, str> {
        result::err("region types are not allowed here")
    }
    fn named_region(id: str) -> result<ty::region, str> {
        if id == "static" { result::ok(ty::re_static) }
        else { result::err("only the static region is allowed here") }
    }
}

enum type_rscope = ast::region_param;
impl of region_scope for type_rscope {
    fn anon_region() -> result<ty::region, str> {
        alt *self {
          ast::rp_self { result::ok(ty::re_bound(ty::br_self)) }
          ast::rp_none {
            result::err("to use region types here, the containing type \
                         must be declared with a region bound")
          }
        }
    }
    fn named_region(id: str) -> result<ty::region, str> {
        empty_rscope.named_region(id).chain_err { |_e|
            if id == "self" { self.anon_region() }
            else {
                result::err("named regions other than `self` are not \
                             allowed as part of a type declaration")
            }
        }
    }
}

impl of region_scope for @fn_ctxt {
    fn anon_region() -> result<ty::region, str> {
        result::ok(self.next_region_var())
    }
    fn named_region(id: str) -> result<ty::region, str> {
        empty_rscope.named_region(id).chain_err { |_e|
            alt self.in_scope_regions.find(ty::br_named(id)) {
              some(r) { result::ok(r) }
              none if id == "blk" { self.block_region() }
              none {
                result::err(#fmt["named region `%s` not in scope here", id])
              }
            }
        }
    }
}

enum anon_rscope = {anon: ty::region, base: region_scope};
fn in_anon_rscope<RS: region_scope copy>(self: RS, r: ty::region)
    -> @anon_rscope {
    @anon_rscope({anon: r, base: self as region_scope})
}
impl of region_scope for @anon_rscope {
    fn anon_region() -> result<ty::region, str> {
        result::ok(self.anon)
    }
    fn named_region(id: str) -> result<ty::region, str> {
        self.base.named_region(id)
    }
}

enum binding_rscope = {base: region_scope};
fn in_binding_rscope<RS: region_scope copy>(self: RS) -> @binding_rscope {
    let base = self as region_scope;
    @binding_rscope({base: base})
}
impl of region_scope for @binding_rscope {
    fn anon_region() -> result<ty::region, str> {
        result::ok(ty::re_bound(ty::br_anon))
    }
    fn named_region(id: str) -> result<ty::region, str> {
        self.base.named_region(id).chain_err {|_e|
            result::ok(ty::re_bound(ty::br_named(id)))
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

/*
  Instantiates the path for the given iface reference, assuming that
  it's bound to a valid iface type. Returns the def_id for the defining
  iface. Fails if the type is a type other than an iface type.
 */
fn instantiate_iface_ref(ccx: @crate_ctxt, t: @ast::iface_ref,
                         rp: ast::region_param)
    -> (ast::def_id, ty_param_substs_and_ty) {

    let sp = t.path.span, err = "can only implement interface types",
        sess = ccx.tcx.sess;

    let rscope = type_rscope(rp);

    alt lookup_def_tcx(ccx.tcx, t.path.span, t.id) {
      ast::def_ty(t_id) {
        let tpt = ast_path_to_ty(ccx, rscope, t_id, t.path, t.id);
        alt ty::get(tpt.ty).struct {
           ty::ty_iface(*) {
              (t_id, tpt)
           }
           _ { sess.span_fatal(sp, err); }
        }
      }
      _ {
          sess.span_fatal(sp, err);
      }
    }
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

fn ty_of_item(ccx: @crate_ctxt, it: @ast::item)
    -> ty::ty_param_bounds_and_ty {

    let def_id = local_def(it.id);
    let tcx = ccx.tcx;
    alt tcx.tcache.find(def_id) {
      some(tpt) { ret tpt; }
      _ {}
    }
    alt it.node {
      ast::item_const(t, _) {
        let typ = ccx.to_ty(empty_rscope, t);
        let tpt = no_params(typ);
        tcx.tcache.insert(local_def(it.id), tpt);
        ret tpt;
      }
      ast::item_fn(decl, tps, _) {
        let bounds = ty_param_bounds(ccx, tps);
        let tofd = ty_of_fn_decl(ccx, empty_rscope, ast::proto_bare,
                                 decl, none);
        let tpt = {bounds: bounds,
                   rp: ast::rp_none, // functions do not have a self
                   ty: ty::mk_fn(ccx.tcx, tofd)};
        ccx.tcx.tcache.insert(local_def(it.id), tpt);
        ret tpt;
      }
      ast::item_ty(t, tps, rp) {
        alt tcx.tcache.find(local_def(it.id)) {
          some(tpt) { ret tpt; }
          none { }
        }

        let tpt = {
            let ty = {
                let t0 = ccx.to_ty(type_rscope(rp), t);
                // Do not associate a def id with a named, parameterized type
                // like "foo<X>".  This is because otherwise ty_to_str will
                // print the name as merely "foo", as it has no way to
                // reconstruct the value of X.
                if !vec::is_empty(tps) { t0 } else {
                    ty::mk_with_id(tcx, t0, def_id)
                }
            };
            {bounds: ty_param_bounds(ccx, tps), rp: rp, ty: ty}
        };

        check_bounds_are_used(ccx, t.span, tps, rp, tpt.ty);

        tcx.tcache.insert(local_def(it.id), tpt);
        ret tpt;
      }
      ast::item_res(decl, tps, _, _, _, rp) {
        let {bounds, substs} = mk_substs(ccx, tps, rp);
        let t_arg = ty_of_arg(ccx, type_rscope(rp),
                              decl.inputs[0], none);
        let t = ty::mk_res(tcx, local_def(it.id), t_arg.ty, substs);
        let t_res = {bounds: bounds, rp: rp, ty: t};
        tcx.tcache.insert(local_def(it.id), t_res);
        ret t_res;
      }
      ast::item_enum(_, tps, rp) {
        // Create a new generic polytype.
        let {bounds, substs} = mk_substs(ccx, tps, rp);
        let t = ty::mk_enum(tcx, local_def(it.id), substs);
        let tpt = {bounds: bounds, rp: rp, ty: t};
        tcx.tcache.insert(local_def(it.id), tpt);
        ret tpt;
      }
      ast::item_iface(tps, rp, ms) {
        let {bounds, substs} = mk_substs(ccx, tps, rp);
        let t = ty::mk_iface(tcx, local_def(it.id), substs);
        let tpt = {bounds: bounds, rp: rp, ty: t};
        tcx.tcache.insert(local_def(it.id), tpt);
        ret tpt;
      }
      ast::item_class(tps, _, _, _, _, rp) {
          let {bounds,substs} = mk_substs(ccx, tps, rp);
          let t = ty::mk_class(tcx, local_def(it.id), substs);
          let tpt = {bounds: bounds, rp: rp, ty: t};
          tcx.tcache.insert(local_def(it.id), tpt);
          ret tpt;
      }
      ast::item_impl(*) | ast::item_mod(_) |
      ast::item_native_mod(_) { fail; }
    }
}

fn ty_of_native_item(ccx: @crate_ctxt, it: @ast::native_item)
    -> ty::ty_param_bounds_and_ty {
    alt it.node {
      ast::native_item_fn(fn_decl, params) {
        ret ty_of_native_fn_decl(ccx, fn_decl, params,
                                 local_def(it.id));
      }
    }
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
        {proto: proto, inputs: input_tys,
         output: output_ty, ret_style: decl.cf, constraints: out_constrs}
    }
}


fn ty_param_bounds(ccx: @crate_ctxt,
                   params: [ast::ty_param]) -> @[ty::param_bounds] {

    fn compute_bounds(ccx: @crate_ctxt,
                      param: ast::ty_param) -> ty::param_bounds {
        @vec::flat_map(*param.bounds) { |b|
            alt b {
              ast::bound_send { [ty::bound_send] }
              ast::bound_copy { [ty::bound_copy] }
              ast::bound_iface(t) {
                let ity = ast_ty_to_ty(ccx, empty_rscope, t);
                alt ty::get(ity).struct {
                  ty::ty_iface(*) {
                    [ty::bound_iface(ity)]
                  }
                  _ {
                    ccx.tcx.sess.span_err(
                        t.span, "type parameter bounds must be \
                                 interface types");
                    []
                  }
                }
              }
            }
        }
    }

    @params.map { |param|
        alt ccx.tcx.ty_param_bounds.find(param.id) {
          some(bs) { bs }
          none {
            let bounds = compute_bounds(ccx, param);
            ccx.tcx.ty_param_bounds.insert(param.id, bounds);
            bounds
          }
        }
    }
}

fn ty_of_native_fn_decl(ccx: @crate_ctxt,
                        decl: ast::fn_decl,
                        ty_params: [ast::ty_param],
                        def_id: ast::def_id) -> ty::ty_param_bounds_and_ty {

    let bounds = ty_param_bounds(ccx, ty_params);
    let rb = in_binding_rscope(empty_rscope);
    let input_tys = decl.inputs.map { |a| ty_of_arg(ccx, rb, a, none) };
    let output_ty = ast_ty_to_ty(ccx, rb, decl.output);

    let t_fn = ty::mk_fn(ccx.tcx, {proto: ast::proto_bare,
                                   inputs: input_tys,
                                   output: output_ty,
                                   ret_style: ast::return_val,
                                   constraints: []});
    let tpt = {bounds: bounds, rp: ast::rp_none, ty: t_fn};
    ccx.tcx.tcache.insert(def_id, tpt);
    ret tpt;
}
