import result::{result, extensions};
import syntax::{ast, ast_util};
import ast::spanned;
import syntax::ast_util::{local_def, respan, split_class_items};
import syntax::visit;
import metadata::csearch;
import driver::session::session;
import util::common::*;
import syntax::codemap::span;
import pat_util::*;
import middle::ty;
import middle::ty::{arg, field, node_type_table, mk_nil,
                    ty_param_bounds_and_ty, lookup_public_fields};
import middle::ty::{ty_vid, region_vid, vid};
import util::ppaux::{ty_to_str, region_to_str,
                     bound_region_to_str, vstore_to_str};
import std::smallintmap;
import std::smallintmap::map;
import std::map;
import std::map::{hashmap, int_hash};
import std::serialization::{serialize_uint, deserialize_uint};
import std::ufind;
import syntax::print::pprust::*;
import util::common::indent;
import std::list;
import list::{list, nil, cons};

export check_crate;
export method_map;
export method_origin, serialize_method_origin, deserialize_method_origin;
export vtable_map;
export vtable_res;
export vtable_origin;

#[auto_serialize]
enum method_origin {
    method_static(ast::def_id),
    // iface id, method num, param num, bound num
    method_param(ast::def_id, uint, uint, uint),
    method_iface(ast::def_id, uint),
}
type method_map = hashmap<ast::node_id, method_origin>;

// Resolutions for bounds of all parameters, left to right, for a given path.
type vtable_res = @[vtable_origin];
enum vtable_origin {
    vtable_static(ast::def_id, [ty::t], vtable_res),
    // Param number, bound number
    vtable_param(uint, uint),
    vtable_iface(ast::def_id, [ty::t]),
}

type vtable_map = hashmap<ast::node_id, vtable_res>;

type ty_table = hashmap<ast::def_id, ty::t>;

type crate_ctxt = {impl_map: resolve::impl_map,
                   method_map: method_map,
                   vtable_map: vtable_map,
                   // Not at all sure it's right to put these here
                   /* node_id for the class this fn is in --
                      none if it's not in a class */
                   enclosing_class_id: option<ast::node_id>,
                   /* map from node_ids for enclosing-class
                      vars and methods to types */
                   enclosing_class: class_map,
                   tcx: ty::ctxt};

type class_map = hashmap<ast::node_id, ty::t>;

// a list of mapping from in-scope-region-names ("isr") to the
// corresponding ty::region
type isr_alist = @list<(ty::bound_region, ty::region)>;

type fn_ctxt =
    // var_bindings, locals and next_var_id are shared
    // with any nested functions that capture the environment
    // (and with any functions whose environment is being captured).
    {self_ty: option<ty::t>,
     ret_ty: ty::t,
     // Used by loop bodies that return from the outer function
     indirect_ret_ty: option<ty::t>,
     purity: ast::purity,
     proto: ast::proto,
     infcx: infer::infer_ctxt,
     locals: hashmap<ast::node_id, ty_vid>,
     ty_var_counter: @mut uint,
     region_var_counter: @mut uint,

     in_scope_regions: isr_alist,

     // While type checking a function, the intermediate types for the
     // expressions, blocks, and so forth contained within the function are
     // stored in these tables.  These types may contain unresolved type
     // variables.  After type checking is complete, the functions in the
     // writeback module are used to take the types from this table, resolve
     // them, and then write them into their permanent home in the type
     // context `ccx.tcx`.
     //
     // This means that during inferencing you should use `fcx.write_ty()`
     // and `fcx.expr_ty()` / `fcx.node_ty()` to write/obtain the types of
     // nodes within the function.
     //
     // The types of top-level items, which never contain unbound type
     // variables, are stored directly into the `tcx` tables.
     //
     // n.b.: A type variable is not the same thing as a type parameter.  A
     // type variable is rather an "instance" of a type parameter: that is,
     // given a generic function `fn foo<T>(t: T)`: while checking the
     // function `foo`, the type `ty_param(0)` refers to the type `T`, which
     // is treated in abstract.  When `foo()` is called, however, `T` will be
     // substituted for a fresh type variable `ty_var(N)`.  This variable will
     // eventually be resolved to some concrete type (which might itself be
     // type parameter).
     node_types: smallintmap::smallintmap<ty::t>,
     node_type_substs: hashmap<ast::node_id, ty::substs>,

     ccx: @crate_ctxt};

// Determines whether the given node ID is a use of the def of
// the self ID for the current method, if there is one
fn self_ref(fcx: @fn_ctxt, id: ast::node_id) -> bool {
    // check what def `id` was resolved to (if anything)
    alt fcx.ccx.tcx.def_map.find(id) {
      some(ast::def_self(_)) { true }
      _ { false }
    }
}

fn lookup_local(fcx: @fn_ctxt, sp: span, id: ast::node_id) -> ty_vid {
    alt fcx.locals.find(id) {
      some(x) { x }
      _ {
        fcx.ccx.tcx.sess.span_fatal(sp,
                                    "internal error looking up a local var")
      }
    }
}

fn lookup_def_tcx(tcx: ty::ctxt, sp: span, id: ast::node_id) -> ast::def {
    alt tcx.def_map.find(id) {
      some(x) { x }
      _ {
        tcx.sess.span_fatal(sp, "internal error looking up a definition")
      }
    }
}

fn lookup_def_ccx(ccx: @crate_ctxt, sp: span, id: ast::node_id) -> ast::def {
    lookup_def_tcx(ccx.tcx, sp, id)
}

fn lookup_def(fcx: @fn_ctxt, sp: span, id: ast::node_id) -> ast::def {
    lookup_def_ccx(fcx.ccx, sp, id)
}

fn no_params(t: ty::t) -> ty::ty_param_bounds_and_ty {
    {bounds: @[], rp: ast::rp_none, ty: t}
}

// Returns the type parameter count and the type for the given definition.
fn ty_param_bounds_and_ty_for_def(fcx: @fn_ctxt, sp: span, defn: ast::def) ->
    ty_param_bounds_and_ty {

    alt defn {
      ast::def_arg(nid, _) {
        assert (fcx.locals.contains_key(nid));
        let typ = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, sp, nid));
        ret no_params(typ);
      }
      ast::def_local(nid, _) {
        assert (fcx.locals.contains_key(nid));
        let typ = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, sp, nid));
        ret no_params(typ);
      }
      ast::def_self(_) {
        alt fcx.self_ty {
          some(self_ty) {
            ret no_params(self_ty);
          }
          none {
              fcx.ccx.tcx.sess.span_bug(sp, "def_self with no self_ty");
          }
        }
      }
      ast::def_fn(id, ast::crust_fn) {
        // Crust functions are just u8 pointers
        ret {
            bounds: @[],
            rp: ast::rp_none,
            ty: ty::mk_ptr(
                fcx.ccx.tcx,
                {
                    ty: ty::mk_mach_uint(fcx.ccx.tcx, ast::ty_u8),
                    mutbl: ast::m_imm
                })
        };
      }
      ast::def_fn(id, _) | ast::def_const(id) |
      ast::def_variant(_, id) | ast::def_class(id)
         { ret ty::lookup_item_type(fcx.ccx.tcx, id); }
      ast::def_binding(nid) {
        assert (fcx.locals.contains_key(nid));
        let typ = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, sp, nid));
        ret no_params(typ);
      }
      ast::def_ty(_) | ast::def_prim_ty(_) {
        fcx.ccx.tcx.sess.span_fatal(sp, "expected value but found type");
      }
      ast::def_upvar(_, inner, _) {
        ret ty_param_bounds_and_ty_for_def(fcx, sp, *inner);
      }
      _ {
        // FIXME: handle other names.
        fcx.ccx.tcx.sess.unimpl("definition variant");
      }
    }
}

// Instantiates the given path, which must refer to an item with the given
// number of type parameters and type.
fn instantiate_path(fcx: @fn_ctxt,
                    pth: @ast::path,
                    tpt: ty_param_bounds_and_ty,
                    sp: span,
                    id: ast::node_id) {
    let ty_param_count = vec::len(*tpt.bounds);
    let ty_substs_len = vec::len(pth.types);

    // For now, there is no way to explicitly specify the region bound.
    // This will have to change eventually.
    let self_r = alt tpt.rp {
      ast::rp_self { some(fcx.next_region_var()) }
      ast::rp_none { none }
    };

    let tps = if ty_substs_len == 0u {
        fcx.next_ty_vars(ty_param_count)
    } else if ty_param_count == 0u {
        fcx.ccx.tcx.sess.span_err
            (sp, "this item does not take type parameters");
        fcx.next_ty_vars(ty_param_count)
    } else if ty_substs_len > ty_param_count {
        fcx.ccx.tcx.sess.span_err
            (sp, "too many type parameters provided for this item");
        fcx.next_ty_vars(ty_param_count)
    } else if ty_substs_len < ty_param_count {
        fcx.ccx.tcx.sess.span_err
            (sp, "not enough type parameters provided for this item");
        fcx.next_ty_vars(ty_param_count)
    } else {
        pth.types.map { |aty| fcx.to_ty(aty) }
    };

    let substs = {self_r: self_r, tps: tps};
    fcx.write_ty_substs(id, tpt.ty, substs);
}

// Type tests
fn structurally_resolved_type(fcx: @fn_ctxt, sp: span, tp: ty::t) -> ty::t {
    alt infer::resolve_shallow(fcx.infcx, tp, true) {
      // note: the bot type doesn't count as resolved; it's what we use when
      // there is no information about a variable.
      result::ok(t_s) if !ty::type_is_bot(t_s) { ret t_s; }
      _ {
        fcx.ccx.tcx.sess.span_fatal
            (sp, "the type of this value must be known in this context");
      }
    }
}


// Returns the one-level-deep structure of the given type.
fn structure_of(fcx: @fn_ctxt, sp: span, typ: ty::t) -> ty::sty {
    ty::get(structurally_resolved_type(fcx, sp, typ)).struct
}

fn type_is_integral(fcx: @fn_ctxt, sp: span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    ret ty::type_is_integral(typ_s);
}

fn type_is_scalar(fcx: @fn_ctxt, sp: span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    ret ty::type_is_scalar(typ_s);
}

fn type_is_c_like_enum(fcx: @fn_ctxt, sp: span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    ret ty::type_is_c_like_enum(fcx.ccx.tcx, typ_s);
}

fn ast_expr_vstore_to_vstore(fcx: @fn_ctxt, e: @ast::expr, n: uint,
                             v: ast::vstore) -> ty::vstore {
    alt v {
      ast::vstore_fixed(none) { ty::vstore_fixed(n) }
      ast::vstore_fixed(some(u)) {
        if n != u {
            let s = #fmt("fixed-size sequence mismatch: %u vs. %u",u, n);
            fcx.ccx.tcx.sess.span_err(e.span,s);
        }
        ty::vstore_fixed(u)
      }
      ast::vstore_uniq { ty::vstore_uniq }
      ast::vstore_box { ty::vstore_box }
      ast::vstore_slice(r) {
        ty::vstore_slice(ast_region_to_region(fcx, fcx, e.span, r))
      }
    }
}

iface ast_conv {
    fn tcx() -> ty::ctxt;
    fn ccx() -> @crate_ctxt;
    fn get_item_ty(id: ast::def_id) -> ty::ty_param_bounds_and_ty;
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
    fn named_region(_id: str) -> result<ty::region, str> {
        result::err("region types are not allowed here")
    }
}

enum type_rscope = ast::region_param;
impl of region_scope for type_rscope {
    fn anon_region() -> result<ty::region, str> {
        alt *self {
          ast::rp_self {
            result::ok(ty::re_bound(ty::br_self))
          }
          ast::rp_none {
            result::err("to use region types here, the containing type \
                         must be declared with a region bound")
          }
        }
    }
    fn named_region(id: str) -> result<ty::region, str> {
        if id == "self" {
            self.anon_region()
        } else {
            result::err("named regions other than `self` are not \
                         allowed as part of a type declaration")
        }
    }
}

impl of region_scope for @fn_ctxt {
    fn anon_region() -> result<ty::region, str> {
        result::ok(self.next_region_var())
    }
    fn named_region(id: str) -> result<ty::region, str> {
        alt self.in_scope_regions.find(ty::br_named(id)) {
          some(r) { result::ok(r) }
          none {
            result::err(#fmt["named region `%s` not in scope here", id])
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
    self: AC, rscope: RS, span: span, a_r: ast::region) -> ty::region {

    let res = alt a_r.node {
      ast::re_anon { rscope.anon_region() }
      ast::re_named(id) { rscope.named_region(id) }
      ast::re_static { result::ok(ty::re_static) }
    };

    get_region_reporting_err(self.tcx(), span, res)
}

// Parses the programmer's textual representation of a type into our
// internal notion of a type. `getter` is a function that returns the type
// corresponding to a definition ID:
fn ast_ty_to_ty<AC: ast_conv, RS: region_scope copy>(
    self: AC, rscope: RS, &&ast_ty: @ast::ty) -> ty::t {

    fn ast_mt_to_mt<AC: ast_conv, RS: region_scope copy>(
        self: AC, rscope: RS, mt: ast::mt) -> ty::mt {

        ret {ty: ast_ty_to_ty(self, rscope, mt.ty), mutbl: mt.mutbl};
    }

    fn instantiate<AC: ast_conv, RS: region_scope copy>(
        self: AC, rscope: RS, sp: span, id: ast::def_id,
        path_id: ast::node_id, args: [@ast::ty]) -> ty::t {

        let tcx = self.tcx();
        let {bounds, rp, ty} = self.get_item_ty(id);

        // If the type is parameterized by the self region, then replace self
        // region with the current anon region binding (in other words,
        // whatever & would get replaced with).
        let self_r = alt rp {
          ast::rp_none { none }
          ast::rp_self {
            let res = rscope.anon_region();
            let r = get_region_reporting_err(self.tcx(), sp, res);
            some(r)
          }
        };

        // Convert the type parameters supplied by the user.
        if vec::len(args) != vec::len(*bounds) {
            tcx.sess.span_fatal(
                sp, #fmt["wrong number of type arguments, \
                          expected %u but found %u",
                         vec::len(*bounds),
                         vec::len(args)]);
        }
        let tps = args.map { |t| ast_ty_to_ty(self, rscope, t) };

        // Perform the substitution and store the tps for future ref.
        let substs = {self_r: self_r, tps: tps};
        let ty = ty::subst(tcx, substs, ty);
        write_substs_to_tcx(tcx, path_id, substs.tps);
        ret ty;
    }

    fn mk_bounded<AC: ast_conv, RS: region_scope copy>(
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

          ty::ty_enum(_, subst) |
          ty::ty_class(_, subst) |
          ty::ty_res(_, _, subst) {
            // n.b.: This is a hacky abuse of the vstore terminology to also
            // make it work for region bounds.  The idea is to allow a type
            // Id/&r where Id is an enum, class, or resource, but not Id/@
            // etc.  We also do not want to allow Id/&r if the given
            // enum/class/resource does not define a region parameter.
            //
            // Really, these "/&r" bounds ought to be part of the path, like
            // type parameters.  (In fact, we could generalize to allowing
            // multiple such bounds someday)
            alt (subst.self_r, vst) {
              (some(_), ty::vstore_slice(_)) { /* ok */ }
              (none, ty::vstore_slice(_)) {
                tcx.sess.span_err(
                    a_seq_ty.span,
                    #fmt["inappropriate bound for %s, \
                          which is not declared as containing \
                          region pointers",
                         ty::ty_sort_str(tcx, seq_ty)]);
              }
              (_, _) {
                tcx.sess.span_err(
                    a_seq_ty.span,
                    #fmt["a %s bound is not appropriate for %s",
                         vstore_to_str(tcx, vst),
                         ty::ty_sort_str(tcx, seq_ty)]);
              }
            }
            ret seq_ty;
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
        ty::mk_fn(tcx, ty_of_fn_decl(self, rscope, proto, decl))
      }
      ast::ty_path(path, id) {
        let a_def = alt tcx.def_map.find(id) {
          none { tcx.sess.span_fatal(ast_ty.span, #fmt("unbound path %s",
                                                       path_to_str(path))); }
          some(d) { d }};
        alt a_def {
          ast::def_ty(did) | ast::def_class(did) {
            instantiate(self, rscope, ast_ty.span, did, id, path.types)
          }
          ast::def_prim_ty(nty) {
            alt nty {
              ast::ty_bool { ty::mk_bool(tcx) }
              ast::ty_int(it) { ty::mk_mach_int(tcx, it) }
              ast::ty_uint(uit) { ty::mk_mach_uint(tcx, uit) }
              ast::ty_float(ft) { ty::mk_mach_float(tcx, ft) }
              ast::ty_str { ty::mk_str(tcx) }
            }
          }
          ast::def_ty_param(id, n) {
            if vec::len(path.types) > 0u {
                tcx.sess.span_err(ast_ty.span, "provided type parameters \
                                                to a type parameter");
            }
            ty::mk_param(tcx, n, id)
          }
          ast::def_self(self_id) {
            alt check tcx.items.get(self_id) {
              ast_map::node_item(@{node: ast::item_iface(tps, _), _}, _) {
                if vec::len(tps) != vec::len(path.types) {
                    tcx.sess.span_err(ast_ty.span, "incorrect number of \
                                                    type parameters to \
                                                    self type");
                }
                ty::mk_self(tcx, vec::map(path.types, {|ast_ty|
                    ast_ty_to_ty(self, rscope, ast_ty)
                }))
              }
            }
          }
          _ {
            tcx.sess.span_fatal(ast_ty.span,
                                "found type name used as a variable");
          }
        }
      }
      ast::ty_vstore(a_t, ast::vstore_slice(a_r)) {
        let r = ast_region_to_region(self, rscope, ast_ty.span, a_r);
        mk_bounded(self, in_anon_rscope(rscope, r), a_t, ty::vstore_slice(r))
      }
      ast::ty_vstore(a_t, ast::vstore_uniq) {
        mk_bounded(self, rscope, a_t, ty::vstore_uniq)
      }
      ast::ty_vstore(a_t, ast::vstore_box) {
        mk_bounded(self, rscope, a_t, ty::vstore_box)
      }
      ast::ty_vstore(a_t, ast::vstore_fixed(some(u))) {
        mk_bounded(self, rscope, a_t, ty::vstore_fixed(u))
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
        self.ty_infer(ast_ty.span)
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
        let tofd = ty_of_fn_decl(ccx, empty_rscope, ast::proto_bare, decl);
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
        tcx.tcache.insert(local_def(it.id), tpt);
        ret tpt;
      }
      ast::item_res(decl, tps, _, _, _, rp) {
        let {bounds, substs} = mk_substs(ccx, tps, rp);
        let t_arg = ty_of_arg(ccx, empty_rscope, decl.inputs[0]);
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
      ast::item_iface(tps, ms) {
        let {bounds, params} = mk_ty_params(ccx, tps);
        let t = ty::mk_iface(tcx, local_def(it.id), params);
        // NDM iface/impl regions
        let tpt = {bounds: bounds, rp: ast::rp_none, ty: t};
        tcx.tcache.insert(local_def(it.id), tpt);
        ret tpt;
      }
      ast::item_class(tps, _, _, _, rp) {
          let {bounds,substs} = mk_substs(ccx, tps, rp);
          let t = ty::mk_class(tcx, local_def(it.id), substs);
          let tpt = {bounds: bounds, rp: rp, ty: t};
          tcx.tcache.insert(local_def(it.id), tpt);
          ret tpt;
      }
      ast::item_impl(_, _, _, _) | ast::item_mod(_) |
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

type next_region_param_id = { mut id: uint };

fn collect_bound_regions_in_tys(
    tcx: ty::ctxt,
    isr: isr_alist,
    tys: [ty::t],
    to_r: fn(ty::bound_region) -> ty::region) -> isr_alist {

    tys.foldl(isr) { |isr, t|
        collect_bound_regions_in_ty(tcx, isr, t, to_r)
    }
}

fn collect_bound_regions_in_ty(
    tcx: ty::ctxt,
    isr: isr_alist,
    ty: ty::t,
    to_r: fn(ty::bound_region) -> ty::region) -> isr_alist {

    fn append_isr(isr: isr_alist,
                  to_r: fn(ty::bound_region) -> ty::region,
                  r: ty::region) -> isr_alist {
        alt r {
          ty::re_free(_, _) | ty::re_static | ty::re_scope(_) |
          ty::re_var(_) {
            isr
          }
          ty::re_bound(br) {
            alt isr.find(br) {
              some(_) { isr }
              none { @cons((br, to_r(br)), isr) }
            }
          }
        }
    }

    let mut isr = isr;

    // Using fold_regions is inefficient, because it constructs new types, but
    // it avoids code duplication in terms of locating all the regions within
    // the various kinds of types.  This had already caused me several bugs
    // so I decided to switch over.
    ty::fold_regions(tcx, ty) { |r, in_fn|
        if !in_fn { isr = append_isr(isr, to_r, r); }
        r
    };

    ret isr;
}

fn replace_bound_self(
    tcx: ty::ctxt,
    self_r: ty::region,
    ty: ty::t) -> ty::t {

    ty::fold_regions(tcx, ty) { |r, _in_fn|
        if r == ty::re_bound(ty::br_self) {self_r} else {r}
    }
}

fn replace_bound_regions(
    tcx: ty::ctxt,
    span: span,
    isr: isr_alist,
    ty: ty::t) -> ty::t {

    ty::fold_regions(tcx, ty) { |r, in_fn|
        alt r {
          // As long as we are not within a fn() type, `&T` is mapped to the
          // free region anon_r.  But within a fn type, it remains bound.
          ty::re_bound(ty::br_anon) if in_fn { r }

          ty::re_bound(br) {
            alt isr.find(br) {
              // In most cases, all named, bound regions will be mapped to
              // some free region.
              some(fr) { fr }

              // But in the case of a fn() type, there may be named regions
              // within that remain bound:
              none if in_fn { r }
              none {
                tcx.sess.span_bug(
                    span,
                    #fmt["Bound region not found in \
                          in_scope_regions list: %s",
                         region_to_str(tcx, r)]);
              }
            }
          }

          // Free regions like these just stay the same:
          ty::re_static |
          ty::re_scope(_) |
          ty::re_free(_, _) |
          ty::re_var(_) { r }
        }
    }
}

fn ty_of_arg<AC: ast_conv, RS: region_scope copy>(
    self: AC, rscope: RS, a: ast::arg) -> ty::arg {

    fn arg_mode(tcx: ty::ctxt, m: ast::mode, ty: ty::t) -> ast::mode {
        alt m {
          ast::infer(_) {
            alt ty::get(ty).struct {
              // If the type is not specified, then this must be a fn expr.
              // Leave the mode as infer(_), it will get inferred based
              // on constraints elsewhere.
              ty::ty_var(_) { m }

              // If the type is known, then use the default for that type.
              // Here we unify m and the default.  This should update the
              // tables in tcx but should never fail, because nothing else
              // will have been unified with m yet:
              _ {
                let m1 = ast::expl(ty::default_arg_mode_for_ty(ty));
                result::get(ty::unify_mode(tcx, m, m1))
              }
            }
          }
          ast::expl(_) { m }
        }
    }

    let ty = ast_ty_to_ty(self, rscope, a.ty);
    let mode = arg_mode(self.tcx(), a.mode, ty);
    {mode: mode, ty: ty}
}
fn ty_of_fn_decl<AC: ast_conv, RS: region_scope copy>(
    self: AC, rscope: RS,
    proto: ast::proto,
    decl: ast::fn_decl) -> ty::fn_ty {

    #debug["ty_of_fn_decl"];
    indent {||
        // new region names that appear inside of the fn decl are bound to
        // that function type
        let rb = in_binding_rscope(rscope);
        let input_tys = vec::map(decl.inputs) { |a| ty_of_arg(self, rb, a) };
        let output_ty = ast_ty_to_ty(self, rb, decl.output);
        let out_constrs = vec::map(decl.constraints) {|constr|
            ty::ast_constr_to_constr(self.tcx(), constr)
        };
        {proto: proto, inputs: input_tys,
         output: output_ty, ret_style: decl.cf, constraints: out_constrs}
    }
}

fn ty_of_native_fn_decl(ccx: @crate_ctxt,
                        decl: ast::fn_decl,
                        ty_params: [ast::ty_param],
                        def_id: ast::def_id) -> ty::ty_param_bounds_and_ty {

    let bounds = ty_param_bounds(ccx, ty_params);
    let rb = in_binding_rscope(empty_rscope);
    let input_tys = vec::map(decl.inputs) { |a| ty_of_arg(ccx, rb, a) };
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

fn ty_param_bounds(ccx: @crate_ctxt,
                   params: [ast::ty_param]) -> @[ty::param_bounds] {

    @params.map { |param|
        alt ccx.tcx.ty_param_bounds.find(param.id) {
          some(bs) { bs }
          none {
            let bounds = @vec::flat_map(*param.bounds) { |b|
                alt b {
                  ast::bound_send { [ty::bound_send] }
                  ast::bound_copy { [ty::bound_copy] }
                  ast::bound_iface(t) {
                    let ity = ccx.to_ty(empty_rscope, t);
                    alt ty::get(ity).struct {
                      ty::ty_iface(_, _) {
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
            };
            ccx.tcx.ty_param_bounds.insert(param.id, bounds);
            bounds
          }
        }
    }
}

fn ty_of_method(ccx: @crate_ctxt,
                m: @ast::method,
                rp: ast::region_param) -> ty::method {
    {ident: m.ident, tps: ty_param_bounds(ccx, m.tps),
     fty: ty_of_fn_decl(ccx, type_rscope(rp), ast::proto_bare, m.decl),
     purity: m.decl.purity, privacy: m.privacy}
}

fn ty_of_ty_method(self: @crate_ctxt, m: ast::ty_method) -> ty::method {
    {ident: m.ident,
     tps: ty_param_bounds(self, m.tps),
     fty: ty_of_fn_decl(self, empty_rscope, ast::proto_bare, m.decl),
     // assume public, because this is only invoked on iface methods
     purity: m.decl.purity, privacy: ast::pub}
}

// Functions that write types into the node type table
fn write_ty_to_tcx(tcx: ty::ctxt, node_id: ast::node_id, ty: ty::t) {
    #debug["write_ty_to_tcx(%d, %s)", node_id, ty_to_str(tcx, ty)];
    smallintmap::insert(*tcx.node_types, node_id as uint, ty);
}
fn write_substs_to_tcx(tcx: ty::ctxt,
                       node_id: ast::node_id,
                       +substs: [ty::t]) {
    if substs.len() > 0u {
        tcx.node_type_substs.insert(node_id, substs);
    }
}

impl methods for @crate_ctxt {
    fn to_ty<RS: region_scope copy>(rs: RS, ast_ty: @ast::ty) -> ty::t {
        ast_ty_to_ty(self, rs, ast_ty)
    }
}

impl methods for isr_alist {
    fn get(br: ty::bound_region) -> ty::region {
        option::get(self.find(br))
    }

    fn find(br: ty::bound_region) -> option<ty::region> {
        for list::each(*self) { |isr|
            let (isr_br, isr_r) = isr;
            if isr_br == br { ret some(isr_r); }
        }
        ret none;
    }
}

impl methods for @fn_ctxt {
    fn tag() -> str { #fmt["%x", ptr::addr_of(*self) as uint] }
    fn ty_to_str(t: ty::t) -> str {
        ty_to_str(self.ccx.tcx, resolve_type_vars_if_possible(self, t))
    }
    fn write_ty(node_id: ast::node_id, ty: ty::t) {
        #debug["write_ty(%d, %s) in fcx %s",
               node_id, ty_to_str(self.tcx(), ty), self.tag()];
        self.node_types.insert(node_id as uint, ty);
    }
    fn write_substs(node_id: ast::node_id, +substs: ty::substs) {
        if !ty::substs_is_noop(substs) {
            self.node_type_substs.insert(node_id, substs);
        }
    }
    fn write_ty_substs(node_id: ast::node_id, ty: ty::t,
                       +substs: ty::substs) {
        let ty = ty::subst(self.tcx(), substs, ty);
        self.write_ty(node_id, ty);
        self.write_substs(node_id, substs);
    }
    fn write_nil(node_id: ast::node_id) {
        self.write_ty(node_id, ty::mk_nil(self.tcx()));
    }
    fn write_bot(node_id: ast::node_id) {
        self.write_ty(node_id, ty::mk_bot(self.tcx()));
    }

    fn to_ty(ast_t: @ast::ty) -> ty::t {
        ast_ty_to_ty(self, self, ast_t)
    }

    fn expr_ty(ex: @ast::expr) -> ty::t {
        alt self.node_types.find(ex.id as uint) {
          some(t) { t }
          none {
            self.tcx().sess.bug(#fmt["no type for expr %d (%s) in fcx %s",
                                     ex.id, expr_to_str(ex), self.tag()]);
          }
        }
    }
    fn node_ty(id: ast::node_id) -> ty::t {
        alt self.node_types.find(id as uint) {
          some(t) { t }
          none {
            self.tcx().sess.bug(
                #fmt["no type for node %d: %s in fcx %s",
                     id, ast_map::node_id_to_str(self.tcx().items, id),
                     self.tag()]);
          }
        }
    }
    fn node_ty_substs(id: ast::node_id) -> ty::substs {
        alt self.node_type_substs.find(id) {
          some(ts) { ts }
          none {
            self.tcx().sess.bug(
                #fmt["no type substs for node %d: %s in fcx %s",
                     id, ast_map::node_id_to_str(self.tcx().items, id),
                     self.tag()]);
          }
        }
    }
    fn opt_node_ty_substs(id: ast::node_id) -> option<ty::substs> {
        self.node_type_substs.find(id)
    }
    fn next_ty_var_id() -> ty_vid {
        let id = *self.ty_var_counter;
        *self.ty_var_counter += 1u;
        ret ty_vid(id);
    }
    fn next_ty_var() -> ty::t {
        ty::mk_var(self.ccx.tcx, self.next_ty_var_id())
    }
    fn next_ty_vars(n: uint) -> [ty::t] {
        vec::from_fn(n) {|_i| self.next_ty_var() }
    }
    fn next_region_var_id() -> region_vid {
        let id = *self.region_var_counter;
        *self.region_var_counter += 1u;
        ret region_vid(id);
    }
    fn next_region_var() -> ty::region {
        ret ty::re_var(self.next_region_var_id());
    }
    fn report_mismatched_types(sp: span, e: ty::t, a: ty::t,
                               err: ty::type_err) {
        self.ccx.tcx.sess.span_err(
            sp,
            #fmt["mismatched types: expected `%s` but found `%s` (%s)",
                 self.ty_to_str(e),
                 self.ty_to_str(a),
                 ty::type_err_to_str(self.ccx.tcx, err)]);
    }
}

fn mk_ty_params(ccx: @crate_ctxt, atps: [ast::ty_param])
    -> {bounds: @[ty::param_bounds], params: [ty::t]} {

    let mut i = 0u;
    let bounds = ty_param_bounds(ccx, atps);
    {bounds: bounds,
     params: vec::map(atps, {|atp|
         let t = ty::mk_param(ccx.tcx, i, local_def(atp.id));
         i += 1u;
         t
     })}
}

fn mk_substs(ccx: @crate_ctxt, atps: [ast::ty_param], rp: ast::region_param)
    -> {bounds: @[ty::param_bounds], substs: ty::substs} {

    let {bounds, params} = mk_ty_params(ccx, atps);
    let self_r = alt rp {
      ast::rp_self { some(ty::re_bound(ty::br_self)) }
      ast::rp_none { none }
    };
    {bounds: bounds, substs: {self_r: self_r, tps: params}}
}

fn compare_impl_method(tcx: ty::ctxt, sp: span, impl_m: ty::method,
                       impl_tps: uint, if_m: ty::method, substs: [ty::t],
                       self_ty: ty::t) -> ty::t {
    if impl_m.tps != if_m.tps {
        tcx.sess.span_err(sp, "method `" + if_m.ident +
                          "` has an incompatible set of type parameters");
        ty::mk_fn(tcx, impl_m.fty)
    } else if vec::len(impl_m.fty.inputs) != vec::len(if_m.fty.inputs) {
        tcx.sess.span_err(sp,#fmt["method `%s` has %u parameters \
                                   but the iface has %u",
                                  if_m.ident,
                                  vec::len(impl_m.fty.inputs),
                                  vec::len(if_m.fty.inputs)]);
        ty::mk_fn(tcx, impl_m.fty)
    } else {
        let auto_modes = vec::map2(impl_m.fty.inputs, if_m.fty.inputs, {|i, f|
            alt ty::get(f.ty).struct {
              ty::ty_param(_, _) | ty::ty_self(_)
              if alt i.mode { ast::infer(_) { true } _ { false } } {
                {mode: ast::expl(ast::by_ref) with i}
              }
              _ { i }
            }
        });
        let impl_fty = ty::mk_fn(tcx, {inputs: auto_modes with impl_m.fty});

        // Add dummy substs for the parameters of the impl method
        let substs = substs + vec::from_fn(vec::len(*if_m.tps), {|i|
            ty::mk_param(tcx, i + impl_tps, {crate: 0, node: 0})
        });
        let mut if_fty = ty::mk_fn(tcx, if_m.fty);
        if_fty = ty::subst_tps(tcx, substs, if_fty);
        if_fty = fixup_self_full(tcx, if_fty, substs, self_ty, impl_tps);
        require_same_types(
            tcx, sp, impl_fty, if_fty,
            {|| "method `" + if_m.ident +
                 "` has an incompatible type"});
        ret impl_fty;
    }
}

// Mangles an iface method ty to make its self type conform to the self type
// of a specific impl or bounded type parameter. This is rather involved
// because the type parameters of ifaces and impls are not required to line up
// (an impl can have less or more parameters than the iface it implements), so
// some mangling of the substituted types is required.
fn fixup_self_full(cx: ty::ctxt, mty: ty::t, m_substs: [ty::t],
                   selfty: ty::t, impl_n_tps: uint) -> ty::t {

    if !ty::type_has_vars(mty) { ret mty; }

    ty::fold_ty(cx, mty) {|t|
        alt ty::get(t).struct {
          ty::ty_self(tps) if vec::len(tps) == 0u { selfty }
          ty::ty_self(tps) {
            // Move the substs into the type param system of the
            // context.
            let mut substs = vec::map(tps) {|t|
                let f = fixup_self_full(cx, t, m_substs, selfty, impl_n_tps);
                ty::subst_tps(cx, m_substs, f)
            };

            // Add extra substs for impl type parameters.
            while vec::len(substs) < impl_n_tps {
                substs += [ty::mk_param(cx, vec::len(substs),
                                        {crate: 0, node: 0})];
            }

            // And for method type parameters.
            let method_n_tps =
                (vec::len(m_substs) - vec::len(tps)) as int;
            if method_n_tps > 0 {
                substs += vec::tailn(m_substs, vec::len(m_substs)
                                     - (method_n_tps as uint));
            }

            // And then instantiate the self type using all those.
            ty::subst_tps(cx, substs, selfty)
          }
          _ {
              t
          }
        }
    }
}

// Mangles an iface method ty to make its self type conform to the self type
// of a specific impl or bounded type parameter. This is rather involved
// because the type parameters of ifaces and impls are not required to line up
// (an impl can have less or more parameters than the iface it implements), so
// some mangling of the substituted types is required.
fn fixup_self_param(fcx: @fn_ctxt, mty: ty::t, m_substs: [ty::t],
                    selfty: ty::t, sp: span) -> ty::t {
    if !ty::type_has_vars(mty) { ret mty; }

    let tcx = fcx.ccx.tcx;
    ty::fold_ty(tcx, mty) {|t|
        alt ty::get(t).struct {
          ty::ty_self(tps) if vec::len(tps) == 0u { selfty }
          ty::ty_self(tps) {
            // Move the substs into the type param system of the
            // context.
            let mut substs = vec::map(tps) {|t|
                let f = fixup_self_param(fcx, t, m_substs, selfty, sp);
                ty::subst_tps(tcx, m_substs, f)
            };

            // Simply ensure that the type parameters for the self
            // type match the context.
            vec::iter2(substs, m_substs) {|s, ms|
                demand::suptype(fcx, sp, s, ms);
            }
            selfty
          }
          _ { t }
        }
    }
}

// Replaces all occurrences of the `self` region with `with_region`.  Note
// that we descend into `fn()` types here, because `fn()` does not bind the
// `self` region.
fn replace_self_region(tcx: ty::ctxt, with_region: ty::region,
                       ty: ty::t) -> ty::t {

   ty::fold_region(tcx, ty) {|r, _under_rptr|
       alt r {
           ty::re_bound(re_self) { with_region }
           _ { r }
       }
   }
}

// Item collection - a pair of bootstrap passes:
//
// (1) Collect the IDs of all type items (typedefs) and store them in a table.
//
// (2) Translate the AST fragments that describe types to determine a type for
//     each item. When we encounter a named type, we consult the table built
//     in pass 1 to find its item, and recursively translate it.
//
// We then annotate the AST with the resulting types and return the annotated
// AST, along with a table mapping item IDs to their types.
mod collect {
    fn get_enum_variant_types(ccx: @crate_ctxt,
                              enum_ty: ty::t,
                              variants: [ast::variant],
                              ty_params: [ast::ty_param],
                              rp: ast::region_param) {
        let tcx = ccx.tcx;

        // Create a set of parameter types shared among all the variants.
        for variants.each {|variant|
            // Nullary enum constructors get turned into constants; n-ary enum
            // constructors get turned into functions.
            let result_ty = if vec::len(variant.node.args) == 0u {
                enum_ty
            } else {
                let rs = type_rscope(rp);
                let args = variant.node.args.map { |va|
                    let arg_ty = ccx.to_ty(rs, va.ty);
                    {mode: ast::expl(ast::by_copy), ty: arg_ty}
                };
                ty::mk_fn(tcx, {proto: ast::proto_box,
                                inputs: args,
                                output: enum_ty,
                                ret_style: ast::return_val,
                                constraints: []})
            };
            let tpt = {bounds: ty_param_bounds(ccx, ty_params),
                       rp: rp,
                       ty: result_ty};
            tcx.tcache.insert(local_def(variant.node.id), tpt);
            write_ty_to_tcx(tcx, variant.node.id, result_ty);
        }
    }

    fn ensure_iface_methods(ccx: @crate_ctxt, id: ast::node_id) {
        fn store_methods<T>(ccx: @crate_ctxt, id: ast::node_id,
                            stuff: [T], f: fn@(T) -> ty::method) {
            ty::store_iface_methods(ccx.tcx, id, @vec::map(stuff, f));
        }

        let tcx = ccx.tcx;
        alt check tcx.items.get(id) {
          ast_map::node_item(@{node: ast::item_iface(_, ms), _}, _) {
              store_methods::<ast::ty_method>(ccx, id, ms) {|m|
                  ty_of_ty_method(ccx, m)
              };
          }
          ast_map::node_item(@{node: ast::item_class(_,_,its,_,rp), _}, _) {
              let (_,ms) = split_class_items(its);
              // All methods need to be stored, since lookup_method
              // relies on the same method cache for self-calls
              store_methods::<@ast::method>(ccx, id, ms) {|m|
                  ty_of_method(ccx, m, rp)
              };
          }
        }
    }

    fn check_methods_against_iface(ccx: @crate_ctxt,
                                   tps: [ast::ty_param],
                                   rp: ast::region_param,
                                   selfty: ty::t,
                                   t: @ast::ty,
                                   ms: [@ast::method]) {

        let tcx = ccx.tcx;
        let i_bounds = ty_param_bounds(ccx, tps);
        let my_methods = convert_methods(ccx, ms, rp, i_bounds, selfty);
        let iface_ty = ccx.to_ty(empty_rscope, t);
        alt ty::get(iface_ty).struct {
          ty::ty_iface(did, tys) {
            // Store the iface type in the type node
            alt check t.node {
              ast::ty_path(_, t_id) {
                write_ty_to_tcx(tcx, t_id, iface_ty);
              }
            }
            if did.crate == ast::local_crate {
                ensure_iface_methods(ccx, did.node);
            }
            for vec::each(*ty::iface_methods(tcx, did)) {|if_m|
                alt vec::find(my_methods,
                              {|m| if_m.ident == m.mty.ident}) {
                  some({mty: m, id, span}) {
                    if m.purity != if_m.purity {
                        ccx.tcx.sess.span_err(
                            span, #fmt["method `%s`'s purity \
                                        not match the iface method's \
                                        purity", m.ident]);
                    }
                    let mt = compare_impl_method(
                        ccx.tcx, span, m, vec::len(tps),
                        if_m, tys, selfty);
                    let old = tcx.tcache.get(local_def(id));
                    if old.ty != mt {
                        tcx.tcache.insert(
                            local_def(id),
                            {bounds: old.bounds,
                             rp: old.rp,
                             ty: mt});
                        write_ty_to_tcx(tcx, id, mt);
                    }
                  }
                  none {
                    tcx.sess.span_err(t.span, "missing method `" +
                                      if_m.ident + "`");
                  }
                } // alt
            } // |if_m|
          } // for
          _ {
            tcx.sess.span_fatal(t.span, "can only implement \
                                         interface types");
        }
      }
    }

    fn convert_class_item(ccx: @crate_ctxt,
                          rp: ast::region_param,
                          v: ast_util::ivar) {
        /* they have these types *within the scope* of the
         class. outside the class, it's done with expr_field */
        let tt = ccx.to_ty(type_rscope(rp), v.ty);
        #debug("convert_class_item: %s %?", v.ident, v.id);
        write_ty_to_tcx(ccx.tcx, v.id, tt);
    }

    fn convert_methods(ccx: @crate_ctxt,
                       ms: [@ast::method],
                       rp: ast::region_param,
                       i_bounds: @[ty::param_bounds],
                       self_ty: ty::t)
        -> [{mty: ty::method, id: ast::node_id, span: span}] {

        let tcx = ccx.tcx;
        vec::map(ms) { |m|
            write_ty_to_tcx(tcx, m.self_id, self_ty);
            let bounds = ty_param_bounds(ccx, m.tps);
            let mty = ty_of_method(ccx, m, rp);
            let fty = ty::mk_fn(tcx, mty.fty);
            tcx.tcache.insert(
                local_def(m.id),
                {bounds: @(*i_bounds + *bounds), rp: rp, ty: fty});
            write_ty_to_tcx(tcx, m.id, fty);
            {mty: mty, id: m.id, span: m.span}
        }
    }

    fn convert(ccx: @crate_ctxt, it: @ast::item) {
        let tcx = ccx.tcx;
        alt it.node {
          // These don't define types.
          ast::item_mod(_) {}
          ast::item_native_mod(m) {
            if syntax::attr::native_abi(it.attrs) ==
               either::right(ast::native_abi_rust_intrinsic) {
                for m.items.each { |item| check_intrinsic_type(ccx, item); }
            }
          }
          ast::item_enum(variants, ty_params, rp) {
            let tpt = ty_of_item(ccx, it);
            write_ty_to_tcx(tcx, it.id, tpt.ty);
            get_enum_variant_types(ccx, tpt.ty, variants,
                                   ty_params, rp);
          }
          ast::item_impl(tps, ifce, selfty, ms) {
            let i_bounds = ty_param_bounds(ccx, tps);
            // NDM iface/impl regions
            let selfty = ccx.to_ty(empty_rscope, selfty);
            write_ty_to_tcx(tcx, it.id, selfty);
            tcx.tcache.insert(local_def(it.id),
                              {bounds: i_bounds,
                               rp: ast::rp_none, // NDM iface/impl regions
                               ty: selfty});
            alt ifce {
              some(t) {
                check_methods_against_iface(
                    ccx, tps, ast::rp_none, // NDM iface/impl regions
                    selfty, t, ms);
              }
              _ {
                // Still have to do this to write method types
                // into the table
                convert_methods(
                    ccx, ms, ast::rp_none, // NDM iface/impl regions
                    i_bounds, selfty);
              }
            }
          }
          ast::item_res(decl, tps, _, dtor_id, ctor_id, rp) {
            let {bounds, substs} = mk_substs(ccx, tps, rp);
            let def_id = local_def(it.id);
            let t_arg = ty_of_arg(ccx, type_rscope(rp), decl.inputs[0]);
            let t_res = ty::mk_res(tcx, def_id, t_arg.ty, substs);

            let t_ctor = ty::mk_fn(tcx, {
                proto: ast::proto_box,
                inputs: [{mode: ast::expl(ast::by_copy), ty: t_arg.ty}],
                output: t_res,
                ret_style: ast::return_val, constraints: []
            });
            let t_dtor = ty::mk_fn(tcx, {
                proto: ast::proto_box,
                inputs: [t_arg], output: ty::mk_nil(tcx),
                ret_style: ast::return_val, constraints: []
            });
            write_ty_to_tcx(tcx, it.id, t_res);
            write_ty_to_tcx(tcx, ctor_id, t_ctor);
            tcx.tcache.insert(local_def(ctor_id),
                              {bounds: bounds,
                               rp: ast::rp_none,
                               ty: t_ctor});
            tcx.tcache.insert(def_id, {bounds: bounds,
                                       rp: ast::rp_none,
                                       ty: t_res});
            write_ty_to_tcx(tcx, dtor_id, t_dtor);
          }
          ast::item_iface(_, ms) {
            let tpt = ty_of_item(ccx, it);
            write_ty_to_tcx(tcx, it.id, tpt.ty);
            ensure_iface_methods(ccx, it.id);
          }
          ast::item_class(tps, ifaces, members, ctor, rp) {
            // Write the class type
            let tpt = ty_of_item(ccx, it);
            write_ty_to_tcx(tcx, it.id, tpt.ty);
            // Write the ctor type
            let t_ctor =
                ty::mk_fn(tcx,
                          ty_of_fn_decl(ccx,
                                        empty_rscope,
                                        ast::proto_any,
                                        ctor.node.dec));
            write_ty_to_tcx(tcx, ctor.node.id, t_ctor);
            tcx.tcache.insert(local_def(ctor.node.id),
                              {bounds: tpt.bounds,
                               rp: ast::rp_none, // NDM self->anon
                               ty: t_ctor});
            ensure_iface_methods(ccx, it.id);
            /* FIXME: check for proper public/privateness */
            // Write the type of each of the members
            let (fields, methods) = split_class_items(members);
            for fields.each {|f|
                convert_class_item(ccx, rp, f);
            }
            // The selfty is just the class type
            let {bounds:_, substs} = mk_substs(ccx, tps, rp);
            let selfty = ty::mk_class(tcx, local_def(it.id), substs);
            // Need to convert all methods so we can check internal
            // references to private methods

            // NDM to TJC---I think we ought to be using bounds here, not @[].
            // But doing so causes errors later on.
            convert_methods(ccx, methods, rp, @[], selfty);

            /*
            Finally, check that the class really implements the ifaces
            that it claims to implement.
            */
            for ifaces.each { |ifce|
                alt lookup_def_tcx(tcx, it.span, ifce.id) {
                  ast::def_ty(t_id) {
                    let t = ty::lookup_item_type(tcx, t_id).ty;
                    alt ty::get(t).struct {
                      ty::ty_iface(_,_) {
                        write_ty_to_tcx(tcx, ifce.id, t);
                        check_methods_against_iface(
                            ccx, tps, rp, selfty,
                            @{id: ifce.id,
                              node: ast::ty_path(ifce.path, ifce.id),
                              span: ifce.path.span},
                            methods);
                      }
                      _ {
                        tcx.sess.span_fatal(
                            ifce.path.span,
                            "can only implement interface types");
                      }
                    }
                  }
                  _ { tcx.sess.span_err(ifce.path.span, "not an interface \
                                                         type"); }
                }
            }
          }
          _ {
            // This call populates the type cache with the converted type
            // of the item in passing. All we have to do here is to write
            // it into the node type table.
            let tpt = ty_of_item(ccx, it);
            write_ty_to_tcx(tcx, it.id, tpt.ty);
          }
        }
    }
    fn convert_native(ccx: @crate_ctxt, i: @ast::native_item) {
        // As above, this call populates the type table with the converted
        // type of the native item. We simply write it into the node type
        // table.
        let tpt = ty_of_native_item(ccx, i);
        alt i.node {
          ast::native_item_fn(_, _) {
            write_ty_to_tcx(ccx.tcx, i.id, tpt.ty);
          }
        }
    }
    fn collect_item_types(ccx: @crate_ctxt, crate: @ast::crate) {
        visit::visit_crate(*crate, (), visit::mk_simple_visitor(@{
            visit_item: bind convert(ccx, _),
            visit_native_item: bind convert_native(ccx, _)
            with *visit::default_simple_visitor()
        }));
    }
}


// Type unification
mod unify {
    fn unify(fcx: @fn_ctxt, expected: ty::t, actual: ty::t) ->
        result<(), ty::type_err> {
        ret infer::mk_subty(fcx.infcx, actual, expected);
    }
}


// FIXME This is almost a duplicate of ty::type_autoderef, with structure_of
// instead of ty::struct.
fn do_autoderef(fcx: @fn_ctxt, sp: span, t: ty::t) -> ty::t {
    let mut t1 = t;
    let mut enum_dids = [];
    loop {
        alt structure_of(fcx, sp, t1) {
          ty::ty_box(inner) | ty::ty_uniq(inner) | ty::ty_rptr(_, inner) {
            alt ty::get(t1).struct {
              ty::ty_var(v1) {
                ty::occurs_check(fcx.ccx.tcx, sp, v1,
                                 ty::mk_box(fcx.ccx.tcx, inner));
              }
              _ { }
            }
            t1 = inner.ty;
          }
          ty::ty_res(_, inner, substs) {
            t1 = ty::subst(fcx.ccx.tcx, substs, inner);
          }
          ty::ty_enum(did, substs) {
            // Watch out for a type like `enum t = @t`.  Such a type would
            // otherwise infinitely auto-deref.  This is the only autoderef
            // loop that needs to be concerned with this, as an error will be
            // reported on the enum definition as well because the enum is not
            // instantiable.
            if vec::contains(enum_dids, did) {
                ret t1;
            }
            vec::push(enum_dids, did);

            let variants = ty::enum_variants(fcx.ccx.tcx, did);
            if vec::len(*variants) != 1u || vec::len(variants[0].args) != 1u {
                ret t1;
            }
            t1 = ty::subst(fcx.ccx.tcx, substs, variants[0].args[0]);
          }
          _ { ret t1; }
        }
    };
}

fn resolve_type_vars_if_possible(fcx: @fn_ctxt, typ: ty::t) -> ty::t {
    alt infer::resolve_deep(fcx.infcx, typ, false) {
      result::ok(new_type) { ret new_type; }
      result::err(_) { ret typ; }
    }
}

// Demands - procedures that require that two types unify and emit an error
// message if they don't.
type ty_param_substs_and_ty = {substs: ty::substs, ty: ty::t};

fn require_same_types(
    tcx: ty::ctxt,
    span: span,
    t1: ty::t,
    t2: ty::t,
    msg: fn() -> str) -> bool {

    alt infer::compare_tys(tcx, t1, t2) {
      result::ok(()) { true }
      result::err(terr) {
        tcx.sess.span_err(
            span, msg() + ": " +
            ty::type_err_to_str(tcx, terr));
        false
      }
    }
}

mod demand {
    // Requires that the two types unify, and prints an error message if they
    // don't.
    fn suptype(fcx: @fn_ctxt, sp: span,
              expected: ty::t, actual: ty::t) {

        alt infer::mk_subty(fcx.infcx, actual, expected) {
          result::ok(()) { /* ok */ }
          result::err(err) {
            fcx.report_mismatched_types(sp, expected, actual, err);
          }
        }
    }

    // Checks that the type `actual` can be assigned to `expected`.
    fn assign(fcx: @fn_ctxt, sp: span, expected: ty::t, expr: @ast::expr) {
        let expr_ty = fcx.expr_ty(expr);
        alt infer::mk_assignty(fcx.infcx, expr.id, expr_ty, expected) {
          result::ok(()) { /* ok */ }
          result::err(err) {
            fcx.report_mismatched_types(sp, expected, expr_ty, err);
          }
        }
    }
}


// Returns true if the two types unify and false if they don't.
fn are_compatible(fcx: @fn_ctxt, expected: ty::t, actual: ty::t) -> bool {
    alt unify::unify(fcx, expected, actual) {
      result::ok(_) { ret true; }
      result::err(_) { ret false; }
    }
}


// Type resolution: the phase that finds all the types in the AST with
// unresolved type variables and replaces "ty_var" types with their
// substitutions.
mod writeback {

    export resolve_type_vars_in_fn;
    export resolve_type_vars_in_expr;

    fn resolve_type_vars_in_type(fcx: @fn_ctxt, sp: span, typ: ty::t) ->
       option<ty::t> {
        if !ty::type_has_vars(typ) { ret some(typ); }
        alt infer::resolve_deep(fcx.infcx, typ, true) {
          result::ok(new_type) { ret some(new_type); }
          result::err(e) {
            if !fcx.ccx.tcx.sess.has_errors() {
                fcx.ccx.tcx.sess.span_err(
                    sp,
                    #fmt["cannot determine a type \
                          for this expression: %s",
                         infer::fixup_err_to_str(e)])
            }
            ret none;
          }
        }
    }
    fn resolve_type_vars_for_node(wbcx: wb_ctxt, sp: span, id: ast::node_id)
        -> option<ty::t> {
        let fcx = wbcx.fcx, tcx = fcx.ccx.tcx;
        let n_ty = fcx.node_ty(id);
        alt resolve_type_vars_in_type(fcx, sp, n_ty) {
          none {
            wbcx.success = false;
            ret none;
          }

          some(t) {
            #debug["resolve_type_vars_for_node(id=%d, n_ty=%s, t=%s)",
                   id, ty_to_str(tcx, n_ty), ty_to_str(tcx, t)];
            write_ty_to_tcx(tcx, id, t);
            alt fcx.opt_node_ty_substs(id) {
              some(substs) {
                let mut new_tps = [];
                for substs.tps.each {|subst|
                    alt resolve_type_vars_in_type(fcx, sp, subst) {
                      some(t) { new_tps += [t]; }
                      none { wbcx.success = false; ret none; }
                    }
                }
                write_substs_to_tcx(tcx, id, new_tps);
              }
              none {}
            }
            ret some(t);
          }
        }
    }

    fn maybe_resolve_type_vars_for_node(wbcx: wb_ctxt, sp: span,
                                        id: ast::node_id)
        -> option<ty::t> {
        if wbcx.fcx.node_types.contains_key(id as uint) {
            resolve_type_vars_for_node(wbcx, sp, id)
        } else {
            none
        }
    }

    type wb_ctxt =
        // As soon as we hit an error we have to stop resolving
        // the entire function
        {fcx: @fn_ctxt, mut success: bool};
    type wb_vt = visit::vt<wb_ctxt>;

    fn visit_stmt(s: @ast::stmt, wbcx: wb_ctxt, v: wb_vt) {
        if !wbcx.success { ret; }
        resolve_type_vars_for_node(wbcx, s.span, ty::stmt_node_id(s));
        visit::visit_stmt(s, wbcx, v);
    }
    fn visit_expr(e: @ast::expr, wbcx: wb_ctxt, v: wb_vt) {
        if !wbcx.success { ret; }
        resolve_type_vars_for_node(wbcx, e.span, e.id);
        alt e.node {
          ast::expr_fn(_, decl, _, _) |
          ast::expr_fn_block(decl, _) {
            vec::iter(decl.inputs) {|input|
                let r_ty = resolve_type_vars_for_node(wbcx, e.span, input.id);

                // Just in case we never constrained the mode to anything,
                // constrain it to the default for the type in question.
                alt (r_ty, input.mode) {
                  (some(t), ast::infer(_)) {
                    let tcx = wbcx.fcx.ccx.tcx;
                    let m_def = ty::default_arg_mode_for_ty(t);
                    ty::set_default_mode(tcx, input.mode, m_def);
                  }
                  _ {}
                }
            }
          }

          ast::expr_new(_, alloc_id, _) {
            resolve_type_vars_for_node(wbcx, e.span, alloc_id);
          }

          ast::expr_binary(_, _, _) | ast::expr_unary(_, _) |
          ast::expr_assign_op(_, _, _) | ast::expr_index(_, _) {
            maybe_resolve_type_vars_for_node(wbcx, e.span,
                                             ast_util::op_expr_callee_id(e));
          }

          _ { }
        }
        visit::visit_expr(e, wbcx, v);
    }
    fn visit_block(b: ast::blk, wbcx: wb_ctxt, v: wb_vt) {
        if !wbcx.success { ret; }
        resolve_type_vars_for_node(wbcx, b.span, b.node.id);
        visit::visit_block(b, wbcx, v);
    }
    fn visit_pat(p: @ast::pat, wbcx: wb_ctxt, v: wb_vt) {
        if !wbcx.success { ret; }
        resolve_type_vars_for_node(wbcx, p.span, p.id);
        #debug["Type for pattern binding %s (id %d) resolved to %s",
               pat_to_str(p), p.id,
               wbcx.fcx.ty_to_str(
                   ty::node_id_to_type(wbcx.fcx.ccx.tcx,
                                       p.id))];
        visit::visit_pat(p, wbcx, v);
    }
    fn visit_local(l: @ast::local, wbcx: wb_ctxt, v: wb_vt) {
        if !wbcx.success { ret; }
        let var_id = lookup_local(wbcx.fcx, l.span, l.node.id);
        alt infer::resolve_deep_var(wbcx.fcx.infcx, var_id, true) {
          result::ok(lty) {
            #debug["Type for local %s (id %d) resolved to %s",
                   pat_to_str(l.node.pat), l.node.id,
                   wbcx.fcx.ty_to_str(lty)];
            write_ty_to_tcx(wbcx.fcx.ccx.tcx, l.node.id, lty);
          }
          result::err(e) {
            wbcx.fcx.ccx.tcx.sess.span_err(
                l.span,
                #fmt["cannot determine a type \
                      for this local variable: %s",
                     infer::fixup_err_to_str(e)]);
            wbcx.success = false;
          }
        }
        visit::visit_local(l, wbcx, v);
    }
    fn visit_item(_item: @ast::item, _wbcx: wb_ctxt, _v: wb_vt) {
        // Ignore items
    }

    fn resolve_type_vars_in_expr(fcx: @fn_ctxt, e: @ast::expr) -> bool {
        let wbcx = {fcx: fcx, mut success: true};
        let visit =
            visit::mk_vt(@{visit_item: visit_item,
                           visit_stmt: visit_stmt,
                           visit_expr: visit_expr,
                           visit_block: visit_block,
                           visit_pat: visit_pat,
                           visit_local: visit_local
                              with *visit::default_visitor()});
        visit.visit_expr(e, wbcx, visit);
        ret wbcx.success;
    }

    fn resolve_type_vars_in_fn(fcx: @fn_ctxt,
                               decl: ast::fn_decl,
                               blk: ast::blk) -> bool {
        let wbcx = {fcx: fcx, mut success: true};
        let visit =
            visit::mk_vt(@{visit_item: visit_item,
                           visit_stmt: visit_stmt,
                           visit_expr: visit_expr,
                           visit_block: visit_block,
                           visit_pat: visit_pat,
                           visit_local: visit_local
                              with *visit::default_visitor()});
        visit.visit_block(blk, wbcx, visit);
        for decl.inputs.each {|arg|
            resolve_type_vars_for_node(wbcx, arg.ty.span, arg.id);
        }
        ret wbcx.success;
    }

}

fn check_intrinsic_type(ccx: @crate_ctxt, it: @ast::native_item) {
    fn param(ccx: @crate_ctxt, n: uint) -> ty::t {
        ty::mk_param(ccx.tcx, n, local_def(0))
    }
    fn arg(m: ast::rmode, ty: ty::t) -> ty::arg {
        {mode: ast::expl(m), ty: ty}
    }
    let tcx = ccx.tcx;
    let (n_tps, inputs, output) = alt it.ident {
      "size_of" | "align_of" { (1u, [], ty::mk_uint(ccx.tcx)) }
      "get_tydesc" { (1u, [], ty::mk_nil_ptr(tcx)) }
      "init" { (1u, [], param(ccx, 0u)) }
      "forget" { (1u, [arg(ast::by_move, param(ccx, 0u))],
                  ty::mk_nil(tcx)) }
      "reinterpret_cast" { (2u, [arg(ast::by_ref, param(ccx, 0u))],
                            param(ccx, 1u)) }
      "addr_of" { (1u, [arg(ast::by_ref, param(ccx, 0u))],
                   ty::mk_imm_ptr(tcx, param(ccx, 0u))) }
      "needs_drop" { (1u, [], ty::mk_bool(tcx)) }
      other {
        tcx.sess.span_err(it.span, "unrecognized intrinsic function: `" +
                          other + "`");
        ret;
      }
    };
    let fty = ty::mk_fn(tcx, {proto: ast::proto_bare,
                              inputs: inputs, output: output,
                              ret_style: ast::return_val,
                              constraints: []});
    let i_ty = ty_of_native_item(ccx, it);
    let i_n_tps = (*i_ty.bounds).len();
    if i_n_tps != n_tps {
        tcx.sess.span_err(it.span, #fmt("intrinsic has wrong number \
                                         of type parameters. found %u, \
                                         expected %u", i_n_tps, n_tps));
    } else {
        require_same_types(
            tcx, it.span, i_ty.ty, fty,
            {|| #fmt["intrinsic has wrong type. \
                      expected %s",
                     ty_to_str(ccx.tcx, fty)]});
    }
}

// Local variable gathering. We gather up all locals and create variable IDs
// for them before typechecking the function.
type gather_result =
    {infcx: infer::infer_ctxt,
     locals: hashmap<ast::node_id, ty_vid>,
     ty_var_counter: @mut uint};

// AST fragment checking
fn check_lit(ccx: @crate_ctxt, lit: @ast::lit) -> ty::t {
    alt lit.node {
      ast::lit_str(_) { ty::mk_str(ccx.tcx) }
      ast::lit_int(_, t) { ty::mk_mach_int(ccx.tcx, t) }
      ast::lit_uint(_, t) { ty::mk_mach_uint(ccx.tcx, t) }
      ast::lit_float(_, t) { ty::mk_mach_float(ccx.tcx, t) }
      ast::lit_nil { ty::mk_nil(ccx.tcx) }
      ast::lit_bool(_) { ty::mk_bool(ccx.tcx) }
    }
}

fn valid_range_bounds(ccx: @crate_ctxt, from: @ast::expr, to: @ast::expr)
    -> bool {
    const_eval::compare_lit_exprs(ccx.tcx, from, to) <= 0
}

type pat_ctxt = {
    fcx: @fn_ctxt,
    map: pat_util::pat_id_map,
    alt_region: ty::region,
    block_region: ty::region,
    /* Equal to either alt_region or block_region. */
    pat_region: ty::region
};

// Helper for the other universally_quantify_*() routines.  Extracts the bound
// regions from bound_tys and then replaces those same regions with fresh
// variables in `sty`, returning the resulting type.
fn universally_quantify_from_sty(fcx: @fn_ctxt,
                                 span: span,
                                 bound_tys: [ty::t],
                                 sty: ty::sty) -> ty::t {

    #debug["universally_quantify_from_sty(bound_tys=%?)",
           bound_tys.map {|x| fcx.ty_to_str(x) }];
    indent {||
        let tcx = fcx.tcx();
        let isr = collect_bound_regions_in_tys(tcx, @nil, bound_tys) { |br|
            let rvar = fcx.next_region_var();
            #debug["Bound region %s maps to %s",
                   bound_region_to_str(fcx.ccx.tcx, br),
                   region_to_str(fcx.ccx.tcx, rvar)];
            rvar
        };
        let t_res = ty::fold_sty_to_ty(fcx.ccx.tcx, sty) { |t|
            replace_bound_regions(tcx, span, isr, t)
        };
        #debug["Result of universal quant. is %s", fcx.ty_to_str(t_res)];
        t_res
    }
}

// Replaces all region parameters in the given type with region variables.
// Does not descend into fn types.  This is used when deciding whether an impl
// applies at a given call site.  See also universally_quantify_before_call().
fn universally_quantify_regions(fcx: @fn_ctxt,
                                span: span,
                                ty: ty::t) -> ty::t {
    universally_quantify_from_sty(fcx, span, [ty], ty::get(ty).struct)
}

// Expects a function type.  Replaces all region parameters in the arguments
// and return type with fresh region variables. This is used when typechecking
// function calls, bind expressions, and method calls.
fn universally_quantify_before_call(fcx: @fn_ctxt,
                                    span: span,
                                    ty: ty::t) -> ty::t {

    #debug["universally_quantify_before_call(ty=%s)",
           fcx.ty_to_str(ty)];

    // This is subtle: we expect `ty` to be a function type, which normally
    // introduce a level of binding.  In this case, we want to process the
    // types bound by the function but not by any nested functions.
    // Therefore, we match one level of structure.
    alt structure_of(fcx, span, ty) {
      sty @ ty::ty_fn(fty) {
        let all_tys = fty.inputs.map({|a| a.ty}) + [fty.output];
        universally_quantify_from_sty(fcx, span, all_tys, sty)
      }
      sty {
        #debug["not a fn ty: %?", sty];

        // if not a function type, we're gonna' report an error
        // at some point, since the user is trying to call this thing
        ty
      }
    }
}

fn check_pat_variant(pcx: pat_ctxt, pat: @ast::pat, path: @ast::path,
                     subpats: option<[@ast::pat]>, expected: ty::t) {

    // Typecheck the path.
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    // Lookup the enum and variant def ids:
    let v_def = lookup_def(pcx.fcx, path.span, pat.id);
    let v_def_ids = ast_util::variant_def_ids(v_def);

    // Assign the pattern the type of the *enum*, not the variant.
    let enum_tpt = ty::lookup_item_type(tcx, v_def_ids.enm);
    instantiate_path(pcx.fcx, path, enum_tpt, pat.span, pat.id);

    // Take the enum type params out of `expected`.
    alt structure_of(pcx.fcx, pat.span, expected) {
      ty::ty_enum(_, expected_substs) {
        // check that the type of the value being matched is a subtype
        // of the type of the pattern:
        let pat_ty = fcx.node_ty(pat.id);
        demand::suptype(fcx, pat.span, pat_ty, expected);

        // Get the expected types of the arguments.
        let arg_types = {
            let vinfo =
                ty::enum_variant_with_id(
                    tcx, v_def_ids.enm, v_def_ids.var);
            vinfo.args.map { |t| ty::subst(tcx, expected_substs, t) }
        };
        let arg_len = arg_types.len(), subpats_len = alt subpats {
            none { arg_len }
            some(ps) { ps.len() }};
        if arg_len > 0u {
            // N-ary variant.
            if arg_len != subpats_len {
                let s = #fmt["this pattern has %u field%s, but the \
                              corresponding variant has %u field%s",
                             subpats_len,
                             if subpats_len == 1u { "" } else { "s" },
                             arg_len,
                             if arg_len == 1u { "" } else { "s" }];
                tcx.sess.span_fatal(pat.span, s);
            }

            option::iter(subpats) {|pats|
                vec::iter2(pats, arg_types) {|subpat, arg_ty|
                  check_pat(pcx, subpat, arg_ty);
                }
            };
        } else if subpats_len > 0u {
            tcx.sess.span_fatal
                (pat.span, #fmt["this pattern has %u field%s, \
                                 but the corresponding variant has no fields",
                                subpats_len,
                                if subpats_len == 1u { "" }
                                else { "s" }]);
        }
      }
      _ {
        tcx.sess.span_fatal
            (pat.span,
             #fmt["mismatched types: expected enum but found `%s`",
                  fcx.ty_to_str(expected)]);
      }
    }
}

// Pattern checking is top-down rather than bottom-up so that bindings get
// their types immediately.
fn check_pat(pcx: pat_ctxt, pat: @ast::pat, expected: ty::t) {
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;
    alt pat.node {
      ast::pat_wild {
        fcx.write_ty(pat.id, expected);
      }
      ast::pat_lit(lt) {
        check_expr_with(pcx.fcx, lt, expected);
        fcx.write_ty(pat.id, fcx.expr_ty(lt));
      }
      ast::pat_range(begin, end) {
        check_expr_with(pcx.fcx, begin, expected);
        check_expr_with(pcx.fcx, end, expected);
        let b_ty = resolve_type_vars_if_possible(pcx.fcx,
                                                 fcx.expr_ty(begin));
        if !require_same_types(
            tcx, pat.span, b_ty,
            resolve_type_vars_if_possible(
                pcx.fcx, fcx.expr_ty(end)),
            {|| "mismatched types in range" }) {
            // no-op
        } else if !ty::type_is_numeric(b_ty) {
            tcx.sess.span_err(pat.span, "non-numeric type used in range");
        } else if !valid_range_bounds(pcx.fcx.ccx, begin, end) {
            tcx.sess.span_err(begin.span, "lower range bound must be less \
                                           than upper");
        }
        fcx.write_ty(pat.id, b_ty);
      }
      ast::pat_ident(name, sub)
      if !pat_util::pat_is_variant(tcx.def_map, pat) {
        let vid = lookup_local(pcx.fcx, pat.span, pat.id);
        let mut typ = ty::mk_var(tcx, vid);
        demand::suptype(pcx.fcx, pat.span, expected, typ);
        let canon_id = pcx.map.get(path_to_ident(name));
        if canon_id != pat.id {
            let tv_id = lookup_local(pcx.fcx, pat.span, canon_id);
            let ct = ty::mk_var(tcx, tv_id);
            demand::suptype(pcx.fcx, pat.span, ct, typ);
        }
        fcx.write_ty(pat.id, typ);
        alt sub {
          some(p) { check_pat(pcx, p, expected); }
          _ {}
        }
      }
      ast::pat_ident(path, c) {
        check_pat_variant(pcx, pat, path, some([]), expected);
      }
      ast::pat_enum(path, subpats) {
        check_pat_variant(pcx, pat, path, subpats, expected);
      }
      ast::pat_rec(fields, etc) {
        let ex_fields = alt structure_of(pcx.fcx, pat.span, expected) {
          ty::ty_rec(fields) { fields }
          _ {
            tcx.sess.span_fatal
                (pat.span,
                #fmt["mismatched types: expected `%s` but found record",
                     fcx.ty_to_str(expected)]);
          }
        };
        let f_count = vec::len(fields);
        let ex_f_count = vec::len(ex_fields);
        if ex_f_count < f_count || !etc && ex_f_count > f_count {
            tcx.sess.span_fatal
                (pat.span, #fmt["mismatched types: expected a record \
                      with %u fields, found one with %u \
                      fields",
                                ex_f_count, f_count]);
        }
        fn matches(name: str, f: ty::field) -> bool {
            ret str::eq(name, f.ident);
        }
        for fields.each {|f|
            alt vec::find(ex_fields, bind matches(f.ident, _)) {
              some(field) {
                check_pat(pcx, f.pat, field.mt.ty);
              }
              none {
                tcx.sess.span_fatal(pat.span,
                                    #fmt["mismatched types: did not \
                                          expect a record with a field `%s`",
                                         f.ident]);
              }
            }
        }
        fcx.write_ty(pat.id, expected);
      }
      ast::pat_tup(elts) {
        let ex_elts = alt structure_of(pcx.fcx, pat.span, expected) {
          ty::ty_tup(elts) { elts }
          _ {
            tcx.sess.span_fatal
                (pat.span,
                 #fmt["mismatched types: expected `%s`, found tuple",
                      fcx.ty_to_str(expected)]);
          }
        };
        let e_count = vec::len(elts);
        if e_count != vec::len(ex_elts) {
            tcx.sess.span_fatal
                (pat.span, #fmt["mismatched types: expected a tuple \
                      with %u fields, found one with %u \
                      fields", vec::len(ex_elts), e_count]);
        }
        let mut i = 0u;
        for elts.each {|elt|
            check_pat(pcx, elt, ex_elts[i]);
            i += 1u;
        }

        fcx.write_ty(pat.id, expected);
      }
      ast::pat_box(inner) {
        alt structure_of(pcx.fcx, pat.span, expected) {
          ty::ty_box(e_inner) {
            check_pat(pcx, inner, e_inner.ty);
            fcx.write_ty(pat.id, expected);
          }
          _ {
            tcx.sess.span_fatal(
                pat.span,
                "mismatched types: expected `" +
                pcx.fcx.ty_to_str(expected) +
                "` found box");
          }
        }
      }
      ast::pat_uniq(inner) {
        alt structure_of(pcx.fcx, pat.span, expected) {
          ty::ty_uniq(e_inner) {
            check_pat(pcx, inner, e_inner.ty);
            fcx.write_ty(pat.id, expected);
          }
          _ {
            tcx.sess.span_fatal(
                pat.span,
                "mismatched types: expected `" +
                pcx.fcx.ty_to_str(expected) +
                "` found uniq");
          }
        }
      }
    }
}

fn require_unsafe(sess: session, f_purity: ast::purity, sp: span) {
    alt f_purity {
      ast::unsafe_fn { ret; }
      _ {
        sess.span_err(
            sp,
            "unsafe operation requires unsafe function or block");
      }
    }
}

fn require_impure(sess: session, f_purity: ast::purity, sp: span) {
    alt f_purity {
      ast::unsafe_fn { ret; }
      ast::impure_fn | ast::crust_fn { ret; }
      ast::pure_fn {
        sess.span_err(sp, "found impure expression in pure function decl");
      }
    }
}

fn require_pure_call(ccx: @crate_ctxt, caller_purity: ast::purity,
                     callee: @ast::expr, sp: span) {
    if caller_purity == ast::unsafe_fn { ret; }
    let callee_purity = alt ccx.tcx.def_map.find(callee.id) {
      some(ast::def_fn(_, p)) { p }
      some(ast::def_variant(_, _)) { ast::pure_fn }
      _ {
        alt ccx.method_map.find(callee.id) {
          some(method_static(did)) {
            if did.crate == ast::local_crate {
                alt ccx.tcx.items.get(did.node) {
                  ast_map::node_method(m, _, _) { m.decl.purity }
                  _ { ccx.tcx.sess.span_bug(sp,
                             "Node not bound to a method") }
                }
            } else {
                csearch::lookup_method_purity(ccx.tcx.sess.cstore, did)
            }
          }
          some(method_param(iid, n_m, _, _)) | some(method_iface(iid, n_m)) {
            ty::iface_methods(ccx.tcx, iid)[n_m].purity
          }
          none { ast::impure_fn }
        }
      }
    };
    alt (caller_purity, callee_purity) {
      (ast::impure_fn, ast::unsafe_fn) | (ast::crust_fn, ast::unsafe_fn) {
        ccx.tcx.sess.span_err(sp, "safe function calls function marked \
                                   unsafe");
      }
      (ast::pure_fn, ast::unsafe_fn) | (ast::pure_fn, ast::impure_fn) {
        ccx.tcx.sess.span_err(sp, "pure function calls function not \
                                   known to be pure");
      }
      _ {}
    }
}

fn check_expr(fcx: @fn_ctxt, expr: @ast::expr) -> bool {
    ret check_expr_with_unifier(fcx, expr, ty::mk_nil(fcx.ccx.tcx)) {||
        /* do not take any action on unify */
    };
}

fn check_expr_with(fcx: @fn_ctxt, expr: @ast::expr, expected: ty::t) -> bool {
    ret check_expr_with_unifier(fcx, expr, expected) {||
        demand::suptype(fcx, expr.span, expected, fcx.expr_ty(expr));
    };
}

// determine the `self` type, using fresh variables for all variables declared
// on the impl declaration e.g., `impl<A,B> for [(A,B)]` would return ($0, $1)
// where $0 and $1 are freshly instantiated type variables.
fn impl_self_ty(fcx: @fn_ctxt, did: ast::def_id) -> ty_param_substs_and_ty {
    let tcx = fcx.ccx.tcx;

    let {n_tps, raw_ty} = if did.crate == ast::local_crate {
        alt check tcx.items.get(did.node) {
          ast_map::node_item(@{node: ast::item_impl(ts, _, st, _),
                               _}, _) {
            {n_tps: vec::len(ts),
             raw_ty: fcx.to_ty(st)}
          }
        }
    } else {
        let ity = ty::lookup_item_type(tcx, did);
        {n_tps: vec::len(*ity.bounds),
         raw_ty: ity.ty}
    };

    let self_r = none; // NDM iface/impl regions
    let tps = fcx.next_ty_vars(n_tps);
    let substs = {self_r: self_r, tps: tps};
    let substd_ty = ty::subst(tcx, substs, raw_ty);
    {substs: substs, ty: substd_ty}
}

type self_subst = {selfty: ty::t,
                   fcx: @fn_ctxt,
                   sp: span};

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
          ty::ty_iface(did, tps) {
            self.method_from_iface(did, tps)
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
            let (iid, bound_tps) = alt bound {
              ty::bound_copy | ty::bound_send { cont; /* ok */ }
              ty::bound_iface(bound_t) {
                alt check ty::get(bound_t).struct {
                  ty::ty_iface(i, tps) { (i, tps) }
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
                let bound_substs = { // NDM iface/impl regions
                    self_r: none, tps: bound_tps
                };
                ret some(self.write_mty_from_m(
                    some(self.self_ty), bound_substs, ifce_methods[pos],
                    method_param(iid, pos, n, iface_bnd_idx)));
              }
            }
        }
        ret none;
    }

    fn method_from_iface(
        did: ast::def_id, iface_tps: [ty::t]) -> option<method_origin> {

        let ms = *ty::iface_methods(self.tcx(), did);
        for ms.eachi {|i, m|
            if m.ident != self.m_name { cont; }

            let m_fty = ty::mk_fn(self.tcx(), m.fty);

            if ty::type_has_vars(m_fty) {
                self.tcx().sess.span_fatal(
                    self.expr.span,
                    "can not call a method that contains a \
                     self type through a boxed iface");
            }

            if (*m.tps).len() > 0u {
                self.tcx().sess.span_fatal(
                    self.expr.span,
                    "can not call a generic method through a \
                     boxed iface");
            }

            let iface_substs = { // NDM iface/impl regions
                self_r: none, tps: iface_tps
            };

            ret some(self.write_mty_from_m(
                none, iface_substs, m,
                method_iface(did, i)));
        }

        ret none;
    }

    fn method_from_class(did: ast::def_id, class_substs: ty::substs)
        -> option<method_origin> {

        let ms = *ty::iface_methods(self.tcx(), did);

        for ms.each {|m|
            if m.ident != self.m_name { cont; }

            if m.privacy == ast::priv && !self.include_private {
                self.tcx().sess.span_fatal(
                    self.expr.span,
                    "Call to private method not allowed outside \
                     its defining class");
            }

            // look up method named <name>.
            let m_declared = ty::lookup_class_method_by_name(
                self.tcx(), did, self.m_name, self.expr.span);

            ret some(self.write_mty_from_m(
                none, class_substs, m,
                method_static(m_declared)));
        }

        ret none;
    }

    fn ty_from_did(did: ast::def_id) -> ty::t {
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
                    alt unify::unify(self.fcx, self_ty, ty) {
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
                }

                let (self_substs, n_tps, did) = results[0];
                let fty = self.ty_from_did(did);
                ret some(self.write_mty_from_fty(
                    none, self_substs, n_tps, fty,
                    method_static(did)));
            }
        }

        ret none;
    }

    fn write_mty_from_m(self_ty_sub: option<ty::t>,
                        self_substs: ty::substs,
                        m: ty::method,
                        origin: method_origin) -> method_origin {
        let tcx = self.fcx.ccx.tcx;

        // a bit hokey, but the method unbound has a bare protocol, whereas
        // a.b has a protocol like fn@() (perhaps eventually fn&()):
        let fty = ty::mk_fn(tcx, {proto: ast::proto_box with m.fty});

        ret self.write_mty_from_fty(self_ty_sub, self_substs,
                                    (*m.tps).len(), fty, origin);
    }

    fn write_mty_from_fty(self_ty_sub: option<ty::t>,
                          self_substs: ty::substs,
                          n_tps_m: uint,
                          fty: ty::t,
                          origin: method_origin) -> method_origin {

        let tcx = self.fcx.ccx.tcx;
        let has_self = ty::type_has_vars(fty);

        // Here I will use the "c_" prefix to refer to the method's
        // owner.  You can read it as class, but it may also be an iface.

        let n_tps_supplied = self.supplied_tps.len();
        let m_substs = {
            if n_tps_supplied == 0u {
                self.fcx.next_ty_vars(n_tps_m)
            } else if n_tps_m == 0u {
                tcx.sess.span_err(
                    self.expr.span,
                    "this method does not take type parameters");
                self.fcx.next_ty_vars(n_tps_m)
            } else if n_tps_supplied != n_tps_m {
                tcx.sess.span_err(
                    self.expr.span,
                    "incorrect number of type \
                     parameters given for this method");
                self.fcx.next_ty_vars(n_tps_m)
            } else {
                self.supplied_tps
            }
        };

        let all_substs = {self_r: self_substs.self_r,
                          tps: self_substs.tps + m_substs};
        self.fcx.write_ty_substs(self.node_id, fty, all_substs);

        // n.b. This treatment of self is risky but ok.  As a rule of thumb,
        // one ought to substitute all type parameters at once, and we are not
        // doing so here.  The danger you open up has to do with the
        // possibility that one of the substs in `all_substs` maps to a self
        // type.  However, right now I think it is safe because the types in
        // `all_substs` may not refer to self.  This may not stay true
        // forever, though.

        if has_self && !option::is_none(self_ty_sub) {
            let fty = self.fcx.node_ty(self.node_id);
            let fty = fixup_self_param(
                self.fcx, fty, all_substs.tps, self_ty_sub.get(),
                self.expr.span);
            self.fcx.write_ty(self.node_id, fty);
        }

        ret origin;
    }
}

// Only for fields! Returns <none> for methods>
// FIXME: privacy flags
fn lookup_field_ty(tcx: ty::ctxt, class_id: ast::def_id,
                   items:[ty::field_ty], fieldname: ast::ident,
                   substs: ty::substs) -> option<ty::t> {

    let o_field = vec::find(items, {|f| f.ident == fieldname});
    option::map(o_field) {|f|
        ty::lookup_field_type(tcx, class_id, f.id, substs)
    }
}

/* Returns the region that &expr should be placed into.  If expr is an
 * lvalue, this will be the region of the lvalue.  Otherwise, if region is
 * an rvalue, the semantics are that the result is stored into a temporary
 * stack position and so the resulting region will be the enclosing block.
 */
fn region_of(fcx: @fn_ctxt, expr: @ast::expr) -> ty::region {
    fn borrow(fcx: @fn_ctxt, expr: @ast::expr) -> ty::region {
        let parent_id = fcx.ccx.tcx.region_map.parents.get(expr.id);
        ret ty::re_scope(parent_id);
    }

    fn deref(fcx: @fn_ctxt, base: @ast::expr) -> ty::region {
        let base_ty = fcx.expr_ty(base);
        let base_ty = structurally_resolved_type(fcx, base.span, base_ty);
        alt ty::get(base_ty).struct {
          ty::ty_rptr(region, _) { region }
          ty::ty_box(_) | ty::ty_uniq(_) { borrow(fcx, base) }
          _ { region_of(fcx, base) }
        }
    }

    alt expr.node {
      ast::expr_path(path) {
        let defn = lookup_def(fcx, path.span, expr.id);
        alt defn {
          ast::def_local(local_id, _) |
          ast::def_upvar(local_id, _, _) {
            let local_blocks = fcx.ccx.tcx.region_map.local_blocks;
            let local_block_id = local_blocks.get(local_id);
            ty::re_scope(local_block_id)
          }
          _ {
            ty::re_static
          }
        }
      }
      ast::expr_field(base, _, _) { deref(fcx, base) }
      ast::expr_index(base, _) { deref(fcx, base) }
      ast::expr_unary(ast::deref, base) { deref(fcx, base) }
      _ { borrow(fcx, expr) }
    }
}

fn check_expr_fn_with_unifier(fcx: @fn_ctxt,
                              expr: @ast::expr,
                              proto: ast::proto,
                              decl: ast::fn_decl,
                              body: ast::blk,
                              is_loop_body: bool,
                              unifier: fn()) {
    let tcx = fcx.ccx.tcx;
    let fty = ty::mk_fn(tcx, ty_of_fn_decl(fcx, fcx, proto, decl));

    #debug("check_expr_fn_with_unifier %s fty=%s",
           expr_to_str(expr), fcx.ty_to_str(fty));

    fcx.write_ty(expr.id, fty);

    // Unify the type of the function with the expected type before we
    // typecheck the body so that we have more information about the
    // argument types in the body. This is needed to make binops and
    // record projection work on type inferred arguments.
    unifier();

    let ret_ty = ty::ty_fn_ret(fty);
    let arg_tys = vec::map(ty::ty_fn_args(fty)) {|a| a.ty };

    check_fn(fcx.ccx, proto, decl, body, expr.id,
             ret_ty, arg_tys, is_loop_body, some(fcx),
             fcx.self_ty);
}

fn check_expr_with_unifier(fcx: @fn_ctxt,
                           expr: @ast::expr,
                           expected: ty::t,
                           unifier: fn()) -> bool {

    #debug(">> typechecking expr %d (%s)",
           expr.id, syntax::print::pprust::expr_to_str(expr));

    // A generic function to factor out common logic from call and bind
    // expressions.
    fn check_call_or_bind(
        fcx: @fn_ctxt, sp: span, fty: ty::t,
        args: [option<@ast::expr>]) -> {fty: ty::t, bot: bool} {

        let fty = universally_quantify_before_call(fcx, sp, fty);
        #debug["check_call_or_bind: after universal quant., fty=%s",
               fcx.ty_to_str(fty)];

        let supplied_arg_count = vec::len(args);

        // Grab the argument types
        let arg_tys = alt structure_of(fcx, sp, fty) {
          ty::ty_fn({inputs: arg_tys, output: ret_ty, _}) {
            let expected_arg_count = vec::len(arg_tys);
            if expected_arg_count == supplied_arg_count {
                arg_tys.map { |a| a.ty }
            } else {
                fcx.ccx.tcx.sess.span_err(
                    sp, #fmt["this function takes %u parameter%s but %u \
                              parameter%s supplied", expected_arg_count,
                             if expected_arg_count == 1u {
                                 ""
                             } else {
                                 "s"
                             },
                             supplied_arg_count,
                             if supplied_arg_count == 1u {
                                 " was"
                             } else {
                                 "s were"
                             }]);
                fcx.next_ty_vars(supplied_arg_count)
            }
          }

          _ {
            // I would like to make this span_err, but it's really hard due to
            // the way that expr_bind() is written.
            fcx.ccx.tcx.sess.span_fatal(sp, "mismatched types: \
                                             expected function or native \
                                             function but found "
                                        + fcx.ty_to_str(fty));
          }
        };

        // Check the arguments.
        // We do this in a pretty awful way: first we typecheck any arguments
        // that are not anonymous functions, then we typecheck the anonymous
        // functions. This is so that we have more information about the types
        // of arguments when we typecheck the functions. This isn't really the
        // right way to do this.
        let check_args = fn@(check_blocks: bool) -> bool {
            let mut i = 0u;
            let mut bot = false;
            for args.each {|a_opt|
                alt a_opt {
                  some(a) {
                    let is_block = alt a.node {
                      ast::expr_fn_block(_, _) { true }
                      _ { false }
                    };
                    if is_block == check_blocks {
                        let arg_ty = arg_tys[i];
                        bot |= check_expr_with_unifier(fcx, a, arg_ty) {||
                            demand::assign(fcx, a.span, arg_ty, a);
                        };
                    }
                  }
                  none { }
                }
                i += 1u;
            }
            ret bot;
        };

        let bot = check_args(false) | check_args(true);

        {fty: fty, bot: bot}
    }

    // A generic function for checking assignment expressions
    fn check_assignment(fcx: @fn_ctxt, _sp: span, lhs: @ast::expr,
                        rhs: @ast::expr, id: ast::node_id) -> bool {
        let mut bot = check_expr(fcx, lhs);
        bot |= check_expr_with(fcx, rhs, fcx.expr_ty(lhs));
        fcx.write_ty(id, ty::mk_nil(fcx.ccx.tcx));
        ret bot;
    }

    // A generic function for doing all of the checking for call expressions
    fn check_call(fcx: @fn_ctxt, sp: span, call_expr_id: ast::node_id,
                  f: @ast::expr, args: [@ast::expr]) -> bool {

        let mut bot = check_expr(fcx, f);
        let fn_ty = fcx.expr_ty(f);

        // Call the generic checker.
        let fty = {
            let args_opt = args.map { |arg| some(arg) };
            let r = check_call_or_bind(fcx, sp, fn_ty, args_opt);
            bot |= r.bot;
            r.fty
        };

        // Need to restrict oper to being an explicit expr_path if we're
        // inside a pure function
        require_pure_call(fcx.ccx, fcx.purity, f, sp);

        // Pull the return type out of the type of the function.
        alt structure_of(fcx, sp, fty) {
          ty::ty_fn(f) {
            bot |= (f.ret_style == ast::noreturn);
            fcx.write_ty(call_expr_id, f.output);
            ret bot;
          }
          _ { fcx.ccx.tcx.sess.span_fatal(sp, "calling non-function"); }
        }
    }

    // A generic function for checking for or for-each loops
    fn check_for(fcx: @fn_ctxt, local: @ast::local,
                 element_ty: ty::t, body: ast::blk,
                 node_id: ast::node_id) -> bool {
        let locid = lookup_local(fcx, local.span, local.node.id);
        demand::suptype(fcx, local.span,
                       ty::mk_var(fcx.ccx.tcx, locid),
                       element_ty);
        let bot = check_decl_local(fcx, local);
        check_block_no_value(fcx, body);
        fcx.write_nil(node_id);
        ret bot;
    }

    // A generic function for checking the then and else in an if
    // or if-check
    fn check_then_else(fcx: @fn_ctxt, thn: ast::blk,
                       elsopt: option<@ast::expr>, id: ast::node_id,
                       _sp: span) -> bool {
        let (if_t, if_bot) =
            alt elsopt {
              some(els) {
                let if_t = fcx.next_ty_var();
                let thn_bot = check_block(fcx, thn);
                let thn_t = fcx.node_ty(thn.node.id);
                demand::suptype(fcx, thn.span, if_t, thn_t);
                let els_bot = check_expr_with(fcx, els, if_t);
                (if_t, thn_bot & els_bot)
              }
              none {
                check_block_no_value(fcx, thn);
                (ty::mk_nil(fcx.ccx.tcx), false)
              }
            };
        fcx.write_ty(id, if_t);
        ret if_bot;
    }

    fn binop_method(op: ast::binop) -> option<str> {
        alt op {
          ast::add | ast::subtract | ast::mul | ast::div | ast::rem |
          ast::bitxor | ast::bitand | ast::bitor | ast::lsl | ast::lsr |
          ast::asr { some(ast_util::binop_to_str(op)) }
          _ { none }
        }
    }
    fn lookup_op_method(fcx: @fn_ctxt, op_ex: @ast::expr, self_t: ty::t,
                        opname: str, args: [option<@ast::expr>])
        -> option<(ty::t, bool)> {
        let callee_id = ast_util::op_expr_callee_id(op_ex);
        let lkup = lookup({fcx: fcx,
                           expr: op_ex,
                           node_id: callee_id,
                           m_name: opname,
                           self_ty: self_t,
                           supplied_tps: [],
                           include_private: false});
        alt lkup.method() {
          some(origin) {
            let {fty: method_ty, bot: bot} = {
                let method_ty = fcx.node_ty(callee_id);
                check_call_or_bind(fcx, op_ex.span, method_ty, args)
            };
            fcx.ccx.method_map.insert(op_ex.id, origin);
            some((ty::ty_fn_ret(method_ty), bot))
          }
          _ { none }
        }
    }
    // could be either a expr_binop or an expr_assign_binop
    fn check_binop(fcx: @fn_ctxt, expr: @ast::expr,
                   op: ast::binop,
                   lhs: @ast::expr,
                   rhs: @ast::expr) -> bool {
        let tcx = fcx.ccx.tcx;
        let lhs_bot = check_expr(fcx, lhs);
        let lhs_t = fcx.expr_ty(lhs);
        let lhs_t = structurally_resolved_type(fcx, lhs.span, lhs_t);
        ret alt (op, ty::get(lhs_t).struct) {
          (ast::add, ty::ty_vec(lhs_mt)) {
            // For adding vectors with type L=[ML TL] and R=[MR TR], the the
            // result [ML T] where TL <: T and TR <: T.  In other words, the
            // result type is (generally) the LUB of (TL, TR) and takes the
            // mutability from the LHS.
            let t_var = fcx.next_ty_var();
            let const_vec_t = ty::mk_vec(tcx, {ty: t_var,
                                               mutbl: ast::m_const});
            demand::suptype(fcx, lhs.span, const_vec_t, lhs_t);
            let rhs_bot = check_expr_with(fcx, rhs, const_vec_t);
            let result_vec_t = ty::mk_vec(tcx, {ty: t_var,
                                                mutbl: lhs_mt.mutbl});
            fcx.write_ty(expr.id, result_vec_t);
            lhs_bot | rhs_bot
          }

          (_, _) if ty::type_is_integral(lhs_t) &&
          ast_util::is_shift_binop(op) {
            // Shift is a special case: rhs can be any integral type
            let rhs_bot = check_expr(fcx, rhs);
            let rhs_t = fcx.expr_ty(rhs);
            require_integral(fcx, rhs.span, rhs_t);
            fcx.write_ty(expr.id, lhs_t);
            lhs_bot | rhs_bot
          }

          (_, _) if ty::is_binopable(tcx, lhs_t, op) {
            let tvar = fcx.next_ty_var();
            demand::suptype(fcx, expr.span, tvar, lhs_t);
            let rhs_bot = check_expr_with(fcx, rhs, tvar);
            let rhs_t = alt op {
              ast::eq | ast::lt | ast::le | ast::ne | ast::ge |
              ast::gt {
                // these comparison operators are handled in a
                // separate case below.
                tcx.sess.span_bug(
                    expr.span,
                    #fmt["Comparison operator in expr_binop: %s",
                         ast_util::binop_to_str(op)]);
              }
              _ { lhs_t }
            };
            fcx.write_ty(expr.id, rhs_t);
            if !ast_util::lazy_binop(op) { lhs_bot | rhs_bot }
            else { lhs_bot }
          }

          (_, _) {
            let (result, rhs_bot) =
                check_user_binop(fcx, expr, lhs_t, op, rhs);
            fcx.write_ty(expr.id, result);
            lhs_bot | rhs_bot
          }
        };
    }
    fn check_user_binop(fcx: @fn_ctxt, ex: @ast::expr, lhs_resolved_t: ty::t,
                        op: ast::binop, rhs: @ast::expr) -> (ty::t, bool) {
        let tcx = fcx.ccx.tcx;
        alt binop_method(op) {
          some(name) {
            alt lookup_op_method(fcx, ex, lhs_resolved_t, name, [some(rhs)]) {
              some(pair) { ret pair; }
              _ {}
            }
          }
          _ {}
        }
        check_expr(fcx, rhs);
        tcx.sess.span_err(
            ex.span, "binary operation " + ast_util::binop_to_str(op) +
            " cannot be applied to type `" +
            fcx.ty_to_str(lhs_resolved_t) +
            "`");
        (lhs_resolved_t, false)
    }
    fn check_user_unop(fcx: @fn_ctxt, op_str: str, mname: str,
                       ex: @ast::expr, rhs_t: ty::t) -> ty::t {
        alt lookup_op_method(fcx, ex, rhs_t, mname, []) {
          some((ret_ty, _)) { ret_ty }
          _ {
            fcx.ccx.tcx.sess.span_err(
                ex.span, #fmt["cannot apply unary operator `%s` to type `%s`",
                              op_str, fcx.ty_to_str(rhs_t)]);
            rhs_t
          }
        }
    }

    let tcx = fcx.ccx.tcx;
    let id = expr.id;
    let mut bot = false;
    alt expr.node {
      ast::expr_vstore(ev, vst) {
        let typ = alt ev.node {
          ast::expr_lit(@{node: ast::lit_str(s), span:_}) {
            let tt = ast_expr_vstore_to_vstore(fcx, ev, str::len(s), vst);
            ty::mk_estr(tcx, tt)
          }
          ast::expr_vec(args, mutbl) {
            let tt = ast_expr_vstore_to_vstore(fcx, ev, vec::len(args), vst);
            let t: ty::t = fcx.next_ty_var();
            for args.each {|e| bot |= check_expr_with(fcx, e, t); }
            ty::mk_evec(tcx, {ty: t, mutbl: mutbl}, tt)
          }
          _ {
            tcx.sess.span_bug(expr.span, "vstore modifier on non-sequence")
          }
        };
        fcx.write_ty(ev.id, typ);
        fcx.write_ty(id, typ);
      }

      ast::expr_lit(lit) {
        let typ = check_lit(fcx.ccx, lit);
        fcx.write_ty(id, typ);
      }

      // Something of a hack: special rules for comparison operators that
      // simply unify LHS and RHS.  This helps with inference as LHS and RHS
      // do not need to be "resolvable".  Some tests, particularly those with
      // complicated iface requirements, fail without this---I think this code
      // can be removed if we improve iface resolution to be more eager when
      // possible.
      ast::expr_binary(ast::eq, lhs, rhs) |
      ast::expr_binary(ast::ne, lhs, rhs) |
      ast::expr_binary(ast::lt, lhs, rhs) |
      ast::expr_binary(ast::le, lhs, rhs) |
      ast::expr_binary(ast::gt, lhs, rhs) |
      ast::expr_binary(ast::ge, lhs, rhs) {
        let tcx = fcx.ccx.tcx;
        let tvar = fcx.next_ty_var();
        bot |= check_expr_with(fcx, lhs, tvar);
        bot |= check_expr_with(fcx, rhs, tvar);
        fcx.write_ty(id, ty::mk_bool(tcx));
      }
      ast::expr_binary(op, lhs, rhs) {
        bot |= check_binop(fcx, expr, op, lhs, rhs);
      }
      ast::expr_assign_op(op, lhs, rhs) {
        require_impure(tcx.sess, fcx.purity, expr.span);
        bot |= check_binop(fcx, expr, op, lhs, rhs);
        let lhs_t = fcx.expr_ty(lhs);
        let result_t = fcx.expr_ty(expr);
        demand::suptype(fcx, expr.span, result_t, lhs_t);

        // Overwrite result of check_binop...this preserves existing behavior
        // but seems quite dubious with regard to user-defined methods
        // and so forth. - Niko
        fcx.write_nil(expr.id);
      }
      ast::expr_unary(unop, oper) {
        bot = check_expr(fcx, oper);
        let mut oper_t = fcx.expr_ty(oper);
        alt unop {
          ast::box(mutbl) {
            oper_t = ty::mk_box(tcx, {ty: oper_t, mutbl: mutbl});
          }
          ast::uniq(mutbl) {
            oper_t = ty::mk_uniq(tcx, {ty: oper_t, mutbl: mutbl});
          }
          ast::deref {
            alt structure_of(fcx, expr.span, oper_t) {
              ty::ty_box(inner) { oper_t = inner.ty; }
              ty::ty_uniq(inner) { oper_t = inner.ty; }
              ty::ty_res(_, inner, _) { oper_t = inner; }
              ty::ty_enum(id, substs) {
                let variants = ty::enum_variants(tcx, id);
                if vec::len(*variants) != 1u ||
                       vec::len(variants[0].args) != 1u {
                    tcx.sess.span_fatal(expr.span,
                                        "can only dereference enums " +
                                        "with a single variant which has a "
                                            + "single argument");
                }
                oper_t = ty::subst(tcx, substs, variants[0].args[0]);
              }
              ty::ty_ptr(inner) {
                oper_t = inner.ty;
                require_unsafe(tcx.sess, fcx.purity, expr.span);
              }
              ty::ty_rptr(_, inner) { oper_t = inner.ty; }
              _ {
                  tcx.sess.span_err(expr.span,
                      #fmt("Type %s cannot be dereferenced",
                           ty_to_str(tcx, oper_t)));
              }
            }
          }
          ast::not {
            oper_t = structurally_resolved_type(fcx, oper.span, oper_t);
            if !(ty::type_is_integral(oper_t) ||
                 ty::get(oper_t).struct == ty::ty_bool) {
                oper_t = check_user_unop(fcx, "!", "!", expr, oper_t);
            }
          }
          ast::neg {
            oper_t = structurally_resolved_type(fcx, oper.span, oper_t);
            if !(ty::type_is_integral(oper_t) ||
                 ty::type_is_fp(oper_t)) {
                oper_t = check_user_unop(fcx, "-", "unary-", expr, oper_t);
            }
          }
        }
        fcx.write_ty(id, oper_t);
      }
      ast::expr_addr_of(mutbl, oper) {
        bot = check_expr(fcx, oper);
        let mut oper_t = fcx.expr_ty(oper);

        let region = region_of(fcx, oper);
        let tm = { ty: oper_t, mutbl: mutbl };
        oper_t = ty::mk_rptr(tcx, region, tm);
        fcx.write_ty(id, oper_t);
      }
      ast::expr_path(pth) {
        let defn = lookup_def(fcx, pth.span, id);

        let tpt = ty_param_bounds_and_ty_for_def(fcx, expr.span, defn);
        instantiate_path(fcx, pth, tpt, expr.span, expr.id);
      }
      ast::expr_mac(_) { tcx.sess.bug("unexpanded macro"); }
      ast::expr_fail(expr_opt) {
        bot = true;
        alt expr_opt {
          none {/* do nothing */ }
          some(e) { check_expr_with(fcx, e, ty::mk_str(tcx)); }
        }
        fcx.write_bot(id);
      }
      ast::expr_break { fcx.write_bot(id); bot = true; }
      ast::expr_cont { fcx.write_bot(id); bot = true; }
      ast::expr_ret(expr_opt) {
        bot = true;
        let ret_ty = alt fcx.indirect_ret_ty {
          some(t) { t } none { fcx.ret_ty }
        };
        alt expr_opt {
          none {
            if !are_compatible(fcx, ret_ty, ty::mk_nil(tcx)) {
                tcx.sess.span_err(expr.span,
                                  "ret; in function returning non-nil");
            }
          }
          some(e) { check_expr_with(fcx, e, ret_ty); }
        }
        fcx.write_bot(id);
      }
      ast::expr_be(e) {
        if !ast_util::is_call_expr(e) {
           tcx.sess.span_err(expr.span,
              "non-call expression in tail call");
        }
        check_expr_with(fcx, e, fcx.ret_ty);
        bot = true;
        fcx.write_nil(id);
      }
      ast::expr_log(_, lv, e) {
        bot = check_expr_with(fcx, lv, ty::mk_mach_uint(tcx, ast::ty_u32));
        // Note: this does not always execute, so do not propagate bot:
        check_expr(fcx, e);
        fcx.write_nil(id);
      }
      ast::expr_check(_, e) {
        bot = check_pred_expr(fcx, e);
        fcx.write_nil(id);
      }
      ast::expr_if_check(cond, thn, elsopt) {
        bot =
            check_pred_expr(fcx, cond) |
                check_then_else(fcx, thn, elsopt, id, expr.span);
      }
      ast::expr_assert(e) {
        bot = check_expr_with(fcx, e, ty::mk_bool(tcx));
        fcx.write_nil(id);
      }
      ast::expr_copy(a) {
        bot = check_expr_with(fcx, a, expected);
        fcx.write_ty(id, fcx.expr_ty(a));
      }
      ast::expr_move(lhs, rhs) {
        require_impure(tcx.sess, fcx.purity, expr.span);
        bot = check_assignment(fcx, expr.span, lhs, rhs, id);
      }
      ast::expr_assign(lhs, rhs) {
        require_impure(tcx.sess, fcx.purity, expr.span);
        bot = check_assignment(fcx, expr.span, lhs, rhs, id);
      }
      ast::expr_swap(lhs, rhs) {
        require_impure(tcx.sess, fcx.purity, expr.span);
        bot = check_assignment(fcx, expr.span, lhs, rhs, id);
      }
      ast::expr_if(cond, thn, elsopt) {
        bot =
            check_expr_with(fcx, cond, ty::mk_bool(tcx)) |
                check_then_else(fcx, thn, elsopt, id, expr.span);
      }
      ast::expr_while(cond, body) {
        bot = check_expr_with(fcx, cond, ty::mk_bool(tcx));
        check_block_no_value(fcx, body);
        fcx.write_ty(id, ty::mk_nil(tcx));
      }
      ast::expr_do_while(body, cond) {
        bot = check_expr_with(fcx, cond, ty::mk_bool(tcx)) |
              check_block_no_value(fcx, body);
        fcx.write_ty(id, fcx.node_ty(body.node.id));
      }
      ast::expr_loop(body) {
          check_block_no_value(fcx, body);
          fcx.write_ty(id, ty::mk_nil(tcx));
          bot = !may_break(body);
      }
      ast::expr_alt(discrim, arms, _) {
        let pattern_ty = fcx.next_ty_var();
        bot = check_expr_with(fcx, discrim, pattern_ty);

        // Typecheck the patterns first, so that we get types for all the
        // bindings.
        //let pattern_ty = fcx.expr_ty(discrim);
        for arms.each {|arm|
            let pcx = {
                fcx: fcx,
                map: pat_util::pat_id_map(tcx.def_map, arm.pats[0]),
                alt_region: ty::re_scope(expr.id),
                block_region: ty::re_scope(arm.body.node.id),
                pat_region: ty::re_scope(expr.id)
            };

            for arm.pats.each {|p| check_pat(pcx, p, pattern_ty);}
        }
        // Now typecheck the blocks.
        let mut result_ty = fcx.next_ty_var();
        let mut arm_non_bot = false;
        for arms.each {|arm|
            alt arm.guard {
              some(e) { check_expr_with(fcx, e, ty::mk_bool(tcx)); }
              none { }
            }
            if !check_block(fcx, arm.body) { arm_non_bot = true; }
            let bty = fcx.node_ty(arm.body.node.id);
            demand::suptype(fcx, arm.body.span, result_ty, bty);
        }
        bot |= !arm_non_bot;
        if !arm_non_bot { result_ty = ty::mk_bot(tcx); }
        fcx.write_ty(id, result_ty);
      }
      ast::expr_fn(proto, decl, body, captures) {
        check_expr_fn_with_unifier(fcx, expr, proto, decl, body,
                                   false, unifier);
        capture::check_capture_clause(tcx, expr.id, proto, *captures);
      }
      ast::expr_fn_block(decl, body) {
        // Take the prototype from the expected type, but default to block:
        let proto = alt ty::get(expected).struct {
          ty::ty_fn({proto, _}) { proto }
          _ { ast::proto_box }
        };
        #debug("checking expr_fn_block %s expected=%s",
               expr_to_str(expr),
               ty_to_str(tcx, expected));
        check_expr_fn_with_unifier(fcx, expr, proto, decl, body,
                                   false, unifier);
      }
      ast::expr_loop_body(b) {
        let rty = structurally_resolved_type(fcx, expr.span, expected);
        let (inner_ty, proto) = alt check ty::get(rty).struct {
          ty::ty_fn(fty) {
            alt infer::mk_subty(fcx.infcx, fty.output, ty::mk_bool(tcx)) {
              result::ok(_) {}
              result::err(err) {
                tcx.sess.span_fatal(
                    expr.span, #fmt("a loop function's last argument should \
                                     return `bool`, not `%s`",
                                    ty_to_str(tcx, fty.output)));
              }
            }
            (ty::mk_fn(tcx, {output: ty::mk_nil(tcx) with fty}), fty.proto)
          }
          _ {
            tcx.sess.span_fatal(expr.span, "a loop function's last argument \
                                            should be of function type");
          }
        };
        alt check b.node {
          ast::expr_fn_block(decl, body) {
            check_expr_fn_with_unifier(fcx, b, proto, decl, body, true) {||
                demand::suptype(fcx, b.span, inner_ty, fcx.expr_ty(b));
            }
          }
        }
        let block_ty = structurally_resolved_type(
            fcx, expr.span, fcx.node_ty(b.id));
        alt check ty::get(block_ty).struct {
          ty::ty_fn(fty) {
            fcx.write_ty(expr.id, ty::mk_fn(tcx, {output: ty::mk_bool(tcx)
                                                  with fty}));
          }
        }
      }
      ast::expr_block(b) {
        // If this is an unchecked block, turn off purity-checking
        bot = check_block(fcx, b);
        let typ =
            alt b.node.expr {
              some(expr) { fcx.expr_ty(expr) }
              none { ty::mk_nil(tcx) }
            };
        fcx.write_ty(id, typ);
      }
      ast::expr_bind(f, args) {
        // Call the generic checker.
        bot = check_expr(fcx, f);

        let {fty, bot: ccob_bot} = {
            let fn_ty = fcx.expr_ty(f);
            check_call_or_bind(fcx, expr.span, fn_ty, args)
        };
        bot |= ccob_bot;

        // TODO: Perform substitutions on the return type.

        // Pull the argument and return types out.
        let mut proto, arg_tys, rt, cf, constrs;
        alt structure_of(fcx, expr.span, fty) {
          // FIXME:
          // probably need to munge the constrs to drop constraints
          // for any bound args
          ty::ty_fn(f) {
            proto = f.proto;
            arg_tys = f.inputs;
            rt = f.output;
            cf = f.ret_style;
            constrs = f.constraints;
          }
          _ { fail "LHS of bind expr didn't have a function type?!"; }
        }

        let proto = alt proto {
          ast::proto_bare | ast::proto_box | ast::proto_uniq {
            ast::proto_box
          }
          ast::proto_any | ast::proto_block {
            tcx.sess.span_err(expr.span,
                              #fmt["cannot bind %s closures",
                                   proto_to_str(proto)]);
            proto // dummy value so compilation can proceed
          }
        };

        // For each blank argument, add the type of that argument
        // to the resulting function type.
        let mut out_args = [];
        let mut i = 0u;
        while i < vec::len(args) {
            alt args[i] {
              some(_) {/* no-op */ }
              none { out_args += [arg_tys[i]]; }
            }
            i += 1u;
        }

        let ft = ty::mk_fn(tcx, {proto: proto,
                                 inputs: out_args, output: rt,
                                 ret_style: cf, constraints: constrs});
        fcx.write_ty(id, ft);
      }
      ast::expr_call(f, args, _) {
        bot = check_call(fcx, expr.span, expr.id, f, args);
      }
      ast::expr_cast(e, t) {
        bot = check_expr(fcx, e);
        let t_1 = fcx.to_ty(t);
        let t_e = fcx.expr_ty(e);

        alt ty::get(t_1).struct {
          // This will be looked up later on
          ty::ty_iface(_, _) {}
          _ {
            if ty::type_is_nil(t_e) {
                tcx.sess.span_err(expr.span, "cast from nil: " +
                                  ty_to_str(tcx, t_e) + " as " +
                                  ty_to_str(tcx, t_1));
            } else if ty::type_is_nil(t_1) {
                tcx.sess.span_err(expr.span, "cast to nil: " +
                                  ty_to_str(tcx, t_e) + " as " +
                                  ty_to_str(tcx, t_1));
            }

            let t_1_is_scalar = type_is_scalar(fcx, expr.span, t_1);
            if type_is_c_like_enum(fcx,expr.span,t_e) && t_1_is_scalar {
                /* this case is allowed */
            } else if !(type_is_scalar(fcx,expr.span,t_e) && t_1_is_scalar) {
                // FIXME there are more forms of cast to support, eventually.
                tcx.sess.span_err(expr.span,
                                  "non-scalar cast: " +
                                  ty_to_str(tcx, t_e) + " as " +
                                  ty_to_str(tcx, t_1));
            }
          }
        }
        fcx.write_ty(id, t_1);
      }
      ast::expr_vec(args, mutbl) {
        let t: ty::t = fcx.next_ty_var();
        for args.each {|e| bot |= check_expr_with(fcx, e, t); }
        let typ = ty::mk_vec(tcx, {ty: t, mutbl: mutbl});
        fcx.write_ty(id, typ);
      }
      ast::expr_tup(elts) {
        let mut elt_ts = [];
        vec::reserve(elt_ts, vec::len(elts));
        for elts.each {|e|
            check_expr(fcx, e);
            let ety = fcx.expr_ty(e);
            elt_ts += [ety];
        }
        let typ = ty::mk_tup(tcx, elt_ts);
        fcx.write_ty(id, typ);
      }
      ast::expr_rec(fields, base) {
        option::iter(base) {|b| check_expr(fcx, b); }
        let fields_t = vec::map(fields, {|f|
            bot |= check_expr(fcx, f.node.expr);
            let expr_t = fcx.expr_ty(f.node.expr);
            let expr_mt = {ty: expr_t, mutbl: f.node.mutbl};
            // for the most precise error message,
            // should be f.node.expr.span, not f.span
            respan(f.node.expr.span, {ident: f.node.ident, mt: expr_mt})
        });
        alt base {
          none {
            fn get_node(f: spanned<field>) -> field { f.node }
            let typ = ty::mk_rec(tcx, vec::map(fields_t, get_node));
            fcx.write_ty(id, typ);
          }
          some(bexpr) {
            bot |= check_expr(fcx, bexpr);
            let bexpr_t = fcx.expr_ty(bexpr);
            let mut base_fields: [field] = [];
            alt structure_of(fcx, expr.span, bexpr_t) {
              ty::ty_rec(flds) { base_fields = flds; }
              _ {
                tcx.sess.span_fatal(expr.span,
                                    "record update has non-record base");
              }
            }
            fcx.write_ty(id, bexpr_t);
            for fields_t.each {|f|
                let mut found = false;
                for base_fields.each {|bf|
                    if str::eq(f.node.ident, bf.ident) {
                        demand::suptype(fcx, f.span, bf.mt.ty, f.node.mt.ty);
                        found = true;
                    }
                }
                if !found {
                    tcx.sess.span_fatal(f.span,
                                        "unknown field in record update: " +
                                            f.node.ident);
                }
            }
          }
        }
      }
      ast::expr_field(base, field, tys) {
        bot |= check_expr(fcx, base);
        let expr_t = structurally_resolved_type(fcx, expr.span,
                                                fcx.expr_ty(base));
        let base_t = do_autoderef(fcx, expr.span, expr_t);
        let mut handled = false;
        let n_tys = vec::len(tys);
        alt structure_of(fcx, expr.span, base_t) {
          ty::ty_rec(fields) {
            alt ty::field_idx(field, fields) {
              some(ix) {
                if n_tys > 0u {
                    tcx.sess.span_err(expr.span,
                                      "can't provide type parameters \
                                       to a field access");
                }
                fcx.write_ty(id, fields[ix].mt.ty);
                handled = true;
              }
              _ {}
            }
          }
          ty::ty_class(base_id, substs) {
              // This is just for fields -- the same code handles
              // methods in both classes and ifaces

              // (1) verify that the class id actually has a field called
              // field
              #debug("class named %s", ty_to_str(tcx, base_t));
              /*
                check whether this is a self-reference or not, which
                determines whether we look at all fields or only public
                ones
               */
              let cls_items = if self_ref(fcx, base.id) {
                  // base expr is "self" -- consider all fields
                  ty::lookup_class_fields(tcx, base_id)
              }
              else {
                  lookup_public_fields(tcx, base_id)
              };
              alt lookup_field_ty(tcx, base_id, cls_items, field, substs) {
                 some(field_ty) {
                    // (2) look up what field's type is, and return it
                    // FIXME: actually instantiate any type params
                     fcx.write_ty(id, field_ty);
                     handled = true;
                 }
                 none {}
              }
          }
          _ {}
        }
        if !handled {
            let tps = vec::map(tys) { |ty| fcx.to_ty(ty) };
            let lkup = lookup({fcx: fcx,
                               expr: expr,
                               node_id: expr.id,
                               m_name: field,
                               self_ty: expr_t,
                               supplied_tps: tps,
                               include_private: self_ref(fcx, base.id)});
            alt lkup.method() {
              some(origin) {
                fcx.ccx.method_map.insert(id, origin);
              }
              none {
                let t_err = resolve_type_vars_if_possible(fcx, expr_t);
                let msg = #fmt["attempted access of field %s on type %s, but \
                          no public field or method with that name was found",
                               field, ty_to_str(tcx, t_err)];
                tcx.sess.span_err(expr.span, msg);
                // NB: Adding a bogus type to allow typechecking to continue
                fcx.write_ty(id, fcx.next_ty_var());
              }
            }
        }
      }
      ast::expr_index(base, idx) {
        bot |= check_expr(fcx, base);
        let raw_base_t = fcx.expr_ty(base);
        let base_t = do_autoderef(fcx, expr.span, raw_base_t);
        bot |= check_expr(fcx, idx);
        let idx_t = fcx.expr_ty(idx);
        alt structure_of(fcx, expr.span, base_t) {
          ty::ty_evec(mt, _) |
          ty::ty_vec(mt) {
            require_integral(fcx, idx.span, idx_t);
            fcx.write_ty(id, mt.ty);
          }
          ty::ty_estr(_) |
          ty::ty_str {
            require_integral(fcx, idx.span, idx_t);
            let typ = ty::mk_mach_uint(tcx, ast::ty_u8);
            fcx.write_ty(id, typ);
          }
          _ {
            let resolved = structurally_resolved_type(fcx, expr.span,
                                                      raw_base_t);
            alt lookup_op_method(fcx, expr, resolved, "[]",
                                 [some(idx)]) {
              some((ret_ty, _)) { fcx.write_ty(id, ret_ty); }
              _ {
                tcx.sess.span_fatal(
                    expr.span, "cannot index a value of type `" +
                    ty_to_str(tcx, base_t) + "`");
              }
            }
          }
        }
      }
      ast::expr_new(p, alloc_id, v) {
        bot |= check_expr(fcx, p);
        bot |= check_expr(fcx, v);

        let p_ty = fcx.expr_ty(p);

        let lkup = lookup({fcx: fcx,
                           expr: p,
                           node_id: alloc_id,
                           m_name: "alloc",
                           self_ty: p_ty,
                           supplied_tps: [],
                           include_private: false});
        alt lkup.method() {
          some(origin) {
            fcx.ccx.method_map.insert(alloc_id, origin);

            // Check that the alloc() method has the expected type, which
            // should be fn(sz: uint, align: uint) -> *().
            let expected_ty = {
                let ty_uint = ty::mk_uint(tcx);
                let ty_nilp = ty::mk_ptr(tcx, {ty: ty::mk_nil(tcx),
                                              mutbl: ast::m_imm});
                let m = ast::expl(ty::default_arg_mode_for_ty(ty_uint));
                ty::mk_fn(tcx, {proto: ast::proto_any,
                                inputs: [{mode: m, ty: ty_uint},
                                         {mode: m, ty: ty_uint}],
                                output: ty_nilp,
                                ret_style: ast::return_val,
                                constraints: []})
            };

            demand::suptype(fcx, expr.span,
                           expected_ty, fcx.node_ty(alloc_id));
          }

          none {
            let t_err = resolve_type_vars_if_possible(fcx, p_ty);
            let msg = #fmt["no `alloc()` method found for type `%s`",
                           ty_to_str(tcx, t_err)];
            tcx.sess.span_err(expr.span, msg);
          }
        }

        // The region value must have a type like &r.T.  The resulting
        // memory will be allocated into the region `r`.
        let pool_region = region_of(fcx, p);
        let v_ty = fcx.expr_ty(v);
        let res_ty = ty::mk_rptr(tcx, pool_region, {ty: v_ty,
                                                    mutbl: ast::m_imm});
        fcx.write_ty(expr.id, res_ty);
      }
    }
    if bot { fcx.write_bot(expr.id); }

    #debug("type of expr %s is %s, expected is %s",
           syntax::print::pprust::expr_to_str(expr),
           ty_to_str(tcx, fcx.expr_ty(expr)),
           ty_to_str(tcx, expected));

    unifier();

    #debug("<< bot=%b", bot);
    ret bot;
}

fn require_integral(fcx: @fn_ctxt, sp: span, t: ty::t) {
    if !type_is_integral(fcx, sp, t) {
        fcx.ccx.tcx.sess.span_err(sp, "mismatched types: expected \
                                       `integer` but found `"
                                  + fcx.ty_to_str(t) + "`");
    }
}

fn check_decl_initializer(fcx: @fn_ctxt, nid: ast::node_id,
                          init: ast::initializer) -> bool {
    let lty = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, init.expr.span, nid));
    ret check_expr_with(fcx, init.expr, lty);
}

fn check_decl_local(fcx: @fn_ctxt, local: @ast::local) -> bool {
    let mut bot = false;

    let t = ty::mk_var(fcx.ccx.tcx, fcx.locals.get(local.node.id));
    fcx.write_ty(local.node.id, t);
    alt local.node.init {
      some(init) {
        bot = check_decl_initializer(fcx, local.node.id, init);
      }
      _ {/* fall through */ }
    }

    let region =
        ty::re_scope(
            fcx.ccx.tcx.region_map.local_blocks.get(local.node.id));
    let pcx = {
        fcx: fcx,
        map: pat_util::pat_id_map(fcx.ccx.tcx.def_map, local.node.pat),
        alt_region: region,
        block_region: region,
        pat_region: region
    };

    check_pat(pcx, local.node.pat, t);
    ret bot;
}

fn check_stmt(fcx: @fn_ctxt, stmt: @ast::stmt) -> bool {
    let mut node_id;
    let mut bot = false;
    alt stmt.node {
      ast::stmt_decl(decl, id) {
        node_id = id;
        alt decl.node {
          ast::decl_local(ls) {
            for ls.each {|l| bot |= check_decl_local(fcx, l); }
          }
          ast::decl_item(_) {/* ignore for now */ }
        }
      }
      ast::stmt_expr(expr, id) {
        node_id = id;
        bot = check_expr_with(fcx, expr, ty::mk_nil(fcx.ccx.tcx));
      }
      ast::stmt_semi(expr, id) {
        node_id = id;
        bot = check_expr(fcx, expr);
      }
    }
    fcx.write_nil(node_id);
    ret bot;
}

fn check_block_no_value(fcx: @fn_ctxt, blk: ast::blk) -> bool {
    let bot = check_block(fcx, blk);
    if !bot {
        let blkty = fcx.node_ty(blk.node.id);
        let nilty = ty::mk_nil(fcx.ccx.tcx);
        demand::suptype(fcx, blk.span, nilty, blkty);
    }
    ret bot;
}

fn check_block(fcx0: @fn_ctxt, blk: ast::blk) -> bool {
    let fcx = alt blk.node.rules {
      ast::unchecked_blk { @{purity: ast::impure_fn with *fcx0} }
      ast::unsafe_blk { @{purity: ast::unsafe_fn with *fcx0} }
      ast::default_blk { fcx0 }
    };
    let mut bot = false;
    let mut warned = false;
    for blk.node.stmts.each {|s|
        if bot && !warned &&
               alt s.node {
                 ast::stmt_decl(@{node: ast::decl_local(_), _}, _) |
                 ast::stmt_expr(_, _) | ast::stmt_semi(_, _) {
                   true
                 }
                 _ { false }
               } {
            fcx.ccx.tcx.sess.span_warn(s.span, "unreachable statement");
            warned = true;
        }
        bot |= check_stmt(fcx, s);
    }
    alt blk.node.expr {
      none { fcx.write_nil(blk.node.id); }
      some(e) {
        if bot && !warned {
            fcx.ccx.tcx.sess.span_warn(e.span, "unreachable expression");
        }
        bot |= check_expr(fcx, e);
        let ety = fcx.expr_ty(e);
        fcx.write_ty(blk.node.id, ety);
      }
    }
    if bot {
        fcx.write_bot(blk.node.id);
    }
    ret bot;
}

fn check_const(ccx: @crate_ctxt, _sp: span, e: @ast::expr, id: ast::node_id) {
    // FIXME: this is kinda a kludge; we manufacture a fake function context
    // and statement context for checking the initializer expression.
    let rty = ty::node_id_to_type(ccx.tcx, id);
    let fcx: @fn_ctxt =
        @{self_ty: none,
          ret_ty: rty,
          indirect_ret_ty: none,
          purity: ast::pure_fn,
          proto: ast::proto_box,
          infcx: infer::new_infer_ctxt(ccx.tcx),
          locals: int_hash(),
          ty_var_counter: @mut 0u,
          region_var_counter: @mut 0u,
          in_scope_regions: @nil,
          node_types: smallintmap::mk(),
          node_type_substs: map::int_hash(),
          ccx: ccx};
    check_expr(fcx, e);
    let cty = fcx.expr_ty(e);
    let declty = fcx.ccx.tcx.tcache.get(local_def(id)).ty;
    demand::suptype(fcx, e.span, declty, cty);
    writeback::resolve_type_vars_in_expr(fcx, e);
}

fn check_instantiable(tcx: ty::ctxt,
                      sp: span,
                      item_id: ast::node_id) {
    let rty = ty::node_id_to_type(tcx, item_id);
    if !ty::is_instantiable(tcx, rty) {
        tcx.sess.span_err(sp, #fmt["this type cannot be instantiated \
                                    without an instance of itself. \
                                    Consider using option<%s>.",
                                   ty_to_str(tcx, rty)]);
    }
}

fn check_enum_variants(ccx: @crate_ctxt,
                       sp: span,
                       vs: [ast::variant],
                       id: ast::node_id) {
    // FIXME: this is kinda a kludge; we manufacture a fake function context
    // and statement context for checking the initializer expression.
    let rty = ty::node_id_to_type(ccx.tcx, id);
    let fcx: @fn_ctxt =
        @{self_ty: none,
          ret_ty: rty,
          indirect_ret_ty: none,
          purity: ast::pure_fn,
          proto: ast::proto_box,
          infcx: infer::new_infer_ctxt(ccx.tcx),
          locals: int_hash(),
          ty_var_counter: @mut 0u,
          region_var_counter: @mut 0u,
          in_scope_regions: @nil,
          node_types: smallintmap::mk(),
          node_type_substs: map::int_hash(),
          ccx: ccx};
    let mut disr_vals: [int] = [];
    let mut disr_val = 0;
    for vs.each {|v|
        alt v.node.disr_expr {
          some(e) {
            check_expr(fcx, e);
            let cty = fcx.expr_ty(e);
            let declty = ty::mk_int(ccx.tcx);
            demand::suptype(fcx, e.span, declty, cty);
            // FIXME: issue #1417
            // Also, check_expr (from check_const pass) doesn't guarantee that
            // the expression in an form that eval_const_expr can handle, so
            // we may still get an internal compiler error
            alt const_eval::eval_const_expr(ccx.tcx, e) {
              const_eval::const_int(val) {
                disr_val = val as int;
              }
              _ {
                ccx.tcx.sess.span_err(e.span,
                                      "expected signed integer constant");
              }
            }
          }
          _ {}
        }
        if vec::contains(disr_vals, disr_val) {
            ccx.tcx.sess.span_err(v.span,
                                  "discriminator value already exists.");
        }
        disr_vals += [disr_val];
        disr_val += 1;
    }

    // Check that it is possible to represent this enum:
    let mut outer = true, did = local_def(id);
    if ty::type_structurally_contains(ccx.tcx, rty, {|sty|
        alt sty {
          ty::ty_enum(id, _) if id == did {
            if outer { outer = false; false }
            else { true }
          }
          _ { false }
        }
    }) {
        ccx.tcx.sess.span_err(sp, "illegal recursive enum type. \
                                   wrap the inner value in a box to \
                                   make it representable");
    }

    // Check that it is possible to instantiate this enum:
    check_instantiable(ccx.tcx, sp, id);
}

// A generic function for checking the pred in a check
// or if-check
fn check_pred_expr(fcx: @fn_ctxt, e: @ast::expr) -> bool {
    let bot = check_expr_with(fcx, e, ty::mk_bool(fcx.ccx.tcx));

    /* e must be a call expr where all arguments are either
    literals or slots */
    alt e.node {
      ast::expr_call(operator, operands, _) {
        if !ty::is_pred_ty(fcx.expr_ty(operator)) {
            fcx.ccx.tcx.sess.span_err
                (operator.span,
                 "operator in constraint has non-boolean return type");
        }

        alt operator.node {
          ast::expr_path(oper_name) {
            alt fcx.ccx.tcx.def_map.find(operator.id) {
              some(ast::def_fn(_, ast::pure_fn)) {
                // do nothing
              }
              _ {
                fcx.ccx.tcx.sess.span_err(operator.span,
                                            "impure function as operator \
                                             in constraint");
              }
            }
            for operands.each {|operand|
                if !ast_util::is_constraint_arg(operand) {
                    let s =
                        "constraint args must be slot variables or literals";
                    fcx.ccx.tcx.sess.span_err(e.span, s);
                }
            }
          }
          _ {
            let s = "in a constraint, expected the \
                     constraint name to be an explicit name";
            fcx.ccx.tcx.sess.span_err(e.span, s);
          }
        }
      }
      _ { fcx.ccx.tcx.sess.span_err(e.span, "check on non-predicate"); }
    }
    ret bot;
}

fn check_constraints(fcx: @fn_ctxt, cs: [@ast::constr], args: [ast::arg]) {
    let num_args = vec::len(args);
    for cs.each {|c|
        let mut c_args = [];
        for c.node.args.each {|a|
            c_args += [
                 // "base" should not occur in a fn type thing, as of
                 // yet, b/c we don't allow constraints on the return type

                 // Works b/c no higher-order polymorphism
                 /*
                 This is kludgy, and we probably shouldn't be assigning
                 node IDs here, but we're creating exprs that are
                 ephemeral, just for the purposes of typechecking. So
                 that's my justification.
                 */
                 @alt a.node {
                    ast::carg_base {
                      fcx.ccx.tcx.sess.span_bug(a.span,
                                                "check_constraints:\
                    unexpected carg_base");
                    }
                    ast::carg_lit(l) {
                      let tmp_node_id = fcx.ccx.tcx.sess.next_node_id();
                      {id: tmp_node_id, node: ast::expr_lit(l), span: a.span}
                    }
                    ast::carg_ident(i) {
                      if i < num_args {
                          let p = @{span: a.span, global: false,
                                    idents: [args[i].ident], types: []};
                          let arg_occ_node_id =
                              fcx.ccx.tcx.sess.next_node_id();
                          fcx.ccx.tcx.def_map.insert
                              (arg_occ_node_id,
                               ast::def_arg(args[i].id, args[i].mode));
                          {id: arg_occ_node_id,
                           node: ast::expr_path(p),
                           span: a.span}
                      } else {
                          fcx.ccx.tcx.sess.span_bug(
                              a.span, "check_constraints:\
                                       carg_ident index out of bounds");
                      }
                    }
                  }];
        }
        let p_op: ast::expr_ = ast::expr_path(c.node.path);
        let oper: @ast::expr = @{id: c.node.id, node: p_op, span: c.span};
        // Another ephemeral expr
        let call_expr_id = fcx.ccx.tcx.sess.next_node_id();
        let call_expr =
            @{id: call_expr_id,
              node: ast::expr_call(oper, c_args, false),
              span: c.span};
        check_pred_expr(fcx, call_expr);
    }
}

fn check_bare_fn(ccx: @crate_ctxt,
                 decl: ast::fn_decl,
                 body: ast::blk,
                 id: ast::node_id,
                 self_ty: option<ty::t>) {
    let fty = ty::node_id_to_type(ccx.tcx, id);
    let ret_ty = ty::ty_fn_ret(fty);
    let arg_tys = vec::map(ty::ty_fn_args(fty)) {|a| a.ty };
    check_fn(ccx, ast::proto_bare, decl, body, id,
             ret_ty, arg_tys, false, none, self_ty);
}

fn check_fn(ccx: @crate_ctxt,
            proto: ast::proto,
            decl: ast::fn_decl,
            body: ast::blk,
            fid: ast::node_id,
            ret_ty: ty::t,
            arg_tys: [ty::t],
            indirect_ret: bool,
            old_fcx: option<@fn_ctxt>,
            self_ty: option<ty::t>) {

    let tcx = ccx.tcx;

    let isr = {
        // Find the list of in-scope regions.  These are derived from the
        // various regions that are bound in the argument, return, and self
        // types.  For each of those bound regions, we will create a mapping
        // to a free region tied to the node_id of this function.  For an
        // in-depth discussion of why we must distinguish bound/free regions,
        // see the big comment in region.rs.
        let all_tys = arg_tys + [ret_ty] + self_ty.to_vec();
        let old_isr = option::map_default(old_fcx, @nil) {
            |fcx| fcx.in_scope_regions };
        collect_bound_regions_in_tys(tcx, old_isr, all_tys) {
            |br| ty::re_free(fid, br) }
    };

    // Replace the bound regions that appear in the arg tys, ret ty, etc with
    // the free versions we just collected.
    let arg_tys = arg_tys.map {
        |arg_ty| replace_bound_regions(tcx, body.span, isr, arg_ty)
    };
    let ret_ty = {
        replace_bound_regions(tcx, body.span, isr, ret_ty)
    };
    let self_ty = option::map(self_ty) {
        |self_ty| replace_bound_regions(tcx, body.span, isr, self_ty)
    };

    #debug["check_fn(arg_tys=%?, ret_ty=%?, self_ty=%?)",
           arg_tys.map {|a| ty_to_str(tcx, a) },
           ty_to_str(tcx, ret_ty),
           option::map(self_ty) {|st| ty_to_str(tcx, st) }];

    // Create the function context.  This is either derived from scratch or,
    // in the case of function expressions, based on the outer context.
    let fcx: @fn_ctxt = {
        let {infcx, locals, tvc, rvc, purity,
             node_types, node_type_substs} = alt old_fcx {
          none {
            {infcx: infer::new_infer_ctxt(tcx),
             locals: int_hash(),
             tvc: @mut 0u,
             rvc: @mut 0u,
             purity: decl.purity,
             node_types: smallintmap::mk(),
             node_type_substs: map::int_hash()}
          }
          some(fcx) {
            assert decl.purity == ast::impure_fn;
            {infcx: fcx.infcx,
             locals: fcx.locals,
             tvc: fcx.ty_var_counter,
             rvc: fcx.region_var_counter,
             purity: fcx.purity,
             node_types: fcx.node_types,
             node_type_substs: fcx.node_type_substs}
          }
        };

        let indirect_ret_ty = if indirect_ret {
            let ofcx = option::get(old_fcx);
            alt ofcx.indirect_ret_ty {
              some(t) { some(t) }
              none { some(ofcx.ret_ty) }
            }
        } else { none };

        @{self_ty: self_ty,
          ret_ty: ret_ty,
          indirect_ret_ty: indirect_ret_ty,
          purity: purity,
          proto: proto,
          infcx: infcx,
          locals: locals,
          ty_var_counter: tvc,
          region_var_counter: rvc,
          in_scope_regions: isr,
          node_types: node_types,
          node_type_substs: node_type_substs,
          ccx: ccx}
    };

    gather_locals(fcx, decl, body, arg_tys);
    check_constraints(fcx, decl.constraints, decl.inputs);
    check_block(fcx, body);

    // We unify the tail expr's type with the
    // function result type, if there is a tail expr.
    alt body.node.expr {
      some(tail_expr) {
        let tail_expr_ty = fcx.expr_ty(tail_expr);
        demand::suptype(fcx, tail_expr.span, fcx.ret_ty, tail_expr_ty);
      }
      none { }
    }

    let mut i = 0u;
    vec::iter(arg_tys) {|arg|
        fcx.write_ty(decl.inputs[i].id, arg);
        i += 1u;
    }

    // If we don't have any enclosing function scope, it is time to
    // force any remaining type vars to be resolved.
    // If we have an enclosing function scope, our type variables will be
    // resolved when the enclosing scope finishes up.
    if option::is_none(old_fcx) {
        vtable::resolve_in_block(fcx, body);
        writeback::resolve_type_vars_in_fn(fcx, decl, body);
    }

    fn gather_locals(fcx: @fn_ctxt,
                     decl: ast::fn_decl,
                     body: ast::blk,
                     arg_tys: [ty::t]) {
        let tcx = fcx.ccx.tcx;

        let assign = fn@(nid: ast::node_id, ty_opt: option<ty::t>) {
            let var_id = fcx.next_ty_var_id();
            fcx.locals.insert(nid, var_id);
            alt ty_opt {
              none {/* nothing to do */ }
              some(typ) {
                infer::mk_eqty(fcx.infcx, ty::mk_var(tcx, var_id), typ);
              }
            }
        };

        // Add formal parameters.
        vec::iter2(arg_tys, decl.inputs) {|arg_ty, input|
            assign(input.id, some(arg_ty));
            #debug["Argument %s is assigned to %s",
                   input.ident, fcx.locals.get(input.id).to_str()];
        }

        // Add explicitly-declared locals.
        let visit_local = fn@(local: @ast::local,
                              &&e: (), v: visit::vt<()>) {
            let o_ty = alt local.node.ty.node {
              ast::ty_infer { none }
              _ { some(fcx.to_ty(local.node.ty)) }
            };
            assign(local.node.id, o_ty);
            #debug["Local variable %s is assigned to %s",
                   pat_to_str(local.node.pat),
                   fcx.locals.get(local.node.id).to_str()];
            visit::visit_local(local, e, v);
        };

        // Add pattern bindings.
        let visit_pat = fn@(p: @ast::pat, &&e: (), v: visit::vt<()>) {
            alt p.node {
              ast::pat_ident(path, _)
              if !pat_util::pat_is_variant(fcx.ccx.tcx.def_map, p) {
                assign(p.id, none);
                #debug["Pattern binding %s is assigned to %s",
                       path.idents[0],
                       fcx.locals.get(p.id).to_str()];
              }
              _ {}
            }
            visit::visit_pat(p, e, v);
        };

        // Don't descend into fns and items
        fn visit_fn<T>(_fk: visit::fn_kind, _decl: ast::fn_decl,
                       _body: ast::blk, _sp: span,
                       _id: ast::node_id, _t: T, _v: visit::vt<T>) {
        }
        fn visit_item<E>(_i: @ast::item, _e: E, _v: visit::vt<E>) { }

        let visit =
            @{visit_local: visit_local,
              visit_pat: visit_pat,
              visit_fn: bind visit_fn(_, _, _, _, _, _, _),
              visit_item: bind visit_item(_, _, _)
              with *visit::default_visitor()};

        visit::visit_block(body, (), visit::mk_vt(visit));
    }
}

fn check_method(ccx: @crate_ctxt, method: @ast::method, self_ty: ty::t) {
    check_bare_fn(ccx, method.decl, method.body, method.id, some(self_ty));
}

fn class_types(ccx: @crate_ctxt, members: [@ast::class_member],
               rp: ast::region_param) -> class_map {

    let rslt = int_hash::<ty::t>();
    let rs = type_rscope(rp);
    for members.each { |m|
      alt m.node {
         ast::instance_var(_,t,_,id,_) {
           rslt.insert(id, ccx.to_ty(rs, t));
         }
         ast::class_method(mth) {
           rslt.insert(mth.id,
                       ty::mk_fn(ccx.tcx,
                                 ty_of_method(ccx, mth, rp).fty));
         }
      }
    }
    rslt
}

fn check_class_member(ccx: @crate_ctxt, class_t: ty::t,
                      cm: @ast::class_member) {
    alt cm.node {
      ast::instance_var(_,t,_,_,_) { }
      ast::class_method(m) {
          check_method(ccx, m, class_t);
      }
    }
}

fn check_item(ccx: @crate_ctxt, it: @ast::item) {
    alt it.node {
      ast::item_const(_, e) { check_const(ccx, it.span, e, it.id); }
      ast::item_enum(vs, _, _) {
        check_enum_variants(ccx, it.span, vs, it.id);
      }
      ast::item_fn(decl, tps, body) {
        check_bare_fn(ccx, decl, body, it.id, none);
      }
      ast::item_res(decl, tps, body, dtor_id, _, rp) {
        check_instantiable(ccx.tcx, it.span, it.id);
        check_bare_fn(ccx, decl, body, dtor_id, none);
      }
      ast::item_impl(tps, _, ty, ms) {
        let self_ty = ccx.to_ty(empty_rscope, ty); // NDM iface/impl regions
        let self_region = ty::re_free(it.id, ty::br_self);
        let self_ty = replace_self_region(ccx.tcx, self_region, self_ty);
        for ms.each {|m| check_method(ccx, m, self_ty);}
      }
      ast::item_class(tps, ifaces, members, ctor, rp) {
          let cid = some(it.id), tcx = ccx.tcx;
          let class_t = ty::node_id_to_type(tcx, it.id);
          let members_info = class_types(ccx, members, rp);
          // can also ditch the enclosing_class stuff once we move to self
          // FIXME
          let class_ccx = @{enclosing_class_id:cid,
                            enclosing_class:members_info with *ccx};
          // typecheck the ctor
          check_bare_fn(class_ccx, ctor.node.dec,
                        ctor.node.body, ctor.node.id,
                        some(class_t));

          // typecheck the members
          for members.each {|m| check_class_member(class_ccx, class_t, m); }
      }
      _ {/* nothing to do */ }
    }
}

fn arg_is_argv_ty(_tcx: ty::ctxt, a: ty::arg) -> bool {
    alt ty::get(a.ty).struct {
      ty::ty_vec(mt) {
        if mt.mutbl != ast::m_imm { ret false; }
        alt ty::get(mt.ty).struct {
          ty::ty_str { ret true; }
          _ { ret false; }
        }
      }
      _ { ret false; }
    }
}

fn check_main_fn_ty(ccx: @crate_ctxt,
                    main_id: ast::node_id,
                    main_span: span) {

    let tcx = ccx.tcx;
    let main_t = ty::node_id_to_type(tcx, main_id);
    alt ty::get(main_t).struct {
      ty::ty_fn({proto: ast::proto_bare, inputs, output,
                 ret_style: ast::return_val, constraints}) {
        alt tcx.items.find(main_id) {
         some(ast_map::node_item(it,_)) {
             alt it.node {
               ast::item_fn(_,ps,_) if vec::is_not_empty(ps) {
                  tcx.sess.span_err(main_span,
                    "main function is not allowed to have type parameters");
                  ret;
               }
               _ {}
             }
         }
         _ {}
        }
        let mut ok = vec::len(constraints) == 0u;
        ok &= ty::type_is_nil(output);
        let num_args = vec::len(inputs);
        ok &= num_args == 0u || num_args == 1u &&
              arg_is_argv_ty(tcx, inputs[0]);
        if !ok {
                tcx.sess.span_err(main_span,
                   #fmt("Wrong type in main function: found `%s`, \
                   expecting `native fn([str]) -> ()` or `native fn() -> ()`",
                         ty_to_str(tcx, main_t)));
         }
      }
      _ {
        tcx.sess.span_bug(main_span,
                          "main has a non-function type: found `" +
                              ty_to_str(tcx, main_t) + "`");
      }
    }
}

fn check_for_main_fn(ccx: @crate_ctxt, crate: @ast::crate) {
    let tcx = ccx.tcx;
    if !tcx.sess.building_library {
        alt tcx.sess.main_fn {
          some((id, sp)) { check_main_fn_ty(ccx, id, sp); }
          none { tcx.sess.span_err(crate.span, "main function not found"); }
        }
    }
}

mod vtable {
    fn has_iface_bounds(tps: [ty::param_bounds]) -> bool {
        vec::any(tps, {|bs|
            vec::any(*bs, {|b|
                alt b { ty::bound_iface(_) { true } _ { false } }
            })
        })
    }

    fn lookup_vtables(fcx: @fn_ctxt, isc: resolve::iscopes, sp: span,
                      bounds: @[ty::param_bounds], tys: [ty::t],
                      allow_unsafe: bool) -> vtable_res {
        let tcx = fcx.ccx.tcx;
        let mut result = [], i = 0u;
        for tys.each {|ty|
            for vec::each(*bounds[i]) {|bound|
                alt bound {
                  ty::bound_iface(i_ty) {
                    let i_ty = ty::subst_tps(tcx, tys, i_ty);
                    result += [lookup_vtable(fcx, isc, sp, ty, i_ty,
                                             allow_unsafe)];
                  }
                  _ {}
                }
            }
            i += 1u;
        }
        @result
    }

    fn lookup_vtable(fcx: @fn_ctxt, isc: resolve::iscopes, sp: span,
                     ty: ty::t, iface_ty: ty::t, allow_unsafe: bool)
        -> vtable_origin {
        let tcx = fcx.ccx.tcx;
        let (iface_id, iface_tps) = alt check ty::get(iface_ty).struct {
            ty::ty_iface(did, tps) { (did, tps) }
        };
        let ty = fixup_ty(fcx, sp, ty);
        alt ty::get(ty).struct {
          ty::ty_param(n, did) {
            let mut n_bound = 0u;
            for vec::each(*tcx.ty_param_bounds.get(did.node)) {|bound|
                alt bound {
                  ty::bound_iface(ity) {
                    alt check ty::get(ity).struct {
                      ty::ty_iface(idid, _) {
                        if iface_id == idid { ret vtable_param(n, n_bound); }
                      }
                    }
                    n_bound += 1u;
                  }
                  _ {}
                }
            }
          }
          ty::ty_iface(did, tps) if iface_id == did {
            if !allow_unsafe {
                for vec::each(*ty::iface_methods(tcx, did)) {|m|
                    if ty::type_has_vars(ty::mk_fn(tcx, m.fty)) {
                        tcx.sess.span_err(
                            sp, "a boxed iface with self types may not be \
                                 passed as a bounded type");
                    } else if (*m.tps).len() > 0u {
                        tcx.sess.span_err(
                            sp, "a boxed iface with generic methods may not \
                                 be passed as a bounded type");

                    }
                }
            }
            ret vtable_iface(did, tps);
          }
          _ {
            for list::each(isc) {|impls|
                let mut found = none;
                for vec::each(*impls) {|im|
                    let match = alt ty::impl_iface(tcx, im.did) {
                      some(ity) {
                        alt check ty::get(ity).struct {
                          ty::ty_iface(id, _) { id == iface_id }
                        }
                      }
                      _ { false }
                    };
                    if match {
                        let {substs: substs, ty: self_ty} =
                            impl_self_ty(fcx, im.did);
                        let im_bs = ty::lookup_item_type(tcx, im.did).bounds;
                        alt unify::unify(fcx, ty, self_ty) {
                          result::ok(_) {
                            if option::is_some(found) {
                                tcx.sess.span_err(
                                    sp, "multiple applicable implementations \
                                         in scope");
                            } else {
                                let vars = substs.tps;
                                connect_iface_tps(fcx, sp, vars,
                                                  iface_tps, im.did);
                                let params = vec::map(vars, {|t|
                                    fixup_ty(fcx, sp, t)});
                                let subres = lookup_vtables(
                                    fcx, isc, sp, im_bs, params, false);
                                found = some(vtable_static(im.did, params,
                                                           subres));
                            }
                          }
                          result::err(_) {}
                        }
                    }
                }
                alt found { some(x) { ret x; } _ {} }
            }
          }
        }

        tcx.sess.span_fatal(
            sp, "failed to find an implementation of interface " +
            ty_to_str(tcx, iface_ty) + " for " +
            ty_to_str(tcx, ty));
    }

    fn fixup_ty(fcx: @fn_ctxt, sp: span, ty: ty::t) -> ty::t {
        let tcx = fcx.ccx.tcx;
        alt infer::resolve_deep(fcx.infcx, ty, true) {
          result::ok(new_type) { new_type }
          result::err(e) {
            tcx.sess.span_fatal(
                sp,
                #fmt["cannot determine a type \
                      for this bounded type parameter: %s",
                     infer::fixup_err_to_str(e)])
          }
        }
    }

    fn connect_iface_tps(fcx: @fn_ctxt, sp: span, impl_tys: [ty::t],
                         iface_tys: [ty::t], impl_did: ast::def_id) {
        let tcx = fcx.ccx.tcx;
        let ity = option::get(ty::impl_iface(tcx, impl_did));
        let iface_ty = ty::subst_tps(tcx, impl_tys, ity);
        alt check ty::get(iface_ty).struct {
          ty::ty_iface(_, tps) {
            vec::iter2(tps, iface_tys,
                       {|a, b| demand::suptype(fcx, sp, a, b);});
          }
        }
    }

    fn resolve_expr(ex: @ast::expr, &&fcx: @fn_ctxt, v: visit::vt<@fn_ctxt>) {
        let cx = fcx.ccx;
        alt ex.node {
          ast::expr_path(_) {
            alt fcx.opt_node_ty_substs(ex.id) {
              some(substs) {
                let ts = substs.tps; // NDM regions for iface/impls
                let did = ast_util::def_id_of_def(cx.tcx.def_map.get(ex.id));
                let item_ty = ty::lookup_item_type(cx.tcx, did);
                if has_iface_bounds(*item_ty.bounds) {
                    let impls = cx.impl_map.get(ex.id);
                    cx.vtable_map.insert(ex.id, lookup_vtables(
                        fcx, impls, ex.span,
                        item_ty.bounds, ts, false));
                }
              }
              _ {}
            }
          }
          // Must resolve bounds on methods with bounded params
          ast::expr_field(_, _, _) | ast::expr_binary(_, _, _) |
          ast::expr_unary(_, _) | ast::expr_assign_op(_, _, _) |
          ast::expr_index(_, _) {
            alt cx.method_map.find(ex.id) {
              some(method_static(did)) {
                let bounds = ty::lookup_item_type(cx.tcx, did).bounds;
                if has_iface_bounds(*bounds) {
                    let callee_id = alt ex.node {
                      ast::expr_field(_, _, _) { ex.id }
                      _ { ast_util::op_expr_callee_id(ex) }
                    };
                    // NDM iface/impl regions
                    let ts = fcx.node_ty_substs(callee_id).tps;
                    let iscs = cx.impl_map.get(ex.id);
                    cx.vtable_map.insert(callee_id, lookup_vtables(
                        fcx, iscs, ex.span, bounds, ts, false));
                }
              }
              _ {}
            }
          }
          ast::expr_cast(src, _) {
            let target_ty = fcx.expr_ty(ex);
            alt ty::get(target_ty).struct {
              ty::ty_iface(_, _) {
                let impls = cx.impl_map.get(ex.id);
                let vtable = lookup_vtable(fcx, impls, ex.span,
                                           fcx.expr_ty(src), target_ty,
                                           true);
                cx.vtable_map.insert(ex.id, @[vtable]);
              }
              _ {}
            }
          }
          _ {}
        }
        visit::visit_expr(ex, fcx, v);
    }

    // Detect points where an interface-bounded type parameter is
    // instantiated, resolve the impls for the parameters.
    fn resolve_in_block(fcx: @fn_ctxt, bl: ast::blk) {
        visit::visit_block(bl, fcx, visit::mk_vt(@{
            visit_expr: resolve_expr,
            visit_item: fn@(_i: @ast::item, &&_e: @fn_ctxt,
                            _v: visit::vt<@fn_ctxt>) {}
            with *visit::default_visitor()
        }));
    }
}

fn check_crate(tcx: ty::ctxt, impl_map: resolve::impl_map,
               crate: @ast::crate) -> (method_map, vtable_map) {
    let ccx = @{impl_map: impl_map,
                method_map: std::map::int_hash(),
                vtable_map: std::map::int_hash(),
                enclosing_class_id: none,
                enclosing_class: std::map::int_hash(),
                tcx: tcx};
    collect::collect_item_types(ccx, crate);
    let visit = visit::mk_simple_visitor(@{
        visit_item: bind check_item(ccx, _)
        with *visit::default_simple_visitor()
    });
    visit::visit_crate(*crate, (), visit);
    check_for_main_fn(ccx, crate);
    tcx.sess.abort_if_errors();
    (ccx.method_map, ccx.vtable_map)
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
