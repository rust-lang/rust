
import syntax::ast;
import syntax::ast_util;
import ast::ident;
import ast::fn_ident;
import ast::node_id;
import ast::def_id;
import mut::{expr_root, mut_field, inner_mut};
import syntax::codemap::span;
import syntax::visit;
import visit::vt;
import std::vec;
import std::str;
import std::option;
import std::option::some;
import std::option::none;
import std::option::is_none;

// This is not an alias-analyser (though it would merit from becoming one, or
// getting input from one, to be more precise). It is a pass that checks
// whether aliases are used in a safe way.

tag valid { valid; overwritten(span, ast::path); val_taken(span, ast::path); }

type restrict =
    @{root_var: option::t<node_id>,
      local_id: uint,
      bindings: [node_id],
      unsafe_ty: option::t<ty::t>,
      depends_on: [uint],
      mutable ok: valid};

type scope = @[restrict];

tag local_info { local(uint); }

type ctx =
    {tcx: ty::ctxt,
     local_map: std::map::hashmap<node_id, local_info>,
     mutable next_local: uint};

fn check_crate(tcx: ty::ctxt, crate: &@ast::crate) {
    // Stores information about object fields and function
    // arguments that's otherwise not easily available.
    let cx =
        @{tcx: tcx,
          local_map: std::map::new_int_hash(),
          mutable next_local: 0u};
    let v =
        @{visit_fn: visit_fn,
          visit_expr: bind visit_expr(cx, _, _, _),
          visit_decl: bind visit_decl(cx, _, _, _)
             with *visit::default_visitor::<scope>()};
    visit::visit_crate(*crate, @[], visit::mk_vt(v));
    tcx.sess.abort_if_errors();
}

fn visit_fn(f: &ast::_fn, _tp: &[ast::ty_param], _sp: &span, _name: &fn_ident,
            _id: ast::node_id, sc: &scope, v: &vt<scope>) {
    visit::visit_fn_decl(f.decl, sc, v);
    let scope =
        alt f.proto {

          // Blocks need to obey any restrictions from the enclosing scope.
          ast::proto_block. | ast::proto_closure. {
            sc
          }

          // Non capturing functions start out fresh.
          _ {
            @[]
          }
        };
    v.visit_block(f.body, scope, v);
}

fn visit_expr(cx: &@ctx, ex: &@ast::expr, sc: &scope, v: &vt<scope>) {
    let handled = true;
    alt ex.node {
      ast::expr_call(f, args) {
        check_call(*cx, f, args, sc);
        handled = false;
      }
      ast::expr_alt(input, arms) { check_alt(*cx, input, arms, sc, v); }
      ast::expr_put(val) {
        alt val {
          some(ex) {
            let root = expr_root(cx.tcx, ex, false);
            if mut_field(root.ds) {
                cx.tcx.sess.span_err(ex.span,
                                     "result of put must be" +
                                         " immutably rooted");
            }
            visit_expr(cx, ex, sc, v);
          }
          _ { }
        }
      }
      ast::expr_for_each(decl, call, blk) {
        check_for_each(*cx, decl, call, blk, sc, v);
      }
      ast::expr_for(decl, seq, blk) { check_for(*cx, decl, seq, blk, sc, v); }
      ast::expr_path(pt) {
        check_var(*cx, ex, pt, ex.id, false, sc);
        handled = false;
      }
      ast::expr_swap(lhs, rhs) {
        check_lval(cx, lhs, sc, v);
        check_lval(cx, rhs, sc, v);
        handled = false;
      }
      ast::expr_move(dest, src) {
        check_assign(cx, dest, src, sc, v);
        check_lval(cx, src, sc, v);
      }
      ast::expr_assign(dest, src) | ast::expr_assign_op(_, dest, src) {
        check_assign(cx, dest, src, sc, v);
      }
      _ { handled = false; }
    }
    if !handled { visit::visit_expr(ex, sc, v); }
}

fn register_locals(cx: &ctx, pat: &@ast::pat) {
    for each pat in ast_util::pat_bindings(pat) {
        cx.local_map.insert(pat.id, local(cx.next_local));
        cx.next_local += 1u;
    }
}

fn visit_decl(cx: &@ctx, d: &@ast::decl, sc: &scope, v: &vt<scope>) {
    visit::visit_decl(d, sc, v);
    alt d.node {
      ast::decl_local(locs) {
        for loc: @ast::local in locs {
            alt loc.node.init {
              some(init) {
                if init.op == ast::init_move {
                    check_lval(cx, init.expr, sc, v);
                }
              }
              none. { }
            }
            register_locals(*cx, loc.node.pat);
        }
      }
      _ { }
    }
}

fn check_call(cx: &ctx, f: &@ast::expr, args: &[@ast::expr], sc: &scope) ->
   [restrict] {
    let fty = ty::type_autoderef(cx.tcx, ty::expr_ty(cx.tcx, f));
    let arg_ts = ty::ty_fn_args(cx.tcx, fty);
    let mut_roots: [{arg: uint, node: node_id}] = [];
    let restricts = [];
    let i = 0u;
    for arg_t: ty::arg in arg_ts {
        if arg_t.mode != ty::mo_val {
            let arg = args[i];
            let root = expr_root(cx.tcx, arg, false);
            if arg_t.mode == ty::mo_alias(true) {
                alt path_def(cx, arg) {
                  some(def) {
                    let dnum = ast_util::def_id_of_def(def).node;
                    mut_roots += [{arg: i, node: dnum}];
                  }
                  _ { }
                }
            }
            let root_var = path_def_id(cx, root.ex);
            let unsafe_t =
                alt inner_mut(root.ds) { some(t) { some(t) } _ { none } };
            restricts +=
                [@{root_var: root_var,
                   local_id: cx.next_local,
                   bindings: [arg.id],
                   unsafe_ty: unsafe_t,
                   depends_on: deps(sc, root_var),
                   mutable ok: valid}];
        }
        i += 1u;
    }
    let f_may_close =
        alt f.node {
          ast::expr_path(_) { def_is_local(cx.tcx.def_map.get(f.id), true) }
          _ { true }
        };
    if f_may_close {
        let i = 0u;
        for r in restricts {
            if !option::is_none(r.unsafe_ty) {
                cx.tcx.sess.span_err(f.span,
                                     #fmt["function may alias with argument \
                                           %u, which is not immutably rooted",
                                          i]);
            }
            i += 1u;
        }
    }
    let j = 0u;
    for @{unsafe_ty: unsafe_ty, _} in restricts {
        alt unsafe_ty {
          some(ty) {
            let i = 0u;
            for arg_t: ty::arg in arg_ts {
                let mut_alias = arg_t.mode == ty::mo_alias(true);
                if i != j &&
                       ty_can_unsafely_include(cx, ty, arg_t.ty, mut_alias) {
                    cx.tcx.sess.span_err(
                        args[i].span,
                        #fmt["argument %u may alias with argument %u, \
                               which is not immutably rooted",
                                              i, j]);
                }
                i += 1u;
            }
          }
          _ { }
        }
        j += 1u;
    }
    // Ensure we're not passing a root by mutable alias.

    for {node: node, arg: arg} in mut_roots {
        let mut_alias_to_root = false;
        let mut_alias_to_root_count = 0u;
        for @{root_var: root_var, _} in restricts {
            alt root_var {
              some(root) {
                if node == root {
                    mut_alias_to_root_count += 1u;
                    if mut_alias_to_root_count > 1u {
                        mut_alias_to_root = true;
                        break;
                    }
                }
              }
              none. { }
            }
        }

        if mut_alias_to_root {
            cx.tcx.sess.span_err(args[arg].span,
                                 "passing a mutable alias to a variable \
                                   that roots another alias");
        }
    }
    ret restricts;
}

fn check_alt(cx: &ctx, input: &@ast::expr, arms: &[ast::arm], sc: &scope,
             v: &vt<scope>) {
    v.visit_expr(input, sc, v);
    let root = expr_root(cx.tcx, input, true);
    for a: ast::arm in arms {
        let dnums = ast_util::pat_binding_ids(a.pats[0]);
        let new_sc = sc;
        if vec::len(dnums) > 0u {
            let root_var = path_def_id(cx, root.ex);
            new_sc =
                @(*sc +
                      [@{root_var: root_var,
                         local_id: cx.next_local,
                         bindings: dnums,
                         unsafe_ty: inner_mut(root.ds),
                         depends_on: deps(sc, root_var),
                         mutable ok: valid}]);
        }
        register_locals(cx, a.pats[0]);
        visit::visit_arm(a, new_sc, v);
    }
}

fn check_for_each(cx: &ctx, local: &@ast::local, call: &@ast::expr,
                  blk: &ast::blk, sc: &scope, v: &vt<scope>) {
    v.visit_expr(call, sc, v);
    alt call.node {
      ast::expr_call(f, args) {
        let restricts = check_call(cx, f, args, sc);
        register_locals(cx, local.node.pat);
        visit::visit_block(blk, @(*sc + restricts), v);
      }
    }
}

fn check_for(cx: &ctx, local: &@ast::local, seq: &@ast::expr, blk: &ast::blk,
             sc: &scope, v: &vt<scope>) {
    v.visit_expr(seq, sc, v);
    let root = expr_root(cx.tcx, seq, false);
    let unsafe = inner_mut(root.ds);

    // If this is a mutable vector, don't allow it to be touched.
    let seq_t = ty::expr_ty(cx.tcx, seq);
    alt ty::struct(cx.tcx, seq_t) {
      ty::ty_vec(mt) { if mt.mut != ast::imm { unsafe = some(seq_t); } }
      ty::ty_str. {/* no-op */ }
      _ {
        cx.tcx.sess.span_unimpl(seq.span,
                                "unknown seq type " +
                                    util::ppaux::ty_to_str(cx.tcx, seq_t));
      }
    }
    let root_var = path_def_id(cx, root.ex);
    let new_sc =
        @{root_var: root_var,
          local_id: cx.next_local,
          bindings: ast_util::pat_binding_ids(local.node.pat),
          unsafe_ty: unsafe,
          depends_on: deps(sc, root_var),
          mutable ok: valid};
    register_locals(cx, local.node.pat);
    visit::visit_block(blk, @(*sc + [new_sc]), v);
}

fn check_var(cx: &ctx, ex: &@ast::expr, p: &ast::path, id: ast::node_id,
             assign: bool, sc: &scope) {
    let def = cx.tcx.def_map.get(id);
    if !def_is_local(def, true) { ret; }
    let my_defnum = ast_util::def_id_of_def(def).node;
    let my_local_id =
        alt cx.local_map.find(my_defnum) { some(local(id)) { id } _ { 0u } };
    let var_t = ty::expr_ty(cx.tcx, ex);
    for r: restrict in *sc {

        // excludes variables introduced since the alias was made
        if my_local_id < r.local_id {
            alt r.unsafe_ty {
              some(ty) {
                if ty_can_unsafely_include(cx, ty, var_t, assign) {
                    r.ok = val_taken(ex.span, p);
                }
              }
              _ { }
            }
        } else if vec::member(my_defnum, r.bindings) {
            test_scope(cx, sc, r, p);
        }
    }
}

fn check_lval(cx: &@ctx, dest: &@ast::expr, sc: &scope, v: &vt<scope>) {
    alt dest.node {
      ast::expr_path(p) {
        let def = cx.tcx.def_map.get(dest.id);
        let dnum = ast_util::def_id_of_def(def).node;
        for r: restrict in *sc {
            if r.root_var == some(dnum) { r.ok = overwritten(dest.span, p); }
        }
      }
      _ { visit_expr(cx, dest, sc, v); }
    }
}

fn check_assign(cx: &@ctx, dest: &@ast::expr, src: &@ast::expr, sc: &scope,
                v: &vt<scope>) {
    visit_expr(cx, src, sc, v);
    check_lval(cx, dest, sc, v);
}

fn test_scope(cx: &ctx, sc: &scope, r: &restrict, p: &ast::path) {
    let prob = r.ok;
    for dep: uint in r.depends_on {
        if prob != valid { break; }
        prob = sc[dep].ok;
    }
    if prob != valid {
        let msg =
            alt prob {
              overwritten(sp, wpt) {
                {span: sp, msg: "overwriting " + ast_util::path_name(wpt)}
              }
              val_taken(sp, vpt) {
                {span: sp,
                 msg: "taking the value of " + ast_util::path_name(vpt)}
              }
            };
        cx.tcx.sess.span_err(msg.span,
                             msg.msg + " will invalidate alias " +
                                 ast_util::path_name(p) +
                                 ", which is still used");
    }
}

fn deps(sc: &scope, root: &option::t<node_id>) -> [uint] {
    let result = [];
    alt root {
      some(dn) {
        let i = 0u;
        for r: restrict in *sc {
            if vec::member(dn, r.bindings) { result += [i]; }
            i += 1u;
        }
      }
      _ { }
    }
    ret result;
}

fn path_def(cx: &ctx, ex: &@ast::expr) -> option::t<ast::def> {
    ret alt ex.node {
          ast::expr_path(_) { some(cx.tcx.def_map.get(ex.id)) }
          _ { none }
        }
}

fn path_def_id(cx: &ctx, ex: &@ast::expr) -> option::t<ast::node_id> {
    alt ex.node {
      ast::expr_path(_) {
        ret some(ast_util::def_id_of_def(cx.tcx.def_map.get(ex.id)).node);
      }
      _ { ret none; }
    }
}

fn ty_can_unsafely_include(cx: &ctx, needle: ty::t, haystack: ty::t,
                           mut: bool) -> bool {
    fn get_mut(cur: bool, mt: &ty::mt) -> bool {
        ret cur || mt.mut != ast::imm;
    }
    fn helper(tcx: &ty::ctxt, needle: ty::t, haystack: ty::t, mut: bool) ->
       bool {
        if needle == haystack { ret true; }
        alt ty::struct(tcx, haystack) {
          ty::ty_tag(_, ts) {
            for t: ty::t in ts {
                if helper(tcx, needle, t, mut) { ret true; }
            }
            ret false;
          }
          ty::ty_box(mt) | ty::ty_ptr(mt) {
            ret helper(tcx, needle, mt.ty, get_mut(mut, mt));
          }
          ty::ty_uniq(t) { ret helper(tcx, needle, t, false); }
          ty::ty_rec(fields) {
            for f: ty::field in fields {
                if helper(tcx, needle, f.mt.ty, get_mut(mut, f.mt)) {
                    ret true;
                }
            }
            ret false;
          }
          ty::ty_tup(ts) {
            for t in ts { if helper(tcx, needle, t, mut) { ret true; } }
            ret false;
          }



          // These may contain anything.
          ty::ty_fn(_, _, _, _, _) {
            ret true;
          }
          ty::ty_obj(_) { ret true; }



          // A type param may include everything, but can only be
          // treated as opaque downstream, and is thus safe unless we
          // saw mutable fields, in which case the whole thing can be
          // overwritten.
          ty::ty_param(_, _) {
            ret mut;
          }
          _ { ret false; }
        }
    }
    ret helper(cx.tcx, needle, haystack, mut);
}

fn def_is_local(d: &ast::def, objfields_count: bool) -> bool {
    ret alt d {
          ast::def_local(_) | ast::def_arg(_, _) | ast::def_binding(_) |
          ast::def_upvar(_, _, _) {
            true
          }
          ast::def_obj_field(_, _) { objfields_count }
          _ { false }
        };
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
