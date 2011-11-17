
import syntax::{ast, ast_util};
import ast::{ident, fn_ident, node_id};
import syntax::codemap::span;
import syntax::visit;
import visit::vt;
import std::{vec, option, list};
import std::option::{some, none, is_none};
import list::list;

// This is not an alias-analyser (though it would merit from becoming one, or
// getting input from one, to be more precise). It is a pass that checks
// whether aliases are used in a safe way.

tag copied { not_allowed; copied; not_copied; }
tag invalid_reason { overwritten; val_taken; }
type invalid = {reason: invalid_reason,
                node_id: node_id,
                sp: span, path:
                ast::path};

tag unsafe_ty { contains(ty::t); mut_contains(ty::t); }

type binding = @{node_id: node_id,
                 span: span,
                 root_var: option::t<node_id>,
                 local_id: uint,
                 unsafe_tys: [unsafe_ty],
                 mutable copied: copied};

tag ret_info { by_ref(bool, node_id); other; }
// FIXME it may be worthwhile to use a linked list of bindings instead
type scope = {bs: [binding],
              ret_info: ret_info,
              invalid: @mutable list<@invalid>};

fn mk_binding(cx: ctx, id: node_id, span: span, root_var: option::t<node_id>,
              unsafe_tys: [unsafe_ty]) -> binding {
    ret @{node_id: id, span: span, root_var: root_var,
          local_id: local_id_of_node(cx, id),
          unsafe_tys: unsafe_tys,
          mutable copied: not_copied};
}

tag local_info { local(uint); }

type copy_map = std::map::hashmap<node_id, ()>;

type ctx = {tcx: ty::ctxt,
            copy_map: copy_map,
            mutable silent: bool};

fn check_crate(tcx: ty::ctxt, crate: @ast::crate) -> copy_map {
    // Stores information about object fields and function
    // arguments that's otherwise not easily available.
    let cx = @{tcx: tcx,
               copy_map: std::map::new_int_hash(),
               mutable silent: false};
    let v = @{visit_fn: bind visit_fn(cx, _, _, _, _, _, _, _),
              visit_expr: bind visit_expr(cx, _, _, _),
              visit_block: bind visit_block(cx, _, _, _)
              with *visit::default_visitor::<scope>()};
    let sc = {bs: [], ret_info: other, invalid: @mutable list::nil};
    visit::visit_crate(*crate, sc, visit::mk_vt(v));
    tcx.sess.abort_if_errors();
    ret cx.copy_map;
}

fn visit_fn(cx: @ctx, f: ast::_fn, _tp: [ast::ty_param], sp: span,
            _name: fn_ident, id: ast::node_id, sc: scope, v: vt<scope>) {
    visit::visit_fn_decl(f.decl, sc, v);
    let args = ty::ty_fn_args(cx.tcx, ty::node_id_to_type(cx.tcx, id));
    for arg in args {
        if arg.mode == ast::by_val &&
           ty::type_has_dynamic_size(cx.tcx, arg.ty) {
            err(*cx, sp, "can not pass a dynamically-sized type by value");
        }
    }

    if ast_util::ret_by_ref(f.decl.cf) && !is_none(f.body.node.expr) {
        // FIXME this will be easier to lift once have DPS
        err(*cx, option::get(f.body.node.expr).span,
            "reference-returning functions may not return implicitly");
    }
    let ret_info = alt f.decl.cf {
      ast::return_ref(mut, n_arg) {
        by_ref(mut, f.decl.inputs[n_arg - 1u].id)
      }
      _ { other }
    };
    // Blocks need to obey any restrictions from the enclosing scope, and may
    // be called multiple times.
    if f.proto == ast::proto_block {
        let sc = {ret_info: ret_info with sc};
        check_loop(*cx, sc) {|| v.visit_block(f.body, sc, v);}
    } else {
        let sc = {bs: [], ret_info: ret_info, invalid: @mutable list::nil};
        v.visit_block(f.body, sc, v);
    }
}

fn visit_expr(cx: @ctx, ex: @ast::expr, sc: scope, v: vt<scope>) {
    let handled = true;
    alt ex.node {
      ast::expr_call(f, args, _) {
        check_call(*cx, f, args);
        handled = false;
      }
      ast::expr_alt(input, arms) { check_alt(*cx, input, arms, sc, v); }
      ast::expr_for(decl, seq, blk) {
        v.visit_expr(seq, sc, v);
        check_loop(*cx, sc) {|| check_for(*cx, decl, seq, blk, sc, v); }
      }
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
      ast::expr_ret(oexpr) {
        if !is_none(oexpr) {
            alt sc.ret_info {
              by_ref(mut, arg_node_id) {
                check_ret_ref(*cx, sc, mut, arg_node_id, option::get(oexpr));
              }
              _ {}
            }
        }
        handled = false;
      }
      ast::expr_if(c, then, els) { check_if(c, then, els, sc, v); }
      ast::expr_while(_, _) | ast::expr_do_while(_, _) {
        check_loop(*cx, sc) {|| visit::visit_expr(ex, sc, v); }
      }
      _ { handled = false; }
    }
    if !handled { visit::visit_expr(ex, sc, v); }
}

fn visit_block(cx: @ctx, b: ast::blk, sc: scope, v: vt<scope>) {
    let bs = sc.bs, sc = sc;
    for stmt in b.node.stmts {
        alt stmt.node {
          ast::stmt_decl(@{node: ast::decl_item(it), _}, _) {
            v.visit_item(it, sc, v);
          }
          ast::stmt_decl(@{node: ast::decl_local(locs), _}, _) {
            for (st, loc) in locs {
                if st == ast::let_ref {
                    add_bindings_for_let(*cx, bs, loc);
                    sc = {bs: bs with sc};
                }
                alt loc.node.init {
                  some(init) {
                    if init.op == ast::init_move {
                        check_lval(cx, init.expr, sc, v);
                    }
                  }
                  none. { }
                }
            }
          }
          ast::stmt_expr(ex, _) {
            v.visit_expr(ex, sc, v);
          }
          ast::stmt_crate_directive(cd) {
            visit::visit_crate_directive(cd, sc, v);
          }
        }
    }
    visit::visit_expr_opt(b.node.expr, sc, v);
}

fn add_bindings_for_let(cx: ctx, &bs: [binding], loc: @ast::local) {
    alt loc.node.init {
      some(init) {
        if init.op == ast::init_move {
            err(cx, loc.span, "can not move into a by-reference binding");
        }
        let root = expr_root(cx, init.expr, false);
        let root_var = path_def_id(cx, root.ex);
        if is_none(root_var) {
            err(cx, loc.span, "a reference binding can't be \
                               rooted in a temporary");
        }
        for proot in pattern_roots(cx.tcx, root.mut, loc.node.pat) {
            let bnd = mk_binding(cx, proot.id, proot.span, root_var,
                                 unsafe_set(proot.mut));
            // Don't implicitly copy explicit references
            bnd.copied = not_allowed;
            bs += [bnd];
        }
      }
      _ {
        err(cx, loc.span, "by-reference bindings must be initialized");
      }
    }
}


fn cant_copy(cx: ctx, b: binding) -> bool {
    alt b.copied {
      not_allowed. { ret true; }
      copied. { ret false; }
      not_copied. {}
    }
    let ty = ty::node_id_to_type(cx.tcx, b.node_id);
    if ty::type_allows_implicit_copy(cx.tcx, ty) {
        b.copied = copied;
        cx.copy_map.insert(b.node_id, ());
        if copy_is_expensive(cx.tcx, ty) {
            cx.tcx.sess.span_warn(b.span,
                                  "inserting an implicit copy for type " +
                                  util::ppaux::ty_to_str(cx.tcx, ty));
        }
        ret false;
    } else { ret true; }
}

fn check_call(cx: ctx, f: @ast::expr, args: [@ast::expr]) -> [binding] {
    let fty = ty::expr_ty(cx.tcx, f);
    let by_ref = alt ty::ty_fn_ret_style(cx.tcx, fty) {
      ast::return_ref(_, arg_n) { arg_n } _ { 0u }
    };
    let arg_ts = ty::ty_fn_args(cx.tcx, fty);
    let mut_roots: [{arg: uint, node: node_id}] = [];
    let bindings = [];
    let i = 0u;
    for arg_t: ty::arg in arg_ts {
        let arg = args[i];
        let root = expr_root(cx, arg, false);
        if arg_t.mode == ast::by_mut_ref {
            alt path_def(cx, arg) {
              some(def) {
                let dnum = ast_util::def_id_of_def(def).node;
                mut_roots += [{arg: i, node: dnum}];
              }
              _ { }
            }
        }
        let root_var = path_def_id(cx, root.ex);
        bindings += [@{node_id: arg.id,
                       span: arg.span,
                       root_var: root_var,
                       local_id: 0u,
                       unsafe_tys: unsafe_set(root.mut),
                       mutable copied: alt arg_t.mode {
                         ast::by_move. { copied }
                         ast::by_mut_ref. { not_allowed }
                         _ { i + 1u == by_ref ? not_allowed : not_copied }
                       }}];
        i += 1u;
    }
    let f_may_close =
        alt f.node {
          ast::expr_path(_) { def_is_local(cx.tcx.def_map.get(f.id), true) }
          _ { true }
        };
    if f_may_close {
        let i = 0u;
        for b in bindings {
            if vec::len(b.unsafe_tys) > 0u && cant_copy(cx, b) {
                err(cx, f.span, #fmt["function may alias with argument \
                                     %u, which is not immutably rooted", i]);
            }
            i += 1u;
        }
    }
    let j = 0u;
    for b in bindings {
        for unsafe_ty in b.unsafe_tys {
            let i = 0u;
            for arg_t: ty::arg in arg_ts {
                let mut_alias = arg_t.mode == ast::by_mut_ref;
                if i != j &&
                       ty_can_unsafely_include(cx, unsafe_ty, arg_t.ty,
                                               mut_alias) &&
                       cant_copy(cx, b) {
                    err(cx, args[i].span,
                        #fmt["argument %u may alias with argument %u, \
                             which is not immutably rooted", i, j]);
                }
                i += 1u;
            }
        }
        j += 1u;
    }
    // Ensure we're not passing a root by mutable alias.

    for {node: node, arg: arg} in mut_roots {
        let i = 0u;
        for b in bindings {
            if i != arg {
                alt b.root_var {
                  some(root) {
                    if node == root && cant_copy(cx, b) {
                        err(cx, args[arg].span,
                            "passing a mutable reference to a \
                             variable that roots another reference");
                        break;
                    }
                  }
                  none. { }
                }
            }
            i += 1u;
        }
    }
    ret bindings;
}

fn check_ret_ref(cx: ctx, sc: scope, mut: bool, arg_node_id: node_id,
                 expr: @ast::expr) {
    let root = expr_root(cx, expr, false);
    let bad = none;
    let mut_field = !is_none(root.mut);
    alt path_def(cx, root.ex) {
      none. {
        bad = some("a temporary");
      }
      some(ast::def_local(did, _)) | some(ast::def_binding(did)) |
      some(ast::def_arg(did, _)) {
        let cur_node = did.node;
        while true {
            alt cx.tcx.items.find(cur_node) {
              some(ast_map::node_arg(arg, _)) {
                if arg.mode == ast::by_move {
                    bad = some("a move-mode parameter");
                }
                if cur_node != arg_node_id {
                    bad = some("the wrong parameter");
                }
                break;
              }
              _ {}
            }
            alt vec::find({|b| b.node_id == cur_node}, sc.bs) {
              some(b) {
                if vec::len(b.unsafe_tys) > 0u {
                    mut_field = true;
                    break;
                }
                if is_none(b.root_var) {
                    bad = some("a function-local value");
                    break;
                }
                if b.copied == copied {
                    bad = some("an implicitly copied reference");
                    break;
                }
                b.copied = not_allowed;
                cur_node = option::get(b.root_var);
              }
              none. {
                bad = some("a function-local value");
                break;
              }
            }
        }
      }
      _ { bad = some("a non-local value"); }
    }
    if mut_field && !mut { bad = some("a mutable field"); }
    alt bad {
      some(name) {
        err(cx, expr.span, "can not return a reference to " + name);
      }
      _ {}
    }
}

fn check_alt(cx: ctx, input: @ast::expr, arms: [ast::arm], sc: scope,
             v: vt<scope>) {
    v.visit_expr(input, sc, v);
    let orig_invalid = *sc.invalid;
    let all_invalid = orig_invalid;
    let root = expr_root(cx, input, true);
    for a: ast::arm in arms {
        let new_bs = sc.bs;
        let root_var = path_def_id(cx, root.ex);
        let pat_id_map = ast_util::pat_id_map(a.pats[0]);
        type info = {
            id: node_id,
            mutable unsafe_tys: [unsafe_ty],
            span: span};
        let binding_info: [info] = [];
        for pat in a.pats {
            for proot in pattern_roots(cx.tcx, root.mut, pat) {
                let canon_id = pat_id_map.get(proot.name);
                alt vec::find({|x| x.id == canon_id}, binding_info) {
                  some(s) { s.unsafe_tys += unsafe_set(proot.mut); }
                  none. {
                      binding_info += [
                          {id: canon_id,
                           mutable unsafe_tys: unsafe_set(proot.mut),
                           span: proot.span}];
                  }
                }
            }
        }
        for info in binding_info {
            new_bs += [mk_binding(cx, info.id, info.span, root_var,
                                  copy info.unsafe_tys)];
        }
        *sc.invalid = orig_invalid;
        visit::visit_arm(a, {bs: new_bs with sc}, v);
        all_invalid = append_invalid(all_invalid, *sc.invalid, orig_invalid);
    }
    *sc.invalid = all_invalid;
}

fn check_for(cx: ctx, local: @ast::local, seq: @ast::expr, blk: ast::blk,
             sc: scope, v: vt<scope>) {
    let root = expr_root(cx, seq, false);

    // If this is a mutable vector, don't allow it to be touched.
    let seq_t = ty::expr_ty(cx.tcx, seq);
    let cur_mut = root.mut;
    alt ty::struct(cx.tcx, seq_t) {
      ty::ty_vec(mt) {
        if mt.mut != ast::imm {
            cur_mut = some(contains(seq_t));
        }
      }
      _ {}
    }
    let root_var = path_def_id(cx, root.ex);
    let new_bs = sc.bs;
    for proot in pattern_roots(cx.tcx, cur_mut, local.node.pat) {
        new_bs += [mk_binding(cx, proot.id, proot.span, root_var,
                              unsafe_set(proot.mut))];
    }
    visit::visit_block(blk, {bs: new_bs with sc}, v);
}

fn check_var(cx: ctx, ex: @ast::expr, p: ast::path, id: ast::node_id,
             assign: bool, sc: scope) {
    let def = cx.tcx.def_map.get(id);
    if !def_is_local(def, false) { ret; }
    let my_defnum = ast_util::def_id_of_def(def).node;
    let my_local_id = local_id_of_node(cx, my_defnum);
    let var_t = ty::expr_ty(cx.tcx, ex);
    for b in sc.bs {
        // excludes variables introduced since the alias was made
        if my_local_id < b.local_id {
            for unsafe_ty in b.unsafe_tys {
                if ty_can_unsafely_include(cx, unsafe_ty, var_t, assign) {
                    let inv = @{reason: val_taken, node_id: b.node_id,
                                sp: ex.span, path: p};
                    *sc.invalid = list::cons(inv, @*sc.invalid);
                }
            }
        } else if b.node_id == my_defnum {
            test_scope(cx, sc, b, p);
        }
    }
}

fn check_lval(cx: @ctx, dest: @ast::expr, sc: scope, v: vt<scope>) {
    alt dest.node {
      ast::expr_path(p) {
        let def = cx.tcx.def_map.get(dest.id);
        let dnum = ast_util::def_id_of_def(def).node;
        for b in sc.bs {
            if b.root_var == some(dnum) {
                let inv = @{reason: overwritten, node_id: b.node_id,
                            sp: dest.span, path: p};
                *sc.invalid = list::cons(inv, @*sc.invalid);
            }
        }
      }
      _ { visit_expr(cx, dest, sc, v); }
    }
}

fn check_assign(cx: @ctx, dest: @ast::expr, src: @ast::expr, sc: scope,
                v: vt<scope>) {
    visit_expr(cx, src, sc, v);
    check_lval(cx, dest, sc, v);
}

fn check_if(c: @ast::expr, then: ast::blk, els: option::t<@ast::expr>,
            sc: scope, v: vt<scope>) {
    v.visit_expr(c, sc, v);
    let orig_invalid = *sc.invalid;
    v.visit_block(then, sc, v);
    let then_invalid = *sc.invalid;
    *sc.invalid = orig_invalid;
    visit::visit_expr_opt(els, sc, v);
    *sc.invalid = append_invalid(*sc.invalid, then_invalid, orig_invalid);
}

fn check_loop(cx: ctx, sc: scope, checker: block()) {
    let orig_invalid = filter_invalid(*sc.invalid, sc.bs);
    checker();
    let new_invalid = filter_invalid(*sc.invalid, sc.bs);
    // Have to check contents of loop again if it invalidated an alias
    if list::len(orig_invalid) < list::len(new_invalid) {
        let old_silent = cx.silent;
        cx.silent = true;
        checker();
        cx.silent = old_silent;
    }
    *sc.invalid = new_invalid;
}

fn test_scope(cx: ctx, sc: scope, b: binding, p: ast::path) {
    let prob = find_invalid(b.node_id, *sc.invalid);
    alt b.root_var {
      some(dn) {
        for other in sc.bs {
            if !is_none(prob) { break; }
            if other.node_id == dn {
                prob = find_invalid(other.node_id, *sc.invalid);
            }
        }
      }
      _ {}
    }
    if !is_none(prob) && cant_copy(cx, b) {
        let i = option::get(prob);
        let msg = alt i.reason {
          overwritten. { "overwriting " + ast_util::path_name(i.path) }
          val_taken. { "taking the value of " + ast_util::path_name(i.path) }
        };
        err(cx, i.sp, msg + " will invalidate reference " +
            ast_util::path_name(p) + ", which is still used");
    }
}

fn path_def(cx: ctx, ex: @ast::expr) -> option::t<ast::def> {
    ret alt ex.node {
          ast::expr_path(_) { some(cx.tcx.def_map.get(ex.id)) }
          _ { none }
        }
}

fn path_def_id(cx: ctx, ex: @ast::expr) -> option::t<ast::node_id> {
    alt ex.node {
      ast::expr_path(_) {
        ret some(ast_util::def_id_of_def(cx.tcx.def_map.get(ex.id)).node);
      }
      _ { ret none; }
    }
}

fn ty_can_unsafely_include(cx: ctx, needle: unsafe_ty, haystack: ty::t,
                           mut: bool) -> bool {
    fn get_mut(cur: bool, mt: ty::mt) -> bool {
        ret cur || mt.mut != ast::imm;
    }
    fn helper(tcx: ty::ctxt, needle: unsafe_ty, haystack: ty::t, mut: bool)
        -> bool {
        if alt needle {
          contains(ty) { ty == haystack }
          mut_contains(ty) { mut && ty == haystack }
        } { ret true; }
        alt ty::struct(tcx, haystack) {
          ty::ty_tag(_, ts) {
            for t: ty::t in ts {
                if helper(tcx, needle, t, mut) { ret true; }
            }
            ret false;
          }
          ty::ty_box(mt) | ty::ty_ptr(mt) | ty::ty_uniq(mt) {
            ret helper(tcx, needle, mt.ty, get_mut(mut, mt));
          }
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
          ty::ty_fn(ast::proto_bare., _, _, _, _) { ret false; }
          // These may contain anything.
          ty::ty_fn(_, _, _, _, _) | ty::ty_obj(_) { ret true; }
          // A type param may include everything, but can only be
          // treated as opaque downstream, and is thus safe unless we
          // saw mutable fields, in which case the whole thing can be
          // overwritten.
          ty::ty_param(_, _) { ret mut; }
          _ { ret false; }
        }
    }
    ret helper(cx.tcx, needle, haystack, mut);
}

fn def_is_local(d: ast::def, objfields_count: bool) -> bool {
    ret alt d {
          ast::def_local(_, _) | ast::def_arg(_, _) | ast::def_binding(_) |
          ast::def_upvar(_, _, _) {
            true
          }
          ast::def_obj_field(_, _) { objfields_count }
          _ { false }
        };
}

fn local_id_of_node(cx: ctx, id: node_id) -> uint {
    alt cx.tcx.items.get(id) {
      ast_map::node_arg(_, id) | ast_map::node_local(id) { id }
    }
}

// Heuristic, somewhat random way to decide whether to warn when inserting an
// implicit copy.
fn copy_is_expensive(tcx: ty::ctxt, ty: ty::t) -> bool {
    fn score_ty(tcx: ty::ctxt, ty: ty::t) -> uint {
        ret alt ty::struct(tcx, ty) {
          ty::ty_nil. | ty::ty_bot. | ty::ty_bool. | ty::ty_int. |
          ty::ty_uint. | ty::ty_float. | ty::ty_machine(_) |
          ty::ty_char. | ty::ty_type. | ty::ty_native(_) |
          ty::ty_ptr(_) { 1u }
          ty::ty_box(_) { 3u }
          ty::ty_constr(t, _) | ty::ty_res(_, t, _) { score_ty(tcx, t) }
          ty::ty_fn(_, _, _, _, _) | ty::ty_native_fn(_, _, _) |
          ty::ty_obj(_) { 4u }
          ty::ty_str. | ty::ty_vec(_) | ty::ty_param(_, _) { 50u }
          ty::ty_uniq(mt) { 1u + score_ty(tcx, mt.ty) }
          ty::ty_tag(_, ts) | ty::ty_tup(ts) {
            let sum = 0u;
            for t in ts { sum += score_ty(tcx, t); }
            sum
          }
          ty::ty_rec(fs) {
            let sum = 0u;
            for f in fs { sum += score_ty(tcx, f.mt.ty); }
            sum
          }
        };
    }
    ret score_ty(tcx, ty) > 8u;
}

type pattern_root = {id: node_id,
                     name: ident,
                     mut: option::t<unsafe_ty>,
                     span: span};

fn pattern_roots(tcx: ty::ctxt, mut: option::t<unsafe_ty>, pat: @ast::pat)
    -> [pattern_root] {
    fn walk(tcx: ty::ctxt, mut: option::t<unsafe_ty>, pat: @ast::pat,
            &set: [pattern_root]) {
        alt pat.node {
          ast::pat_wild. | ast::pat_lit(_) | ast::pat_range(_, _) {}
          ast::pat_bind(nm) {
            set += [{id: pat.id, name: nm, mut: mut, span: pat.span}];
          }
          ast::pat_tag(_, ps) | ast::pat_tup(ps) {
            for p in ps { walk(tcx, mut, p, set); }
          }
          ast::pat_rec(fs, _) {
            let ty = ty::node_id_to_type(tcx, pat.id);
            for f in fs {
                let m = ty::get_field(tcx, ty, f.ident).mt.mut != ast::imm;
                walk(tcx, m ? some(contains(ty)) : mut, f.pat, set);
            }
          }
          ast::pat_box(p) {
            let ty = ty::node_id_to_type(tcx, pat.id);
            let m = alt ty::struct(tcx, ty) {
              ty::ty_box(mt) { mt.mut != ast::imm }
            };
            walk(tcx, m ? some(contains(ty)) : mut, p, set);
          }
          ast::pat_uniq(p) {
            let ty = ty::node_id_to_type(tcx, pat.id);
            let m = alt ty::struct(tcx, ty) {
              ty::ty_uniq(mt) { mt.mut != ast::imm }
            };
            walk(tcx, m ? some(contains(ty)) : mut, p, set);
          }
        }
    }
    let set = [];
    walk(tcx, mut, pat, set);
    ret set;
}

// Wraps the expr_root in mut.rs to also handle roots that exist through
// return-by-reference
fn expr_root(cx: ctx, ex: @ast::expr, autoderef: bool)
    -> {ex: @ast::expr, mut: option::t<unsafe_ty>} {
    let base_root = mut::expr_root(cx.tcx, ex, autoderef);
    let unsafe_ty = none;
    for d in *base_root.ds {
        if d.mut { unsafe_ty = some(contains(d.outer_t)); break; }
    }
    alt base_root.ex.node {
      ast::expr_path(_) {
        alt cx.tcx.def_map.get(base_root.ex.id) {
          ast::def_obj_field(_, ast::mut.) {
            unsafe_ty = some(mut_contains(ty::expr_ty(cx.tcx, base_root.ex)));
          }
          _ {}
        }
      }
      ast::expr_call(f, args, _) {
        let fty = ty::expr_ty(cx.tcx, f);
        alt ty::ty_fn_ret_style(cx.tcx, fty) {
          ast::return_ref(mut, arg_n) {
            let arg = args[arg_n - 1u];
            let arg_root = expr_root(cx, arg, false);
            if mut {
                let ret_ty = ty::expr_ty(cx.tcx, base_root.ex);
                unsafe_ty = some(mut_contains(ret_ty));
            }
            if !is_none(arg_root.mut) { unsafe_ty = arg_root.mut; }
            ret {ex: arg_root.ex, mut: unsafe_ty};
          }
          _ {}
        }
      }
      _ {}
    }
    ret {ex: base_root.ex, mut: unsafe_ty};
}

fn unsafe_set(from: option::t<unsafe_ty>) -> [unsafe_ty] {
    alt from { some(t) { [t] } _ { [] } }
}

fn find_invalid(id: node_id, lst: list<@invalid>)
    -> option::t<@invalid> {
    let cur = lst;
    while true {
        alt cur {
          list::nil. { break; }
          list::cons(head, tail) {
            if head.node_id == id { ret some(head); }
            cur = *tail;
          }
        }
    }
    ret none;
}

fn append_invalid(dest: list<@invalid>, src: list<@invalid>,
                  stop: list<@invalid>) -> list<@invalid> {
    let cur = src, dest = dest;
    while cur != stop {
        alt cur {
          list::cons(head, tail) {
            if is_none(find_invalid(head.node_id, dest)) {
                dest = list::cons(head, @dest);
            }
            cur = *tail;
          }
        }
    }
    ret dest;
}

fn filter_invalid(src: list<@invalid>, bs: [binding]) -> list<@invalid> {
    let out = list::nil, cur = src;
    while cur != list::nil {
        alt cur {
          list::cons(head, tail) {
            let p = vec::position_pred({|b| b.node_id == head.node_id}, bs);
            if !is_none(p) { out = list::cons(head, @out); }
            cur = *tail;
          }
        }
    }
    ret out;
}

fn err(cx: ctx, sp: span, err: str) {
    if !cx.silent || !cx.tcx.sess.has_errors() {
        cx.tcx.sess.span_err(sp, err);
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
