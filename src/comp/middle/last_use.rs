import syntax::{visit, ast_util};
import syntax::ast::*;
import syntax::codemap::span;
import std::list::{is_not_empty, list, nil, cons, tail};
import core::{vec, option};
import std::list;

// Last use analysis pass.
//
// Finds the last read of each value stored in a local variable or
// callee-owned argument (arguments with by-move or by-copy passing
// style). This is a limited form of liveness analysis, peformed
// (perhaps foolishly) directly on the AST.
//
// The algorithm walks the AST, keeping a set of (def, last_use)
// pairs. When the function is exited, or the local is overwritten,
// the current set of last uses is marked with 'true' in a table.
// Other branches may later overwrite them with 'false' again, since
// they may find a use coming after them. (Marking an expression as a
// last use is only done if it has not already been marked with
// 'false'.)
//
// Some complexity is added to deal with joining control flow branches
// (by `break` or conditionals), and for handling loops.

// Marks expr_paths that are last uses.
type last_uses = std::map::hashmap<node_id, ()>;

enum seen { unset, seen(node_id), }
enum block_type { func, loop, }

type set = [{def: node_id, exprs: list<node_id>}];
type bl = @{type: block_type, mutable second: bool, mutable exits: [set]};

type ctx = {last_uses: std::map::hashmap<node_id, bool>,
            def_map: resolve::def_map,
            ref_map: alias::ref_map,
            tcx: ty::ctxt,
            // The current set of local last uses
            mutable current: set,
            mutable blocks: list<bl>};

fn find_last_uses(c: @crate, def_map: resolve::def_map,
                  ref_map: alias::ref_map, tcx: ty::ctxt) -> last_uses {
    let v = visit::mk_vt(@{visit_expr: visit_expr,
                           visit_fn: visit_fn
                           with *visit::default_visitor()});
    let cx = {last_uses: std::map::new_int_hash(),
              def_map: def_map,
              ref_map: ref_map,
              tcx: tcx,
              mutable current: [],
              mutable blocks: nil};
    visit::visit_crate(*c, cx, v);
    let mini_table = std::map::new_int_hash();
    cx.last_uses.items {|key, val|
        if val {
            mini_table.insert(key, ());
            let def_node = ast_util::def_id_of_def(def_map.get(key)).node;
            mini_table.insert(def_node, ());
        }
    }
    ret mini_table;
}

fn ex_is_blockish(cx: ctx, id: node_id) -> bool {
    alt ty::struct(cx.tcx, ty::node_id_to_type(cx.tcx, id)) {
      ty::ty_fn({proto: p, _}) if is_blockish(p) { true }
      _ { false }
    }
}

fn visit_expr(ex: @expr, cx: ctx, v: visit::vt<ctx>) {
    alt ex.node {
      expr_ret(oexpr) {
        visit::visit_expr_opt(oexpr, cx, v);
        if !add_block_exit(cx, func) { leave_fn(cx); }
      }
      expr_fail(oexpr) {
        visit::visit_expr_opt(oexpr, cx, v);
        leave_fn(cx);
      }
      expr_break { add_block_exit(cx, loop); }
      expr_while(_, _) | expr_do_while(_, _) {
        visit_block(loop, cx) {|| visit::visit_expr(ex, cx, v);}
      }
      expr_for(_, coll, blk) {
        v.visit_expr(coll, cx, v);
        visit_block(loop, cx) {|| visit::visit_block(blk, cx, v);}
      }
      expr_alt(input, arms) {
        v.visit_expr(input, cx, v);
        let before = cx.current, sets = [];
        for arm in arms {
            cx.current = before;
            v.visit_arm(arm, cx, v);
            sets += [cx.current];
        }
        cx.current = join_branches(sets);
      }
      expr_if(cond, then, els) {
        v.visit_expr(cond, cx, v);
        let cur = cx.current;
        visit::visit_block(then, cx, v);
        cx.current <-> cur;
        visit::visit_expr_opt(els, cx, v);
        cx.current = join_branches([cur, cx.current]);
      }
      expr_path(_) {
        let my_def = ast_util::def_id_of_def(cx.def_map.get(ex.id)).node;
        alt cx.ref_map.find(my_def) {
          option::some(root_id) { clear_in_current(cx, root_id, false); }
          _ {
            alt clear_if_path(cx, ex, v, false) {
              option::some(my_def) {
                cx.current += [{def: my_def, exprs: cons(ex.id, @nil)}];
              }
              _ {}
            }
          }
        }
      }
      expr_swap(lhs, rhs) {
        clear_if_path(cx, lhs, v, false);
        clear_if_path(cx, rhs, v, false);
      }
      expr_move(dest, src) | expr_assign(dest, src) {
        v.visit_expr(src, cx, v);
        clear_if_path(cx, dest, v, true);
      }
      expr_assign_op(_, dest, src) {
        v.visit_expr(src, cx, v);
        v.visit_expr(dest, cx, v);
        clear_if_path(cx, dest, v, true);
      }
      expr_fn(_, _, _, cap_clause) {
        // n.b.: safe to ignore copies, as if they are unused
        // then they are ignored, otherwise they will show up
        // as freevars in the body.

        vec::iter(cap_clause.moves) {|ci|
            clear_def_if_path(cx, cx.def_map.get(ci.id), true);
        }
        visit::visit_expr(ex, cx, v);
      }
      expr_call(f, args, _) {
        v.visit_expr(f, cx, v);
        let i = 0u, fns = [];
        let arg_ts = ty::ty_fn_args(cx.tcx, ty::expr_ty(cx.tcx, f));
        for arg in args {
            alt arg.node {
              expr_fn(p, _, _, _) if is_blockish(p) {
                fns += [arg];
              }
              expr_fn_block(_, _) if ex_is_blockish(cx, arg.id) {
                fns += [arg];
              }
              _ {
                alt arg_ts[i].mode {
                  by_mut_ref { clear_if_path(cx, arg, v, false); }
                  _ { v.visit_expr(arg, cx, v); }
                }
              }
            }
            i += 1u;
        }
        for f in fns { v.visit_expr(f, cx, v); }
      }
      _ { visit::visit_expr(ex, cx, v); }
    }
}

fn visit_fn(fk: visit::fn_kind, decl: fn_decl, body: blk,
            sp: span, id: node_id,
            cx: ctx, v: visit::vt<ctx>) {
    let fty = ty::node_id_to_type(cx.tcx, id);
    let proto = ty::ty_fn_proto(cx.tcx, fty);
    alt proto {
      proto_any | proto_block {
        visit_block(func, cx, {||
            visit::visit_fn(fk, decl, body, sp, id, cx, v);
        });
      }
      proto_box | proto_uniq | proto_bare {
        alt cx.tcx.freevars.find(id) {
          some(vars) {
            for v in *vars {
                clear_in_current(cx, ast_util::def_id_of_def(v.def).node,
                                 false);
            }
          }
          _ {}
        }
        let old = nil;
        cx.blocks <-> old;
        visit::visit_fn(fk, decl, body, sp, id, cx, v);
        cx.blocks <-> old;
        leave_fn(cx);
      }
    }
}

fn visit_block(tp: block_type, cx: ctx, visit: fn()) {
    let local = @{type: tp, mutable second: false, mutable exits: []};
    cx.blocks = cons(local, @cx.blocks);
    visit();
    local.second = true;
    visit();
    let cx_blocks = cx.blocks;
    check is_not_empty(cx_blocks);
    cx.blocks = tail(cx_blocks);
    cx.current = join_branches(local.exits);
}

fn add_block_exit(cx: ctx, tp: block_type) -> bool {
    let cur = cx.blocks;
    while cur != nil {
        alt cur {
          cons(b, tail) {
            if (b.type == tp) {
                if !b.second { b.exits += [cx.current]; }
                ret true;
            }
            cur = *tail;
          }
        }
    }
    ret false;
}

fn join_branches(branches: [set]) -> set {
    let found: set = [], i = 0u, l = vec::len(branches);
    for set in branches {
        i += 1u;
        for {def, exprs} in set {
            if !vec::any(found, {|v| v.def == def}) {
                let j = i, nne = exprs;
                while j < l {
                    for {def: d2, exprs} in branches[j] {
                        if d2 == def {
                            list::iter(exprs) {|e|
                                if !list::has(nne, e) { nne = cons(e, @nne); }
                            }
                        }
                    }
                    j += 1u;
                }
                found += [{def: def, exprs: nne}];
            }
        }
    }
    ret found;
}

fn leave_fn(cx: ctx) {
    for {def, exprs} in cx.current {
        list::iter(exprs) {|ex_id|
            if !cx.last_uses.contains_key(ex_id) {
                cx.last_uses.insert(ex_id, true);
            }
        }
    }
}

fn clear_in_current(cx: ctx, my_def: node_id, to: bool) {
    for {def, exprs} in cx.current {
        if def == my_def {
            list::iter(exprs) {|expr|
                if !to || !cx.last_uses.contains_key(expr) {
                     cx.last_uses.insert(expr, to);
                }
            }
            cx.current = vec::filter(copy cx.current,
                                     {|x| x.def != my_def});
            break;
        }
    }
}

fn clear_def_if_path(cx: ctx, d: def, to: bool)
    -> option<node_id> {
    alt d {
      def_local(def_id, let_copy) | def_arg(def_id, by_copy) |
      def_arg(def_id, by_move) {
        clear_in_current(cx, def_id.node, to);
        some(def_id.node)
      }
      _ {
        none
      }
    }
}

fn clear_if_path(cx: ctx, ex: @expr, v: visit::vt<ctx>, to: bool)
    -> option::t<node_id> {
    alt ex.node {
      expr_path(_) {
        ret clear_def_if_path(cx, cx.def_map.get(ex.id), to);
      }
      _ { v.visit_expr(ex, cx, v); }
    }
    ret option::none;
}
