import syntax::ast::*;
import syntax::visit;
import syntax::ast_util;
import driver::session::session;
import std::map::hashmap;

enum deref_t { unbox(bool), field, index, }

type deref = @{mutbl: bool, kind: deref_t, outer_t: ty::t};

// Finds the root (the thing that is dereferenced) for the given expr, and a
// vec of dereferences that were used on this root. Note that, in this vec,
// the inner derefs come in front, so foo.bar[1] becomes rec(ex=foo,
// ds=[index,field])
fn expr_root(cx: @ctx, ex: @expr, autoderef: bool)
    -> {ex: @expr, ds: @[deref]} {
    expr_root_(cx.tcx, cx.in_ctor, ex, autoderef)
}

fn expr_root_(tcx: ty::ctxt, ctor_self: option<node_id>,
             ex: @expr, autoderef: bool) ->
   {ex: @expr, ds: @[deref]} {
    fn maybe_auto_unbox(tcx: ty::ctxt, t: ty::t) -> {t: ty::t, ds: [deref]} {
        let mut ds = [], t = t;
        loop {
            alt ty::get(t).struct {
              ty::ty_box(mt) | ty::ty_uniq(mt) | ty::ty_rptr(_, mt) {
                ds += [@{mutbl: mt.mutbl == m_mutbl,
                         kind: unbox(false),
                         outer_t: t}];
                t = mt.ty;
              }
              ty::ty_res(_, inner, tps) {
                ds += [@{mutbl: false, kind: unbox(false), outer_t: t}];
                t = ty::substitute_type_params(tcx, tps, inner);
              }
              ty::ty_enum(did, tps) {
                let variants = ty::enum_variants(tcx, did);
                if vec::len(*variants) != 1u ||
                       vec::len(variants[0].args) != 1u {
                    break;
                }
                ds += [@{mutbl: false, kind: unbox(false), outer_t: t}];
                t = ty::substitute_type_params(tcx, tps, variants[0].args[0]);
              }
              _ { break; }
            }
        }
        ret {t: t, ds: ds};
    }
    let mut ds: [deref] = [], ex = ex;
    loop {
        alt copy ex.node {
          expr_field(base, ident, _) {
            let auto_unbox = maybe_auto_unbox(tcx, ty::expr_ty(tcx, base));
            let mut is_mutbl = false;
            alt ty::get(auto_unbox.t).struct {
              ty::ty_rec(fields) {
                for fields.each {|fld|
                    if str::eq(ident, fld.ident) {
                        is_mutbl = fld.mt.mutbl == m_mutbl;
                        break;
                    }
                }
              }
              ty::ty_class(did, _) {
                  util::common::log_expr(*base);
                  let in_self = alt ctor_self {
                          some(selfid) {
                              alt tcx.def_map.find(base.id) {
                                 some(def_self(slfid)) { slfid == selfid }
                                 _ { false }
                              }
                          }
                          none { false }
                  };
                  for ty::lookup_class_fields(tcx, did).each {|fld|
                    if str::eq(ident, fld.ident) {
                        is_mutbl = fld.mutability == class_mutable
                            || in_self; // all fields can be mutated
                                        // in the ctor
                        break;
                    }
                  }
              }
              _ {}
            }
            ds += [@{mutbl: is_mutbl, kind: field, outer_t: auto_unbox.t}];
            ds += auto_unbox.ds;
            ex = base;
          }
          expr_index(base, _) {
            let auto_unbox = maybe_auto_unbox(tcx, ty::expr_ty(tcx, base));
            alt ty::get(auto_unbox.t).struct {
              ty::ty_vec(mt) {
                ds +=
                    [@{mutbl: mt.mutbl == m_mutbl,
                       kind: index,
                       outer_t: auto_unbox.t}];
              }
              ty::ty_str {
                ds += [@{mutbl: false, kind: index, outer_t: auto_unbox.t}];
              }
              _ { break; }
            }
            ds += auto_unbox.ds;
            ex = base;
          }
          expr_unary(op, base) {
            if op == deref {
                let base_t = ty::expr_ty(tcx, base);
                let mut is_mutbl = false, ptr = false;
                alt ty::get(base_t).struct {
                  ty::ty_box(mt) { is_mutbl = mt.mutbl == m_mutbl; }
                  ty::ty_uniq(mt) { is_mutbl = mt.mutbl == m_mutbl; }
                  ty::ty_res(_, _, _) { }
                  ty::ty_enum(_, _) { }
                  ty::ty_ptr(mt) | ty::ty_rptr(_, mt) {
                    is_mutbl = mt.mutbl == m_mutbl;
                    ptr = true;
                  }
                  _ { tcx.sess.span_bug(base.span, "ill-typed base \
                        expression in deref"); }
                }
                ds += [@{mutbl: is_mutbl, kind: unbox(ptr && is_mutbl),
                         outer_t: base_t}];
                ex = base;
            } else { break; }
          }
          _ { break; }
        }
    }
    if autoderef {
        let auto_unbox = maybe_auto_unbox(tcx, ty::expr_ty(tcx, ex));
        ds += auto_unbox.ds;
    }
    ret {ex: ex, ds: @ds};
}

// Actual mutbl-checking pass

type mutbl_map = std::map::hashmap<node_id, ()>;
// Keep track of whether we're inside a ctor, so as to
// allow mutating immutable fields in the same class
// if we are in a ctor, we track the self id
type ctx = {tcx: ty::ctxt, mutbl_map: mutbl_map, in_ctor: option<node_id>};

fn check_crate(tcx: ty::ctxt, crate: @crate) -> mutbl_map {
    let cx = @{tcx: tcx, mutbl_map: std::map::int_hash(), in_ctor: none};
    let v = @{visit_expr: visit_expr,
              visit_decl: visit_decl,
              visit_item: visit_item
              with *visit::default_visitor()};
    visit::visit_crate(*crate, cx, visit::mk_vt(v));
    ret cx.mutbl_map;
}

enum msg { msg_assign, msg_move_out, msg_mutbl_ref, }

fn mk_err(cx: @ctx, span: syntax::codemap::span, msg: msg, name: str) {
    cx.tcx.sess.span_err(span, alt msg {
      msg_assign { "assigning to " + name }
      msg_move_out { "moving out of " + name }
      msg_mutbl_ref { "passing " + name + " by mut reference" }
    });
}

fn visit_decl(d: @decl, &&cx: @ctx, v: visit::vt<@ctx>) {
    visit::visit_decl(d, cx, v);
    alt d.node {
      decl_local(locs) {
        for locs.each {|loc|
            alt loc.node.init {
              some(init) {
                if init.op == init_move { check_move_rhs(cx, init.expr); }
              }
              none { }
            }
        }
      }
      _ { }
    }
}

fn visit_expr(ex: @expr, &&cx: @ctx, v: visit::vt<@ctx>) {
    alt ex.node {
      expr_call(f, args, _) { check_call(cx, f, args); }
      expr_bind(f, args) { check_bind(cx, f, args); }
      expr_swap(lhs, rhs) {
        check_lval(cx, lhs, msg_assign);
        check_lval(cx, rhs, msg_assign);
      }
      expr_move(dest, src) {
        check_lval(cx, dest, msg_assign);
        check_move_rhs(cx, src);
      }
      expr_assign(dest, src) | expr_assign_op(_, dest, src) {
        check_lval(cx, dest, msg_assign);
      }
      expr_fn(_, _, _, cap) {
        for cap.moves.each {|moved|
            let def = cx.tcx.def_map.get(moved.id);
            alt is_illegal_to_modify_def(cx, def, msg_move_out) {
              some(name) { mk_err(cx, moved.span, msg_move_out, moved.name); }
              _ { }
            }
            cx.mutbl_map.insert(ast_util::def_id_of_def(def).node, ());
        }
      }
      _ { }
    }
    visit::visit_expr(ex, cx, v);
}

fn visit_item(item: @item, &&cx: @ctx, v: visit::vt<@ctx>) {
    alt item.node {
      item_class(tps, items, ctor) {
         v.visit_ty_params(tps, cx, v);
         vec::map::<@class_member, ()>(items,
             {|i| v.visit_class_item(i, cx, v); });
         visit::visit_class_ctor_helper(ctor, item.ident, tps,
                                        ast_util::local_def(item.id),
                    @{in_ctor: some(ctor.node.self_id) with *cx}, v);
      }
      _ { visit::visit_item(item, cx, v); }
    }
}

fn check_lval(cx: @ctx, dest: @expr, msg: msg) {
    alt dest.node {
      expr_path(p) {
        let def = cx.tcx.def_map.get(dest.id);
        alt is_illegal_to_modify_def(cx, def, msg) {
          some(name) { mk_err(cx, dest.span, msg, name); }
          _ { }
        }
        cx.mutbl_map.insert(ast_util::def_id_of_def(def).node, ());
      }
      _ {
        let root = expr_root(cx, dest, false);
        if vec::len(*root.ds) == 0u {
            if msg != msg_move_out {
                mk_err(cx, dest.span, msg, "non-lvalue");
            }
        } else if !root.ds[0].mutbl {
            let name =
                alt root.ds[0].kind {
                  mutbl::unbox(_) { "immutable box" }
                  mutbl::field { "immutable field" }
                  mutbl::index { "immutable vec content" }
                };
            mk_err(cx, dest.span, msg, name);
        }
      }
    }
}

fn check_move_rhs(cx: @ctx, src: @expr) {
    alt src.node {
      expr_path(p) {
        alt cx.tcx.def_map.get(src.id) {
          def_self(_) {
            mk_err(cx, src.span, msg_move_out, "method self");
          }
          _ { }
        }
        check_lval(cx, src, msg_move_out);
      }
      _ {
        let root = expr_root(cx, src, false);

        // Not a path and no-derefs means this is a temporary.
        if vec::len(*root.ds) != 0u &&
           root.ds[vec::len(*root.ds) - 1u].kind != unbox(true) {
            cx.tcx.sess.span_err(src.span, "moving out of a data structure");
        }
      }
    }
}

fn check_call(cx: @ctx, f: @expr, args: [@expr]) {
    let arg_ts = ty::ty_fn_args(ty::expr_ty(cx.tcx, f));
    let mut i = 0u;
    for arg_ts.each {|arg_t|
        alt ty::resolved_mode(cx.tcx, arg_t.mode) {
          by_mutbl_ref { check_lval(cx, args[i], msg_mutbl_ref); }
          by_move { check_lval(cx, args[i], msg_move_out); }
          by_ref | by_val | by_copy { }
        }
        i += 1u;
    }
}

fn check_bind(cx: @ctx, f: @expr, args: [option<@expr>]) {
    let arg_ts = ty::ty_fn_args(ty::expr_ty(cx.tcx, f));
    let mut i = 0u;
    for args.each {|arg|
        alt arg {
          some(expr) {
            let o_msg = alt ty::resolved_mode(cx.tcx, arg_ts[i].mode) {
              by_mutbl_ref { some("by mut reference") }
              by_move { some("by move") }
              _ { none }
            };
            alt o_msg {
              some(name) {
                cx.tcx.sess.span_err(
                    expr.span, "can not bind an argument passed " + name);
              }
              none {}
            }
          }
          _ {}
        }
        i += 1u;
    }
}

// returns some if the def cannot be modified.  the kind of modification is
// indicated by `msg`.
fn is_illegal_to_modify_def(cx: @ctx, def: def, msg: msg) -> option<str> {
    alt def {
      def_fn(_, _) | def_mod(_) | def_native_mod(_) | def_const(_) |
      def_use(_) {
        some("static item")
      }
      def_arg(_, m) {
        alt ty::resolved_mode(cx.tcx, m) {
          by_ref | by_val { some("argument of enclosing function") }
          by_mutbl_ref | by_move | by_copy { none }
        }
      }
      def_self(_) { some("self argument") }
      def_upvar(_, inner, node_id) {
        let ty = ty::node_id_to_type(cx.tcx, node_id);
        let proto = ty::ty_fn_proto(ty);
        ret alt proto {
          proto_any | proto_block {
            is_illegal_to_modify_def(cx, *inner, msg)
          }
          proto_bare | proto_uniq | proto_box {
            some("upvar")
          }
        };
      }

      // Note: we should *always* allow all local variables to be assigned
      // here and then guarantee in the typestate pass that immutable local
      // variables are assigned at most once.  But this requires a new kind of
      // propagation (def. not assigned), so I didn't do that.
      def_local(_, false) if msg == msg_move_out { none }
      def_local(_, false) {
        some("immutable local variable")
      }

      def_binding(_) { some("binding") }
      _ { none }
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
