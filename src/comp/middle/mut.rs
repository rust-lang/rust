import std::{vec, str, option};
import option::{some, none};
import syntax::ast::*;
import syntax::visit;
import syntax::ast_util;

tag deref_t { unbox; field; index; }

type deref = @{mut: bool, kind: deref_t, outer_t: ty::t};

// Finds the root (the thing that is dereferenced) for the given expr, and a
// vec of dereferences that were used on this root. Note that, in this vec,
// the inner derefs come in front, so foo.bar[1] becomes rec(ex=foo,
// ds=[index,field])
fn expr_root(tcx: ty::ctxt, ex: @expr, autoderef: bool) ->
   {ex: @expr, ds: @[deref]} {
    fn maybe_auto_unbox(tcx: ty::ctxt, t: ty::t) -> {t: ty::t, ds: [deref]} {
        let ds = [];
        while true {
            alt ty::struct(tcx, t) {
              ty::ty_box(mt) {
                ds += [@{mut: mt.mut == mut, kind: unbox, outer_t: t}];
                t = mt.ty;
              }
              ty::ty_uniq(mt) {
                ds += [@{mut: mt.mut == mut, kind: unbox, outer_t: t}];
                t = mt.ty;
              }
              ty::ty_res(_, inner, tps) {
                ds += [@{mut: false, kind: unbox, outer_t: t}];
                t = ty::substitute_type_params(tcx, tps, inner);
              }
              ty::ty_tag(did, tps) {
                let variants = ty::tag_variants(tcx, did);
                if vec::len(variants) != 1u ||
                       vec::len(variants[0].args) != 1u {
                    break;
                }
                ds += [@{mut: false, kind: unbox, outer_t: t}];
                t = ty::substitute_type_params(tcx, tps, variants[0].args[0]);
              }
              _ { break; }
            }
        }
        ret {t: t, ds: ds};
    }
    let ds: [deref] = [];
    while true {
        alt copy ex.node {
          expr_field(base, ident) {
            let auto_unbox = maybe_auto_unbox(tcx, ty::expr_ty(tcx, base));
            let is_mut = false;
            alt ty::struct(tcx, auto_unbox.t) {
              ty::ty_rec(fields) {
                for fld: ty::field in fields {
                    if str::eq(ident, fld.ident) {
                        is_mut = fld.mt.mut == mut;
                        break;
                    }
                }
              }
              ty::ty_obj(_) { }
            }
            ds += [@{mut: is_mut, kind: field, outer_t: auto_unbox.t}];
            ds += auto_unbox.ds;
            ex = base;
          }
          expr_index(base, _) {
            let auto_unbox = maybe_auto_unbox(tcx, ty::expr_ty(tcx, base));
            alt ty::struct(tcx, auto_unbox.t) {
              ty::ty_vec(mt) {
                ds +=
                    [@{mut: mt.mut == mut,
                       kind: index,
                       outer_t: auto_unbox.t}];
              }
              ty::ty_str. {
                ds += [@{mut: false, kind: index, outer_t: auto_unbox.t}];
              }
            }
            ds += auto_unbox.ds;
            ex = base;
          }
          expr_unary(op, base) {
            if op == deref {
                let base_t = ty::expr_ty(tcx, base);
                let is_mut = false;
                alt ty::struct(tcx, base_t) {
                  ty::ty_box(mt) { is_mut = mt.mut == mut; }
                  ty::ty_uniq(mt) { is_mut = mt.mut == mut; }
                  ty::ty_res(_, _, _) { }
                  ty::ty_tag(_, _) { }
                  ty::ty_ptr(mt) { is_mut = mt.mut == mut; }
                }
                ds += [@{mut: is_mut, kind: unbox, outer_t: base_t}];
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

// Actual mut-checking pass

type mut_map = std::map::hashmap<node_id, ()>;
type ctx = {tcx: ty::ctxt, mut_map: mut_map};

fn check_crate(tcx: ty::ctxt, crate: @crate) -> mut_map {
    let cx = @{tcx: tcx, mut_map: std::map::new_int_hash()};
    let v =
        @{visit_expr: bind visit_expr(cx, _, _, _),
          visit_decl: bind visit_decl(cx, _, _, _)
             with *visit::default_visitor::<()>()};
    visit::visit_crate(*crate, (), visit::mk_vt(v));
    ret cx.mut_map;
}

tag msg { msg_assign; msg_move_out; msg_mut_ref; }

fn mk_err(cx: @ctx, span: syntax::codemap::span, msg: msg, name: str) {
    cx.tcx.sess.span_err(span, alt msg {
      msg_assign. { "assigning to " + name }
      msg_move_out. { "moving out of " + name }
      msg_mut_ref. { "passing " + name + " by mutable reference" }
    });
}

fn visit_decl(cx: @ctx, d: @decl, &&e: (), v: visit::vt<()>) {
    visit::visit_decl(d, e, v);
    alt d.node {
      decl_local(locs) {
        for (_, loc) in locs {
            alt loc.node.init {
              some(init) {
                if init.op == init_move { check_move_rhs(cx, init.expr); }
              }
              none. { }
            }
        }
      }
      _ { }
    }
}

fn visit_expr(cx: @ctx, ex: @expr, &&e: (), v: visit::vt<()>) {
    alt ex.node {
      expr_call(f, args) { check_call(cx, f, args); }
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
      _ { }
    }
    visit::visit_expr(ex, e, v);
}

fn check_lval(cx: @ctx, dest: @expr, msg: msg) {
    alt dest.node {
      expr_path(p) {
        let def = cx.tcx.def_map.get(dest.id);
        alt is_immutable_def(def) {
          some(name) { mk_err(cx, dest.span, msg, name); }
          _ { }
        }
        cx.mut_map.insert(ast_util::def_id_of_def(def).node, ());
      }
      _ {
        let root = expr_root(cx.tcx, dest, false);
        if vec::len(*root.ds) == 0u {
            if msg != msg_move_out {
                mk_err(cx, dest.span, msg, "non-lvalue");
            }
        } else if !root.ds[0].mut {
            let name =
                alt root.ds[0].kind {
                  mut::unbox. { "immutable box" }
                  mut::field. { "immutable field" }
                  mut::index. { "immutable vec content" }
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
          def_obj_field(_, _) {
            mk_err(cx, src.span, msg_move_out, "object field");
          }
          _ { }
        }
        check_lval(cx, src, msg_move_out);
      }
      _ {
        let root = expr_root(cx.tcx, src, false);

        // Not a path and no-derefs means this is a temporary.
        if vec::len(*root.ds) != 0u {
            cx.tcx.sess.span_err(src.span, "moving out of a data structure");
        }
      }
    }
}

fn check_call(cx: @ctx, f: @expr, args: [@expr]) {
    let arg_ts = ty::ty_fn_args(cx.tcx, ty::expr_ty(cx.tcx, f));
    let i = 0u;
    for arg_t: ty::arg in arg_ts {
        alt arg_t.mode {
          by_mut_ref. { check_lval(cx, args[i], msg_mut_ref); }
          by_move. { check_lval(cx, args[i], msg_move_out); }
          _ {}
        }
        i += 1u;
    }
}

fn is_immutable_def(def: def) -> option::t<str> {
    alt def {
      def_fn(_, _) | def_mod(_) | def_native_mod(_) | def_const(_) |
      def_use(_) {
        some("static item")
      }
      def_obj_field(_, imm.) { some("immutable object field") }
      def_upvar(_, inner, mut) {
        if !mut { some("upvar") } else { is_immutable_def(*inner) }
      }
      def_binding(_) { some("binding") }
      def_local(_, let_ref.) { some("by-reference binding") }
      _ { none }
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
