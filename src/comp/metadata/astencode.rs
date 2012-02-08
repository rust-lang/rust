// Encoding of ASTs and the associated side tables.

import middle::base::trans::common::crate_ctxt;
import syntax::ast;
import syntax::codemap::{span, filename};
import std::ebml::writer;
import metadata::common::*;

enum ast_tag {
    at_span,
    at_id,

    at_span_expninfo_callie_name,
    at_span_expninfo_callie_span,

    at_blk,
    at_blk_stmts,
    at_blk_expr,
    at_blk_rules,

    at_stmt,
    at_stmt_node_decl,
    at_stmt_node_expr,

    at_expr,
    at_expr_node_vec,
    at_expr_node_rec,
    at_expr_node_call,
    at_expr_node_tup,
    at_expr_node_bind,
    at_expr_node_bind_args,
    at_expr_node_binary,
    at_expr_node_unary,
    at_expr_node_lit,
    at_expr_node_cast,
    at_expr_node_if,
    at_expr_node_while,

    at_none,
    at_some,
}

type ast_ctxt = {
    embl_w: ebml::writer,
    ccx: crate_ctxt,
};

impl ast_output for ast_ctxt {
    fn tag(tag: ast_tag, blk: fn()) {
        self.embl_w.wr_tag(tag as uint, blk)
    }

    fn uint(v: uint) {
        self.embl_w.wr_uint(v)
    }

    fn opt<T>(x: option<T>, blk: fn(T)) {
        alt x {
          none { self.tag(at_none) {||} }
          some(v) { self.tag(at_some) {|| blk(v) } }
        }
    }

    fn str(tag: ast_tag, v: str) {
        self.tag(tag) {|| self.embl_w.wr_str(v) };
    }

    fn vec<T>(tag: ast_tag, v: [T], blk: fn(T)) {
        self.tag(tag) {||
            self.uint(vec::len(v));
            vec::iter(v) {|e| blk(e) };
        }
    }

    fn span(sp: span) {
        self.tag(at_span) {||
            self.uint(sp.lo);
            self.uint(sp.hi);
            self.opt(sp.expn_info) {|ei|
                self.span(ei.call_site);
                self.str(at_span_expninfo_callie_name, ei.callie.name);
                self.opt(ei.callie.span) {|v| self.span(v) };
            }
        }
    }

    fn id(id: uint) {
        self.tag(at_id) {|| self.uint(id); }
    }

    fn blk(blk: ast::blk) {
        self.tag(at_blk) {||
            self.id(blk.node.id);
            self.span(blk.span);
            self.vec(at_blk_stmts, blk.node.stmts) {|stmt|
                self.stmt(stmt)
            }
            self.tag(at_blk_expr) {||
                self.opt(blk.node.expr) {|e| self.expr(e) }
            }
            self.tag(at_blk_rules) {||
                self.uint(blk.node.rules as uint);
            }
        }
    }

    fn decl(decl: ast::decl) {
        self.span(decl.span);
        alt decl.node {
          ast::decl_local(lcls) {
            self.vec(at_decl_local, lcls) {|lcl|
                self.local(lcl)
            }
          }

          ast::decl_item(item) {
            self.tag(at_decl_item) {||
                self.item(item);
            }
          }
        }
    }

    fn local(lcl: ast::local) {
        self.span(lcl.span);
        self.ty(lcl.ty);
        self.pat(lcl.pat);
        self.opt(lcl.init) {|i| self.initializer(i) };
        self.uint(lcl.id);
    }

    fn pat(pat: ast::pat) {
        self.uint(pat.id);
        self.span(pat.span);
        alt pat_util::normalize_pat(pat.node) {
          pat_wild {
            self.tag(at_pat_wild) {||
            }
          }
          pat_ident(path, o_pat) {
            self.tag(at_pat_ident) {||
                self.path(path);
                self.opt(o_pat) {|p|
                    self.pat(p)
                }
            }
          }
          pat_enum(path, pats) {
            self.tag(at_pat_enum) {||
                self.path(path);
                self.vec(at_pat_enum_pats, pats) {|p| self.pat(p) };
            }
          }
          pat_rec(field_pats, b) {
            self.tag(at_pat_rec) {||
                self.vec(at_pat_rec_fields, field_pats) {|p|
                    self.field_pat(p)
                }
            }
          }
          pat_tup(pats) {
            self.vec(at_pat_tup, pats) {|p| self.pat(p); }
          }
          pat_box(pat) {
            self.tag(at_pat_box) {|| self.pat(pat) }
          }
          pat_lit(expr) {
            self.tag(at_pat_lit) {|| self.expr(expr) }
          }
          pat_range(l, h) {
            self.tag(at_pat_range) {||
                self.expr(l);
                self.expr(h);
            }
          }
        }
    }

    fn stmt(stmt: ast::stmt) {
        self.tag(at_stmt) {||
            self.span(stmt.span);
            alt stmt.node {
              ast::stmt_decl(d, nid) {
                self.id(nid);
                self.tag(at_stmt_node_decl) {|| self.decl(d) };
              }
              ast::stmt_expr(e, nid) | ast::stmt_semi(e, nid) {
                self.id(nid);
                self.tag(at_stmt_node_expr) {|| self.expr(e) };
              }
            }
        }
    }

    fn exprs(exprs: [ast::expr]) {
        self.vec(at_exprs, exprs) {|e| self.expr(e) };
    }

    fn expr(expr: ast:expr) {
        self.id(expr.id);
        self.span(expr.span);
        alt expr.node {
          ast::expr_vec(subexprs, mutbl) {
            self.tag(at_expr_node_vec) {||
                self.exprs(subexprs);
                self.mutbl(mutbl);
            }
          }

          ast::expr_rec(fields, opt_expr) {
            self.tag(at_expr_node_rec) {||
                self.fields(fields);
                self.opt(opt_expr) {|e| self.expr(e) };
            }
          }

          ast::expr_call(func, args, _) {
            self.tag(at_expr_node_call) {||
                self.expr(func);
                self.exprs(args);
            }
          }

          ast::expr_tup(exprs) {
            self.tag(at_expr_node_tup) {||
                self.exprs(exprs);
            }
          }

          ast::expr_bind(f, args) {
            self.tag(at_expr_node_bind) {||
                self.expr(f);
                self.vec(at_expr_node_bind_args, args) {|opt_e|
                    self.opt(opt_e) {|e| self.expr(e)};
                }
            }
          }

          ast::expr_binary(binop, l, r) {
            self.tag(at_expr_node_binary) {||
                self.uint(binop as uint);
                self.expr(l);
                self.expr(r);
            }
          }

          ast::expr_unary(unop, l, r) {
            self.tag(at_expr_node_unary) {||
                self.uint(unop as uint);
                self.expr(l);
                self.expr(r);
            }
          }

          ast::expr_lit(lit) {
            self.tag(at_expr_node_lit) {|| self.lit(lit) }
          }

          ast::expr_cast(expr, ty) {
            self.tag(at_expr_node_cast) {||
                self.expr(expr);
                self.ty(ty);
            }
          }

          ast::expr_if(cond, blk_then, o_blk_else) {
            self.tag(at_expr_node_if) {||
                self.expr(cond);
                self.blk(blk_then);
                self.opt(o_blk_else) {|b| self.blk(b)};
            }
          }

          ast::expr_while(cond, blk) {
            self.tag(at_expr_node_while) {||
                self.expr(cond);
                self.blk(blk);
            }
          }

          ast::expr_for(lcl, expr, blk) {
            self.tag(at_expr_node_for) {||
                self.local(lcl);
                self.expr(expr);
                self.blk(blk);
            }
          }

          ast::expr_do_while(blk, cond) {
            self.tag(at_expr_node_do_while) {||
                self.blk(blk);
                self.expr(cond);
            }
          }

          ast::expr_alt(cond, arms) {
            self.tag(at_expr_node_alt) {||
                self.blk(blk);
                self.expr(cond);
            }
          }

          ast::expr_block(blk) {
            self.tag(at_expr_node_blk) {||
                self.blk(blk);
            }
          }

          ast::expr_copy(expr) {
            self.tag(at_expr_node_copy) {||
                self.expr(expr);
            }
          }

          ast::expr_move(l, r) {
            self.tag(at_expr_node_move) {||
                self.expr(l);
                self.expr(r);
            }
          }

          ast::expr_assign(l, r) {
            self.tag(at_expr_node_assign) {||
                self.expr(l);
                self.expr(r);
            }
          }

          ast::expr_swap(l, r) {
            self.tag(at_expr_node_swap) {||
                self.expr(l);
                self.expr(r);
            }
          }

          ast::expr_assign_of(binop, l, r) {
            self.tag(at_expr_node_assign_op) {||
                self.uint(binop as uint);
                self.expr(l);
                self.expr(r);
            }
          }

          ast::expr_field(base, f, tys) {
            self.tag(at_expr_node_field) {||
                self.expr(base);
                self.str(at_ident, f);
                self.vec(at_tys) {|v| self.ty(v) }
            }
          }

          ast::expr_index(l, r) {
            self.tag(at_expr_node_index) {||
                self.expr(l);
                self.expr(r);
            }
          }

          ast::expr_path(pth) {
            self.tag(at_expr_node_path) {||
            }
          }

          ast::expr_fail(o_expr) {
            self.tag(at_expr_node_fail) {||
                self.opt(o_expr) {|e| self.expr(e) }
            }
          }

          ast::expr_break {
            self.tag(at_expr_node_break) {||}
          }

          ast::expr_cont {
            self.tag(at_expr_node_cont) {||}
          }

          ast::expr_ret(o_expr) {
            self.tag(at_expr_node_ret) {||
                self.opt(o_expr) {|e| self.expr(e) }
            }
          }

          ast::expr_be(expr) {
            self.tag(at_expr_node_be) {||
                self.expr(expr)
            }
          }

          ast::expr_log(i, e1, e2) {
            self.tag(at_expr_node_log) {||
                self.uint(i);
                self.expr(e1);
                self.expr(e2);
            }
          }

          ast::expr_assert(e) {
            self.tag(at_expr_node_assert) {||
                self.expr(e);
            }
          }

          ast::expr_check(mode, e) {
            self.tag(at_expr_node_check) {||
                self.uint(mode as uint);
                self.expr(e);
            }
          }

          ast::expr_if_check(cond, b, e) {
            self.tag(at_expr_node_if_check) {||
                self.expr(cond);
                self.blk(b);
                self.opt(e) {|e| self.blk(e)};
            }
          }

          ast::expr_mac(m) {
            self.tag(at_expr_node_mac) {||
                /* todo */
            }
          }
        }
    }

    fn lit(l: ast::lit) {
        alt l {
          lit_str(s) {
            self.str(at_lit_str, s);
          }
          lit_int(i, t) {
            self.tag(at_lit_int) {||
                self.i64(i);
                self.int_ty(t);
            }
          }
          lit_uint(i, t) {
            self.tag(at_lit_uint) {||
                self.u64(i);
                self.uint_ty(t);
            }
          }
          lit_float(s, f) {
            self.tag(at_lit_float) {||
                self.str(at_value, s);
                self.float_ty(f);
            }
          }
          lit_nil {
            self.tag(at_lit_nil) {||}
          }
          lit_bool(true) {
            self.tag(at_lit_true) {||}
          }
          lit_bool(false) {
            self.tag(at_lit_false) {||}
          }
        }
    }

    fn int_ty(t: ast::int_ty) {
        self.uint(t as uint);
    }

    fn uint_ty(t: ast::uint_ty) {
        self.uint(t as uint);
    }

    fn float_ty(t: ast::float_ty) {
        self.uint(t as uint);
    }

    fn ty(ty: ast::ty) {
        self.tag(at_ty) {||
            self.span(ty.span);
            alt ty.node {
              ty_nil {
                self.tag(at_ty_nil) {||}
              }

              ty_bot {
                self.tag(at_ty_bot) {||}
              }

              ty_box({ty: ty, mut: m}) {
                self.tag(at_ty_box) {||
                    self.ty(ty);
                    self.mutbl(m);
                }
              }

              ty_uniq({ty: ty, mut: m}) {
                self.tag(at_ty_uniq) {||
                    self.ty(ty);
                    self.mutbl(m);
                }
              }

              ty_vec({ty: ty, mut: m}) {
                self.tag(at_ty_vec) {||
                    self.ty(ty);
                    self.mutbl(m);
                }
              }

              ty_ptr({ty: ty, mut: m}) {
                self.tag(at_ty_ptr) {||
                    self.ty(ty);
                    self.mutbl(m);
                }
              }

              ty_rec(fields) {
                self.vec(at_ty_rec) {|f|
                    self.field(f)
                }
              }

              ty_fn(proto, fd) {
                self.tag(at_ty_fn) {||
                    self.uint(proto as uint);
                    self.fn_decl(fd)
                }
              }

              ty_tup(tys) {
                self.vec(at_ty_tups) {|ty| self.ty(ty)}
              }

              ty_path(p, id) {
                self.tag(at_ty_path) {||
                    self.path(p);
                    self.uint(id);
                }
              }

              ty_constr(t, tcs) {
                self.tag(at_ty_constr) {||
                    self.ty(t);
                    // ... constrs ... who cares ...
                }
              }

              ty_mac(m) {
                self.tag(at_ty_mac) {||
                    self.mac(m);
                };
              }

              ty_infer {
                self.tag(at_ty_infer) {||
                }
              }
            }
        }
    }

    fn item(item: @ast::item) {
        self.tag(at_item) {||
            self.str(at_item_ident, item);
            self.vec(at_item_attrs, item.attrs) {|a| self.attr(a)}
            self.uint(item.id);
            self.span(item.span);

            alt item.node {
              item_const(t, e) {
                self.tag(at_item_const) {||
                    self.ty(t);
                    self.expr(e);
                }
              }
              item_fn(d, tps, blk) {
                self.tag(at_item_fn) {||
                    self.fn_decl(d);
                    self.ty_params(tps);
                }
              }
              item_mod(m) {
                self.tag(at_item_mod) {||
                    self.mod_(m)
                }
              }
              item_native_mod(nm) {
                self.tag(at_item_native_mod) {||
                    self.mod_(nm)
                }
              }
              item_ty(ty, tps) {
                self.tag(at_item_ty) {||
                    self.ty(ty);
                    self.ty_params(tps);
                }
              }
              item_enum(variants, tps) {
                self.tag(at_item_enum) {||
                    self.ty(ty);
                    self.ty_params(tps);
                }
              }
              item_res(fd, tps, blk, node_id, node_id) {
                self.tag(at_item_res) {||
                    self.fn_decl(fd);
                    self.ty_params(tps);
                }
              }
              item_class(tps, citems, fn_decl, blk) {
                self.tag(at_item_class) {||
                    self.ty_params(tps);
                    self.class_items(citems);
                    self.fn_decl(fn_decl);
                    self.blk(blk);
                }
              }
              item_iface(tps, tms) {
                self.tag(at_item_iface) {||
                    self.ty_params(tps);
                    self.ty_methods(tms);
                }
              }
              item_impl(tps, iface_ty, self_ty, mthds) {
                self.tag(at_item_impl) {||
                    self.ty_params(tps);
                    self.opt(iface_ty) {|t| self.ty(t) };
                    self.ty(self_ty);
                    self.methods(mthds);
                }
              }
            }
        }
    }

    fn ty_params(tps: [ast::ty_param]) {
        self.vec(at_item_tps, tps) {|t| self.ty_param(t) }
    }

    fn ty_param(tp: ast::ty_param) {
        self.str(at_ty_param_ident, tp.ident);
        self.uint(at_ty_param_id, tp.id);
        self.vec(at_param_bounds, *tp.bounds) {|b| self.ty_param_bound(b) };
    }

    fn ty_param_bound(b: ast::ty_param_bound) {
        alt b {
          bound_copy { self.tag(at_ty_param_bound_copy) {||} }
          bound_send { self.tag(at_ty_param_bound_send) {||} }
          bound_iface(t) {
            self.tag(at_ty_param_bound_iface) {|| self.ty(t) }
          }
        }
    }
}

