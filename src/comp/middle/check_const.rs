import syntax::ast::*;
import syntax::{visit, ast_util};
import driver::session::session;

fn check_crate(sess: session, crate: @crate) {
    visit::visit_crate(*crate, false, visit::mk_vt(@{
        visit_item: check_item,
        visit_pat: check_pat,
        visit_expr: bind check_expr(sess, _, _, _)
        with *visit::default_visitor()
    }));
    sess.abort_if_errors();
}

fn check_item(it: @item, &&_is_const: bool, v: visit::vt<bool>) {
    alt it.node {
      item_const(_, ex) { v.visit_expr(ex, true, v); }
      _ { visit::visit_item(it, false, v); }
    }
}

fn check_pat(p: @pat, &&_is_const: bool, v: visit::vt<bool>) {
    fn is_str(e: @expr) -> bool {
        alt e.node { expr_lit(@{node: lit_str(_), _}) { true } _ { false } }
    }
    alt p.node {
      // Let through plain string literals here
      pat_lit(a) { if !is_str(a) { v.visit_expr(a, true, v); } }
      pat_range(a, b) {
        if !is_str(a) { v.visit_expr(a, true, v); }
        if !is_str(b) { v.visit_expr(b, true, v); }
      }
      _ { visit::visit_pat(p, false, v); }
    }
}

fn check_expr(sess: session, e: @expr, &&is_const: bool, v: visit::vt<bool>) {
    if is_const {
        alt e.node {
          expr_unary(box(_), _) | expr_unary(uniq(_), _) |
          expr_unary(deref., _){
            sess.span_err(e.span,
                          "disallowed operator in constant expression");
            ret;
          }
          expr_lit(@{node: lit_str(_), _}) {
            sess.span_err(e.span,
                          "string constants are not supported");
          }
          expr_lit(_) | expr_binary(_, _, _) | expr_unary(_, _) {}
          _ {
            sess.span_err(e.span,
                          "constant contains unimplemented expression type");
            ret;
          }
        }
    }
    alt e.node {
      expr_lit(@{node: lit_int(v, t), _}) {
        if t != ty_char {
            if (v as u64) > ast_util::int_ty_max(
                t == ty_i ? sess.get_targ_cfg().int_type : t) {
                sess.span_err(e.span, "literal out of range for its type");
            }
        }
      }
      expr_lit(@{node: lit_uint(v, t), _}) {
        if v > ast_util::uint_ty_max(
            t == ty_u ? sess.get_targ_cfg().uint_type : t) {
            sess.span_err(e.span, "literal out of range for its type");
        }
      }
      _ {}
    }
    visit::visit_expr(e, is_const, v);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
