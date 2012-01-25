import core::{vec, str, option};
import option::{some};
import codemap::span;
import syntax::ext::base::ext_ctxt;

// NOTE: Moved from fmt.rs which had this fixme:
// FIXME: Cleanup the naming of these functions

fn make_new_lit(cx: ext_ctxt, sp: span, lit: ast::lit_) -> @ast::expr {
    let sp_lit = @{node: lit, span: sp};
    ret @{id: cx.next_id(), node: ast::expr_lit(sp_lit), span: sp};
}
fn make_new_str(cx: ext_ctxt, sp: span, s: str) -> @ast::expr {
    let lit = ast::lit_str(s);
    ret make_new_lit(cx, sp, lit);
}
fn make_new_int(cx: ext_ctxt, sp: span, i: int) -> @ast::expr {
    let lit = ast::lit_int(i as i64, ast::ty_i);
    ret make_new_lit(cx, sp, lit);
}
fn make_new_uint(cx: ext_ctxt, sp: span, u: uint) -> @ast::expr {
    let lit = ast::lit_uint(u as u64, ast::ty_u);
    ret make_new_lit(cx, sp, lit);
}
fn make_add_expr(cx: ext_ctxt, sp: span, lhs: @ast::expr, rhs: @ast::expr)
   -> @ast::expr {
    let binexpr = ast::expr_binary(ast::add, lhs, rhs);
    ret @{id: cx.next_id(), node: binexpr, span: sp};
}
fn make_path_expr(cx: ext_ctxt, sp: span, idents: [ast::ident]) ->
   @ast::expr {
    let path = {global: false, idents: idents, types: []};
    let sp_path = @{node: path, span: sp};
    let pathexpr = ast::expr_path(sp_path);
    ret @{id: cx.next_id(), node: pathexpr, span: sp};
}
fn make_vec_expr(cx: ext_ctxt, sp: span, exprs: [@ast::expr]) ->
   @ast::expr {
    let vecexpr = ast::expr_vec(exprs, ast::imm);
    ret @{id: cx.next_id(), node: vecexpr, span: sp};
}
fn make_call(cx: ext_ctxt, sp: span, fn_path: [ast::ident],
             args: [@ast::expr]) -> @ast::expr {
    let pathexpr = make_path_expr(cx, sp, fn_path);
    let callexpr = ast::expr_call(pathexpr, args, false);
    ret @{id: cx.next_id(), node: callexpr, span: sp};
}
