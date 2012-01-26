import core::{vec, str, option};
import option::{some};
import codemap::span;
import syntax::ext::base::ext_ctxt;

// NOTE: Moved from fmt.rs which had this fixme:
// FIXME: Cleanup the naming of these functions

fn mk_lit(cx: ext_ctxt, sp: span, lit: ast::lit_) -> @ast::expr {
    let sp_lit = @{node: lit, span: sp};
    ret @{id: cx.next_id(), node: ast::expr_lit(sp_lit), span: sp};
}
fn mk_str(cx: ext_ctxt, sp: span, s: str) -> @ast::expr {
    let lit = ast::lit_str(s);
    ret mk_lit(cx, sp, lit);
}
fn mk_int(cx: ext_ctxt, sp: span, i: int) -> @ast::expr {
    let lit = ast::lit_int(i as i64, ast::ty_i);
    ret mk_lit(cx, sp, lit);
}
fn mk_uint(cx: ext_ctxt, sp: span, u: uint) -> @ast::expr {
    let lit = ast::lit_uint(u as u64, ast::ty_u);
    ret mk_lit(cx, sp, lit);
}
fn mk_binary(cx: ext_ctxt, sp: span, op: ast::binop,
             lhs: @ast::expr, rhs: @ast::expr)
   -> @ast::expr {
    let binexpr = ast::expr_binary(op, lhs, rhs);
    ret @{id: cx.next_id(), node: binexpr, span: sp};
}
fn mk_unary(cx: ext_ctxt, sp: span, op: ast::unop, e: @ast::expr)
    -> @ast::expr {
    let expr = ast::expr_unary(op, e);
    ret @{id: cx.next_id(), node: expr, span: sp};
}
fn mk_path(cx: ext_ctxt, sp: span, idents: [ast::ident]) ->
    @ast::expr {
    let path = {global: false, idents: idents, types: []};
    let sp_path = @{node: path, span: sp};
    let pathexpr = ast::expr_path(sp_path);
    ret @{id: cx.next_id(), node: pathexpr, span: sp};
}
fn mk_access_(cx: ext_ctxt, sp: span, p: @ast::expr, m: ast::ident)
    -> @ast::expr {
    let expr = ast::expr_field(p, m, []);
    ret @{id: cx.next_id(), node: expr, span: sp};
}
fn mk_access(cx: ext_ctxt, sp: span, p: [ast::ident], m: ast::ident)
    -> @ast::expr {
    let pathexpr = mk_path(cx, sp, p);
    ret mk_access_(cx, sp, pathexpr, m);
}
fn mk_call_(cx: ext_ctxt, sp: span, fn_expr: @ast::expr,
            args: [@ast::expr]) -> @ast::expr {
    let callexpr = ast::expr_call(fn_expr, args, false);
    ret @{id: cx.next_id(), node: callexpr, span: sp};
}
fn mk_call(cx: ext_ctxt, sp: span, fn_path: [ast::ident],
             args: [@ast::expr]) -> @ast::expr {
    let pathexpr = mk_path(cx, sp, fn_path);
    ret mk_call_(cx, sp, pathexpr, args);
}
// e = expr, t = type
fn mk_vec_e(cx: ext_ctxt, sp: span, exprs: [@ast::expr]) ->
   @ast::expr {
    let vecexpr = ast::expr_vec(exprs, ast::imm);
    ret @{id: cx.next_id(), node: vecexpr, span: sp};
}
fn mk_rec_e(cx: ext_ctxt, sp: span,
            fields: [{ident: ast::ident, ex: @ast::expr}]) ->
    @ast::expr {
    let astfields: [ast::field] = [];
    for field: {ident: ast::ident, ex: @ast::expr} in fields {
        let ident = field.ident;
        let val = field.ex;
        let astfield =
            {node: {mut: ast::imm, ident: ident, expr: val}, span: sp};
        astfields += [astfield];
    }
    let recexpr = ast::expr_rec(astfields, option::none::<@ast::expr>);
    ret @{id: cx.next_id(), node: recexpr, span: sp};
}

