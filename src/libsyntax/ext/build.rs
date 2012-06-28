import codemap::span;
import base::ext_ctxt;

fn mk_lit(cx: ext_ctxt, sp: span, lit: ast::lit_) -> @ast::expr {
    let sp_lit = @{node: lit, span: sp};
    ret @{id: cx.next_id(), node: ast::expr_lit(sp_lit), span: sp};
}
fn mk_str(cx: ext_ctxt, sp: span, s: str) -> @ast::expr {
    let lit = ast::lit_str(@s);
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
    cx.next_id(); // see ast_util::op_expr_callee_id
    let binexpr = ast::expr_binary(op, lhs, rhs);
    ret @{id: cx.next_id(), node: binexpr, span: sp};
}
fn mk_unary(cx: ext_ctxt, sp: span, op: ast::unop, e: @ast::expr)
    -> @ast::expr {
    cx.next_id(); // see ast_util::op_expr_callee_id
    let expr = ast::expr_unary(op, e);
    ret @{id: cx.next_id(), node: expr, span: sp};
}
fn mk_path(cx: ext_ctxt, sp: span, idents: [ast::ident]/~) ->
    @ast::expr {
    let path = @{span: sp, global: false, idents: idents,
                 rp: none, types: []/~};
    let pathexpr = ast::expr_path(path);
    ret @{id: cx.next_id(), node: pathexpr, span: sp};
}
fn mk_access_(cx: ext_ctxt, sp: span, p: @ast::expr, m: ast::ident)
    -> @ast::expr {
    let expr = ast::expr_field(p, m, []/~);
    ret @{id: cx.next_id(), node: expr, span: sp};
}
fn mk_access(cx: ext_ctxt, sp: span, p: [ast::ident]/~, m: ast::ident)
    -> @ast::expr {
    let pathexpr = mk_path(cx, sp, p);
    ret mk_access_(cx, sp, pathexpr, m);
}
fn mk_call_(cx: ext_ctxt, sp: span, fn_expr: @ast::expr,
            args: [@ast::expr]/~) -> @ast::expr {
    let callexpr = ast::expr_call(fn_expr, args, false);
    ret @{id: cx.next_id(), node: callexpr, span: sp};
}
fn mk_call(cx: ext_ctxt, sp: span, fn_path: [ast::ident]/~,
             args: [@ast::expr]/~) -> @ast::expr {
    let pathexpr = mk_path(cx, sp, fn_path);
    ret mk_call_(cx, sp, pathexpr, args);
}
// e = expr, t = type
fn mk_vec_e(cx: ext_ctxt, sp: span, exprs: [@ast::expr]/~) ->
   @ast::expr {
    let vecexpr = ast::expr_vec(exprs, ast::m_imm);
    ret @{id: cx.next_id(), node: vecexpr, span: sp};
}
fn mk_vstore_e(cx: ext_ctxt, sp: span, expr: @ast::expr, vst: ast::vstore) ->
   @ast::expr {
    let vstoreexpr = ast::expr_vstore(expr, vst);
    ret @{id: cx.next_id(), node: vstoreexpr, span: sp};
}
fn mk_uniq_vec_e(cx: ext_ctxt, sp: span, exprs: [@ast::expr]/~) ->
   @ast::expr {
    mk_vstore_e(cx, sp, mk_vec_e(cx, sp, exprs), ast::vstore_uniq)
}
fn mk_fixed_vec_e(cx: ext_ctxt, sp: span, exprs: [@ast::expr]/~) ->
   @ast::expr {
    mk_vstore_e(cx, sp, mk_vec_e(cx, sp, exprs), ast::vstore_fixed(none))
}

fn mk_rec_e(cx: ext_ctxt, sp: span,
            fields: [{ident: ast::ident, ex: @ast::expr}]/~) ->
    @ast::expr {
    let mut astfields: [ast::field]/~ = []/~;
    for fields.each {|field|
        let ident = field.ident;
        let val = field.ex;
        let astfield =
            {node: {mutbl: ast::m_imm, ident: ident, expr: val}, span: sp};
        vec::push(astfields, astfield);
    }
    let recexpr = ast::expr_rec(astfields, option::none::<@ast::expr>);
    ret @{id: cx.next_id(), node: recexpr, span: sp};
}

