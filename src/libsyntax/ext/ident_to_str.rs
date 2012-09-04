use base::*;
use build::mk_uniq_str;

fn expand_syntax_ext(cx: ext_ctxt, sp: codemap::span, arg: ast::mac_arg,
                     _body: ast::mac_body) -> @ast::expr {
    let args = get_mac_args(cx,sp,arg,1u,option::Some(1u),~"ident_to_str");

    return mk_uniq_str(cx, sp, *cx.parse_sess().interner.get(
        expr_to_ident(cx, args[0u], ~"expected an ident")));
}
