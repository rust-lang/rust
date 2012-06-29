import base::*;
import build::mk_lit;
import option;

fn expand_syntax_ext(cx: ext_ctxt, sp: codemap::span, arg: ast::mac_arg,
                     _body: ast::mac_body) -> @ast::expr {
    let args = get_mac_args(cx,sp,arg,1u,option::some(1u),"ident_to_str");

    ret mk_lit(cx, sp,
               ast::lit_str(expr_to_ident(cx, args[0u],
                                          "expected an ident")));
}
