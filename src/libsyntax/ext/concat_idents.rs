import base::*;

fn expand_syntax_ext(cx: ext_ctxt, sp: codemap::span, arg: ast::mac_arg,
                     _body: ast::mac_body) -> @ast::expr {
    let args = get_mac_args_no_max(cx,sp,arg,1u,"concat_idents");
    let mut res = "";
    for args.each {|e|
        res += *expr_to_ident(cx, e, "expected an ident");
    }

    ret @{id: cx.next_id(),
          node: ast::expr_path(@{span: sp, global: false, idents: [@res]/~,
                                 rp: none, types: []/~}),
          span: sp};
}
