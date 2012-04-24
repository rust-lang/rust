import base::*;

fn expand_syntax_ext(cx: ext_ctxt, sp: codemap::span, arg: ast::mac_arg,
                     _body: ast::mac_body) -> @ast::expr {
    let arg = get_mac_arg(cx,sp,arg);
    let args: [@ast::expr] =
        alt arg.node {
          ast::expr_vec(elts, _) { elts }
          _ {
            cx.span_fatal(sp, "#concat_idents requires a vector argument .")
          }
        };
    let mut res: ast::ident = "";
    for args.each {|e|
        res += expr_to_ident(cx, e, "expected an ident");
    }

    ret @{id: cx.next_id(),
          node: ast::expr_path(@{span: sp, global: false, idents: [res],
                                 rp: none, types: []}),
          span: sp};
}
