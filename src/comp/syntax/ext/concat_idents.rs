import option;
import base::*;
import syntax::ast;

fn expand_syntax_ext(cx: ext_ctxt, sp: codemap::span, arg: @ast::expr,
                     _body: option::t<str>) -> @ast::expr {
    let args: [@ast::expr] =
        alt arg.node {
          ast::expr_vec(elts, _) { elts }
          _ {
            cx.span_fatal(sp, "#concat_idents requires a vector argument .")
          }
        };
    let res: ast::ident = "";
    for e: @ast::expr in args {
        res += expr_to_ident(cx, e, "expected an ident");
    }

    ret @{id: cx.next_id(),
          node: ast::expr_path(@{node: {global: false, idents: [res],
                                        types: []},
                                 span: sp}),
          span: sp};
}
