import std::vec;
import std::option;
import base::*;
import syntax::ast;

fn expand_syntax_ext(cx: &ext_ctxt, sp: codemap::span, arg: @ast::expr,
                     body: option::t[str]) -> @ast::expr {
    let args: [@ast::expr] = alt arg.node {
      ast::expr_vec(elts, _, _) { elts }
      _ { cx.span_fatal(sp, "#ident_to_str requires a vector argument .") }
    };
    if vec::len[@ast::expr](args) != 1u {
        cx.span_fatal(sp, "malformed #ident_to_str call");
    }

    ret make_new_lit(cx, sp,
                     ast::lit_str(expr_to_ident(cx, args.(0u),
                                                "expected an ident"),
                                  ast::sk_rc));

}
