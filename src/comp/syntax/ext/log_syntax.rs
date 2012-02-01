import base::*;
import syntax::ast;
import std::io::writer_util;

fn expand_syntax_ext(cx: ext_ctxt, sp: codemap::span, arg: @ast::expr,
                     _body: ast::mac_body) -> @ast::expr {

    cx.print_backtrace();
    std::io::stdout().write_line(print::pprust::expr_to_str(arg));

    //trivial expression
    ret @{id: cx.next_id(), node: ast::expr_rec([], option::none), span: sp};
}
