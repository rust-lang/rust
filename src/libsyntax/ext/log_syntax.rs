import base::*;
import io::WriterUtil;

fn expand_syntax_ext(cx: ext_ctxt, sp: codemap::span, tt: ~[ast::token_tree])
    -> base::mac_result {

    cx.print_backtrace();
    io::stdout().write_line(
        print::pprust::tt_to_str(ast::tt_delim(tt),cx.parse_sess().interner));

    //trivial expression
    return mr_expr(@{id: cx.next_id(), callee_id: cx.next_id(),
                     node: ast::expr_rec(~[], option::none), span: sp});
}
