import base::*;
import io::writer_util;

fn expand_syntax_ext(cx: ext_ctxt, sp: codemap::span, arg: ast::mac_arg,
                     _body: ast::mac_body) -> @ast::expr {
    let arg = get_mac_arg(cx,sp,arg);
    cx.print_backtrace();
    io::stdout().write_line(print::pprust::expr_to_str(arg));

    //trivial expression
    ret @{id: cx.next_id(), node: ast::expr_rec([], option::none), span: sp};
}
