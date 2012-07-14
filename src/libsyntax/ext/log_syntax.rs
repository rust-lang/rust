import base::*;
import io::writer_util;

fn expand_syntax_ext(cx: ext_ctxt, sp: codemap::span, arg: ast::mac_arg,
                     _body: ast::mac_body) -> @ast::expr {
    let args = get_mac_args_no_max(cx,sp,arg,0u,~"log_syntax");
    cx.print_backtrace();
    io::stdout().write_line(
        str::connect(vec::map(args,
                              |&&ex| print::pprust::expr_to_str(ex)), ~", ")
    );

    //trivial expression
    ret @{id: cx.next_id(), callee_id: cx.next_id(),
          node: ast::expr_rec(~[], option::none), span: sp};
}
