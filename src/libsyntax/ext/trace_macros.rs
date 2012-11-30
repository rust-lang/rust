use codemap::span;
use ext::base::ext_ctxt;
use ast::tt_delim;
use parse::lexer::{new_tt_reader, reader};
use parse::parser::Parser;

fn expand_trace_macros(cx: ext_ctxt, sp: span,
                       tt: ~[ast::token_tree]) -> base::mac_result
{
    let sess = cx.parse_sess();
    let cfg = cx.cfg();
    let tt_rdr = new_tt_reader(cx.parse_sess().span_diagnostic,
                               cx.parse_sess().interner, None, tt);
    let rdr = tt_rdr as reader;
    let rust_parser = Parser(sess, cfg, rdr.dup());

    if rust_parser.is_keyword(~"true") {
        cx.set_trace_macros(true);
    } else if rust_parser.is_keyword(~"false") {
        cx.set_trace_macros(false);
    } else {
        cx.span_fatal(sp, ~"trace_macros! only accepts `true` or `false`")
    }

    rust_parser.bump();

    let rust_parser = Parser(sess, cfg, rdr.dup());
    let result = rust_parser.parse_expr();
    base::mr_expr(result)
}
