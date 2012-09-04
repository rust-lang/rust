use codemap::span;
use ext::base::ext_ctxt;
use ast::tt_delim;
use parse::lexer::{new_tt_reader, reader};
use parse::parser::{parser, SOURCE_FILE};
use parse::common::parser_common;

fn expand_trace_macros(cx: ext_ctxt, sp: span,
                       tt: ~[ast::token_tree]) -> base::mac_result
{
    let sess = cx.parse_sess();
    let cfg = cx.cfg();
    let tt_rdr = new_tt_reader(cx.parse_sess().span_diagnostic,
                               cx.parse_sess().interner, None, tt);
    let rdr = tt_rdr as reader;
    let rust_parser = parser(sess, cfg, rdr.dup(), SOURCE_FILE);

    let arg = cx.str_of(rust_parser.parse_ident());
    match arg {
      ~"true"  => cx.set_trace_macros(true),
      ~"false" => cx.set_trace_macros(false),
      _ => cx.span_fatal(sp, ~"trace_macros! only accepts `true` or `false`")
    }
    let rust_parser = parser(sess, cfg, rdr.dup(), SOURCE_FILE);
    let result = rust_parser.parse_expr();
    base::mr_expr(result)
}
