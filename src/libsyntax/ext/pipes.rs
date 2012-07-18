
import codemap::span;
import ext::base::ext_ctxt;
import ast::tt_delim;
import parse::lexer::{new_tt_reader, reader, tt_reader_as_reader};
import parse::parser::{parser, SOURCE_FILE};
import parse::common::parser_common;

import pipes::parse_proto::proto_parser;

import pipes::pipec::compile;
import pipes::proto::{visit, protocol};
import pipes::check::proto_check;

fn expand_proto(cx: ext_ctxt, _sp: span, id: ast::ident,
                tt: ~[ast::token_tree]) -> base::mac_result
{
    let sess = cx.parse_sess();
    let cfg = cx.cfg();
    let tt_rdr = new_tt_reader(cx.parse_sess().span_diagnostic,
                               cx.parse_sess().interner, none, tt);
    let rdr = tt_rdr as reader;
    let rust_parser = parser(sess, cfg, rdr.dup(), SOURCE_FILE);

    let proto = rust_parser.parse_proto(id);

    // check for errors
    visit(proto, cx);

    // do analysis
    liveness::analyze(proto, cx);

    // compile
    base::mr_item(proto.compile(cx))
}
