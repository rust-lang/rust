/*! Implementation of proto! extension.

This is frequently called the pipe compiler. It handles code such as...

~~~
proto! pingpong (
    ping: send {
        ping -> pong
    }
    pong: recv {
        pong -> ping
    }
)
~~~

There are several components:

 * The parser (libsyntax/ext/pipes/parse_proto.rs)
   * Responsible for building an AST from a protocol specification.

 * The checker (libsyntax/ext/pipes/check.rs)
   * Basic correctness checking for protocols (i.e. no undefined states, etc.)

 * The analyzer (libsyntax/ext/pipes/liveness.rs)
   * Determines whether the protocol is bounded or unbounded.

 * The compiler (libsynatx/ext/pipes/pipec.rs)
   * Generates a Rust AST from the protocol AST and the results of analysis.

There is more documentation in each of the files referenced above.

FIXME (#3072) - This is still incomplete.

*/

use codemap::span;
use ext::base::ext_ctxt;
use ast::tt_delim;
use parse::lexer::{new_tt_reader, reader};
use parse::parser::Parser;
use parse::common::parser_common;

use pipes::parse_proto::proto_parser;

use pipes::proto::{visit, protocol};

fn expand_proto(cx: ext_ctxt, _sp: span, id: ast::ident,
                tt: ~[ast::token_tree]) -> base::mac_result
{
    let sess = cx.parse_sess();
    let cfg = cx.cfg();
    let tt_rdr = new_tt_reader(cx.parse_sess().span_diagnostic,
                               cx.parse_sess().interner, None, tt);
    let rdr = tt_rdr as reader;
    let rust_parser = Parser(sess, cfg, rdr.dup());

    let proto = rust_parser.parse_proto(cx.str_of(id));

    // check for errors
    visit(proto, cx);

    // do analysis
    liveness::analyze(proto, cx);

    // compile
    base::mr_item(proto.compile(cx))
}
