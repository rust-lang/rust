// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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

use ast;
use codemap::span;
use ext::base;
use ext::base::ExtCtxt;
use ext::pipes::parse_proto::proto_parser;
use ext::pipes::pipec::gen_init;
use ext::pipes::proto::visit;
use parse::lexer::{new_tt_reader, reader};
use parse::parser::Parser;

pub mod ast_builder;
pub mod parse_proto;
pub mod pipec;
pub mod proto;
pub mod check;
pub mod liveness;


pub fn expand_proto(cx: @ExtCtxt, _sp: span, id: ast::ident,
                    tt: ~[ast::token_tree]) -> base::MacResult {
    let sess = cx.parse_sess();
    let cfg = cx.cfg();
    let tt_rdr = new_tt_reader(cx.parse_sess().span_diagnostic,
                               None,
                               tt.clone());
    let rdr = tt_rdr as @reader;
    let rust_parser = Parser(sess, cfg, rdr.dup());

    let proto = rust_parser.parse_proto(cx.str_of(id));

    // check for errors
    visit(proto, cx);

    // do analysis
    liveness::analyze(proto, cx);

    // compile
    base::MRItem(proto.compile(cx))
}
