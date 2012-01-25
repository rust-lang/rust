#[doc = "AST-parsing helpers"];

import rustc::driver::diagnostic;
import rustc::syntax::ast;
import rustc::syntax::codemap;
import rustc::syntax::parse::parser;

export from_file, from_str;

fn new_parse_sess() -> parser::parse_sess {
    let cm = codemap::new_codemap();
    let handler = diagnostic::mk_handler(none);
    let sess = @{
        cm: cm,
        mutable next_id: 1,
        span_diagnostic: diagnostic::mk_span_handler(handler, cm),
        mutable chpos: 0u,
        mutable byte_pos: 0u
    };
    ret sess;
}

fn from_file(file: str) -> @ast::crate {
    parser::parse_crate_from_file(
        file, [], new_parse_sess())
}

fn from_str(source: str) -> @ast::crate {
    parser::parse_crate_from_source_str(
        "-", source, [], new_parse_sess())
}
