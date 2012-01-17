import rustc::driver::diagnostic;
import rustc::syntax::ast;
import rustc::syntax::codemap;
import rustc::syntax::parse::parser;

export from_file, from_str;

fn new_parse_sess() -> parser::parse_sess {
    let cm = codemap::new_codemap();
    let sess = @{
        cm: cm,
        mutable next_id: 0,
        diagnostic: diagnostic::mk_handler(cm, none)
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
