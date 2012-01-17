import rustc::syntax::ast;

export run;

fn run(
    _doc: doc::cratedoc,
    _crate: @ast::crate
) -> doc::cratedoc {
    fail;
}
