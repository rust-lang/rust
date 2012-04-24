// xfail-test
// fails pretty printing for some reason
use syntax(name = "rustsyntax");
import syntax::diagnostic;
import syntax;
import syntax::ast;
import syntax::codemap;
import syntax::print::pprust;
import syntax::parse::parser;

fn new_parse_sess() -> parser::parse_sess {
    let cm = codemap::new_codemap();
    let handler = diagnostic::mk_handler(option::none);
    let sess = @{
        cm: cm,
        mut next_id: 1,
        span_diagnostic: diagnostic::mk_span_handler(handler, cm),
        mut chpos: 0u,
        mut byte_pos: 0u
    };
    ret sess;
}

iface fake_ext_ctxt {
    fn session() -> fake_session;
    fn cfg() -> ast::crate_cfg;
    fn parse_sess() -> parser::parse_sess;
}

type fake_options = {cfg: ast::crate_cfg};

type fake_session = {opts: @fake_options,
                     parse_sess: parser::parse_sess};

impl of fake_ext_ctxt for fake_session {
    fn session() -> fake_session {self}
    fn cfg() -> ast::crate_cfg { self.opts.cfg }
    fn parse_sess() -> parser::parse_sess { self.parse_sess }
}

fn mk_ctxt() -> fake_ext_ctxt {
    let opts : fake_options = {cfg: []};
    {opts: @opts, parse_sess: new_parse_sess()} as fake_ext_ctxt
}


fn main() {
    let ext_cx = mk_ctxt();
    let s = #ast(expr){__s};
    let e = #ast(expr){__e};
    let f = #ast(expr){$(s).foo {|__e| $(e)}};
    log(error, pprust::expr_to_str(f));
}
