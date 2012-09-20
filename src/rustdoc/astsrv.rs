/*!
Provides all access to AST-related, non-sendable info

Rustdoc is intended to be parallel, and the rustc AST is filled with
shared boxes. The AST service attempts to provide a single place to
query AST-related information, shielding the rest of Rustdoc from its
non-sendableness.
*/

use std::map::HashMap;
use rustc::driver::session;
use session::{basic_options, options};
use session::session;
use rustc::driver::driver;
use syntax::diagnostic;
use syntax::diagnostic::handler;
use syntax::ast;
use syntax::codemap;
use syntax::ast_map;
use rustc::back::link;
use rustc::metadata::filesearch;
use rustc::front;

export Ctxt;
export CtxtHandler;
export Srv;
export from_str;
export from_file;
export exec;

type Ctxt = {
    ast: @ast::crate,
    ast_map: ast_map::map
};

type SrvOwner<T> = fn(srv: Srv) -> T;
type CtxtHandler<T> = fn~(ctxt: Ctxt) -> T;
type Parser = fn~(session, ~str) -> @ast::crate;

enum Msg {
    HandleRequest(fn~(Ctxt)),
    Exit
}

enum Srv = {
    ch: comm::Chan<Msg>
};

fn from_str<T>(source: ~str, owner: SrvOwner<T>) -> T {
    run(owner, source, parse::from_str_sess)
}

fn from_file<T>(file: ~str, owner: SrvOwner<T>) -> T {
    run(owner, file, |sess, f| parse::from_file_sess(sess, &Path(f)))
}

fn run<T>(owner: SrvOwner<T>, source: ~str, +parse: Parser) -> T {

    let srv_ = Srv({
        ch: do task::spawn_listener |po| {
            act(po, source, parse);
        }
    });

    let res = owner(srv_);
    comm::send(srv_.ch, Exit);
    return res;
}

fn act(po: comm::Port<Msg>, source: ~str, parse: Parser) {
    let sess = build_session();

    let ctxt = build_ctxt(
        sess,
        parse(sess, source)
    );

    let mut keep_going = true;
    while keep_going {
        match comm::recv(po) {
          HandleRequest(f) => {
            f(ctxt);
          }
          Exit => {
            keep_going = false;
          }
        }
    }
}

fn exec<T:Send>(
    srv: Srv,
    +f: fn~(ctxt: Ctxt) -> T
) -> T {
    let po = comm::Port();
    let ch = comm::Chan(po);
    let msg = HandleRequest(fn~(move f, ctxt: Ctxt) {
        comm::send(ch, f(ctxt))
    });
    comm::send(srv.ch, msg);
    comm::recv(po)
}

fn build_ctxt(sess: session,
              ast: @ast::crate) -> Ctxt {

    use rustc::front::config;

    let ast = config::strip_unconfigured_items(ast);
    let ast = syntax::ext::expand::expand_crate(sess.parse_sess,
                                                sess.opts.cfg, ast);
    let ast = front::test::modify_for_testing(sess, ast);
    let ast_map = ast_map::map_crate(sess.diagnostic(), *ast);

    {
        ast: ast,
        ast_map: ast_map,
    }
}

fn build_session() -> session {
    let sopts: @options = basic_options();
    let codemap = codemap::new_codemap();
    let error_handlers = build_error_handlers(codemap);
    let {emitter, span_handler} = error_handlers;

    let session = driver::build_session_(sopts, codemap, emitter,
                                         span_handler);
    session
}

type ErrorHandlers = {
    emitter: diagnostic::emitter,
    span_handler: diagnostic::span_handler
};

// Build a custom error handler that will allow us to ignore non-fatal
// errors
fn build_error_handlers(
    codemap: codemap::codemap
) -> ErrorHandlers {

    type DiagnosticHandler = {
        inner: diagnostic::handler,
    };

    impl DiagnosticHandler: diagnostic::handler {
        fn fatal(msg: ~str) -> ! { self.inner.fatal(msg) }
        fn err(msg: ~str) { self.inner.err(msg) }
        fn bump_err_count() {
            self.inner.bump_err_count();
        }
        fn has_errors() -> bool { self.inner.has_errors() }
        fn abort_if_errors() { self.inner.abort_if_errors() }
        fn warn(msg: ~str) { self.inner.warn(msg) }
        fn note(msg: ~str) { self.inner.note(msg) }
        fn bug(msg: ~str) -> ! { self.inner.bug(msg) }
        fn unimpl(msg: ~str) -> ! { self.inner.unimpl(msg) }
        fn emit(cmsp: Option<(codemap::codemap, codemap::span)>,
                msg: ~str, lvl: diagnostic::level) {
            self.inner.emit(cmsp, msg, lvl)
        }
    }

    let emitter = fn@(cmsp: Option<(codemap::codemap, codemap::span)>,
                       msg: ~str, lvl: diagnostic::level) {
        diagnostic::emit(cmsp, msg, lvl);
    };
    let inner_handler = diagnostic::mk_handler(Some(emitter));
    let handler = {
        inner: inner_handler,
    };
    let span_handler = diagnostic::mk_span_handler(
        handler as diagnostic::handler, codemap);

    {
        emitter: emitter,
        span_handler: span_handler
    }
}

#[test]
fn should_prune_unconfigured_items() {
    let source = ~"#[cfg(shut_up_and_leave_me_alone)]fn a() { }";
    do from_str(source) |srv| {
        do exec(srv) |ctxt| {
            assert vec::is_empty(ctxt.ast.node.module.items);
        }
    }
}

#[test]
fn srv_should_build_ast_map() {
    let source = ~"fn a() { }";
    do from_str(source) |srv| {
        do exec(srv) |ctxt| {
            assert ctxt.ast_map.size() != 0u
        };
    }
}

#[test]
fn should_ignore_external_import_paths_that_dont_exist() {
    let source = ~"use forble; use forble::bippy;";
    from_str(source, |_srv| { } )
}

#[test]
fn srv_should_return_request_result() {
    let source = ~"fn a() { }";
    do from_str(source) |srv| {
        let result = exec(srv, |_ctxt| 1000 );
        assert result == 1000;
    }
}
