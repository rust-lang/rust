#[doc(
    brief = "Provides all access to AST-related, non-sendable info",
    desc =
    "Rustdoc is intended to be parallel, and the rustc AST is filled \
     with shared boxes. The AST service attempts to provide a single \
     place to query AST-related information, shielding the rest of \
     Rustdoc from its non-sendableness."
)];

import std::map::hashmap;
import rustc::driver::session;
import session::{basic_options, options};
import session::session;
import rustc::driver::driver;
import syntax::diagnostic;
import syntax::diagnostic::handler;
import syntax::ast;
import syntax::codemap;
import syntax::ast_map;
import rustc::back::link;
import rustc::metadata::filesearch;
import rustc::front;

export ctxt;
export ctxt_handler;
export srv;
export from_str;
export from_file;
export exec;

type ctxt = {
    ast: @ast::crate,
    ast_map: ast_map::map
};

type srv_owner<T> = fn(srv: srv) -> T;
type ctxt_handler<T> = fn~(ctxt: ctxt) -> T;
type parser = fn~(session, ~str) -> @ast::crate;

enum msg {
    handle_request(fn~(ctxt)),
    exit
}

enum srv = {
    ch: comm::chan<msg>
};

fn from_str<T>(source: ~str, owner: srv_owner<T>) -> T {
    run(owner, source, parse::from_str_sess)
}

fn from_file<T>(file: ~str, owner: srv_owner<T>) -> T {
    run(owner, file, parse::from_file_sess)
}

fn run<T>(owner: srv_owner<T>, source: ~str, +parse: parser) -> T {

    let srv_ = srv({
        ch: do task::spawn_listener |po| {
            act(po, source, parse);
        }
    });

    let res = owner(srv_);
    comm::send(srv_.ch, exit);
    return res;
}

fn act(po: comm::port<msg>, source: ~str, parse: parser) {
    let sess = build_session();

    let ctxt = build_ctxt(
        sess,
        parse(sess, source)
    );

    let mut keep_going = true;
    while keep_going {
        match comm::recv(po) {
          handle_request(f) => {
            f(ctxt);
          }
          exit => {
            keep_going = false;
          }
        }
    }
}

fn exec<T:send>(
    srv: srv,
    +f: fn~(ctxt: ctxt) -> T
) -> T {
    let po = comm::port();
    let ch = comm::chan(po);
    let msg = handle_request(fn~(move f, ctxt: ctxt) {
        comm::send(ch, f(ctxt))
    });
    comm::send(srv.ch, msg);
    comm::recv(po)
}

fn build_ctxt(sess: session,
              ast: @ast::crate) -> ctxt {

    import rustc::front::config;

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

type error_handlers = {
    emitter: diagnostic::emitter,
    span_handler: diagnostic::span_handler
};

// Build a custom error handler that will allow us to ignore non-fatal
// errors
fn build_error_handlers(
    codemap: codemap::codemap
) -> error_handlers {

    type diagnostic_handler = {
        inner: diagnostic::handler,
    };

    impl diagnostic_handler: diagnostic::handler {
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
        fn emit(cmsp: option<(codemap::codemap, codemap::span)>,
                msg: ~str, lvl: diagnostic::level) {
            self.inner.emit(cmsp, msg, lvl)
        }
    }

    let emitter = fn@(cmsp: option<(codemap::codemap, codemap::span)>,
                       msg: ~str, lvl: diagnostic::level) {
        diagnostic::emit(cmsp, msg, lvl);
    };
    let inner_handler = diagnostic::mk_handler(some(emitter));
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
    let source = ~"use forble; import forble::bippy;";
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
