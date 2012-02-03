#[doc(
    brief = "Provides all access to AST-related, non-sendable info",
    desc =
    "Rustdoc is intended to be parallel, and the rustc AST is filled \
     with shared boxes. The AST service attempts to provide a single \
     place to query AST-related information, shielding the rest of \
     Rustdoc from its non-sendableness."
)];

import rustc::driver::session;
import rustc::driver::driver;
import rustc::driver::diagnostic;
import rustc::syntax::ast;
import rustc::middle::ast_map;
import rustc::back::link;
import rustc::util::filesearch;
import rustc::front;

export ctxt;
export ctxt_handler;
export srv;
export mk_srv_from_str;
export mk_srv_from_file;
export exec;

type ctxt = {
    ast: @ast::crate,
    ast_map: ast_map::map
};

type ctxt_handler<T> = fn~(ctxt: ctxt) -> T;

type srv = {
    ctxt: ctxt
};

fn mk_srv_from_str(source: str) -> srv {
    let sess = build_session();
    {
        ctxt: build_ctxt(sess, parse::from_str_sess(sess, source))
    }
}

fn mk_srv_from_file(file: str) -> srv {
    let sess = build_session();
    {
        ctxt: build_ctxt(sess, parse::from_file_sess(sess, file))
    }
}

fn build_ctxt(sess: session::session, ast: @ast::crate) -> ctxt {

    import rustc::front::config;

    let ast = config::strip_unconfigured_items(ast);
    let ast = front::test::modify_for_testing(sess, ast);
    let ast_map = ast_map::map_crate(*ast);

    {
        ast: ast,
        ast_map: ast_map,
    }
}

fn build_session() -> session::session {
    let sopts: @session::options = @{
        crate_type: session::lib_crate,
        static: false,
        optimize: 0u,
        debuginfo: false,
        extra_debuginfo: false,
        verify: false,
        lint_opts: [],
        save_temps: false,
        stats: false,
        time_passes: false,
        time_llvm_passes: false,
        output_type: link::output_type_exe,
        addl_lib_search_paths: [],
        maybe_sysroot: none,
        target_triple: driver::host_triple(),
        cfg: [],
        test: false,
        parse_only: false,
        no_trans: false,
        no_asm_comments: false,
        warn_unused_imports: false
    };
    driver::build_session(sopts, ".", diagnostic::emit)
}

#[test]
fn should_prune_unconfigured_items() {
    let source = "#[cfg(shut_up_and_leave_me_alone)]fn a() { }";
    let srv = mk_srv_from_str(source);
    exec(srv) {|ctxt|
        assert vec::is_empty(ctxt.ast.node.module.items);
    }
}

#[test]
fn srv_should_build_ast_map() {
    let source = "fn a() { }";
    let srv = mk_srv_from_str(source);
    exec(srv) {|ctxt|
        assert ctxt.ast_map.size() != 0u
    };
}

#[test]
#[ignore]
fn srv_should_build_reexport_map() {
    // FIXME
    /*let source = "import a::b; export b; mod a { mod b { } }";
    let srv = mk_srv_from_str(source);
    exec(srv) {|ctxt|
        assert ctxt.exp_map.size() != 0u
    };*/
}

#[test]
fn srv_should_resolve_external_crates() {
    let source = "use std;\
                  fn f() -> std::sha1::sha1 {\
                  std::sha1::mk_sha1() }";
    // Just testing that resolve doesn't crash
    mk_srv_from_str(source);
}

#[test]
fn srv_should_resolve_core_crate() {
    let source = "fn a() -> option { fail }";
    // Just testing that resolve doesn't crash
    mk_srv_from_str(source);
}

fn exec<T>(
    srv: srv,
    f: fn~(ctxt: ctxt) -> T
) -> T {
    f(srv.ctxt)
}

#[test]
fn srv_should_return_request_result() {
    let source = "fn a() { }";
    let srv = mk_srv_from_str(source);
    let result = exec(srv) {|_ctxt| 1000};
    assert result == 1000;
}
