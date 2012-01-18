#[doc(
    brief = "Provides all access to AST-related, non-sendable info",
    desc =
    "Rustdoc is intended to be parallel, and the rustc AST is filled \
     with shared boxes. The AST service attempts to provide a single \
     place to query AST-related information, shielding the rest of \
     Rustdoc from its non-sendableness."
)];

import rustc::syntax::ast;
import rustc::middle::ast_map;

export ctxt;
export ctxt_handler;
export srv;
export mk_srv_from_str;
export mk_srv_from_file;
export exec;

type ctxt = {
    ast: @ast::crate,
    map: ast_map::map
};

type ctxt_handler<T> = fn~(ctxt: ctxt) -> T;

type srv = {
    ctxt: ctxt
};

fn mk_srv_from_str(source: str) -> srv {
    {
        ctxt: build_ctxt(parse::from_str(source))
    }
}

fn mk_srv_from_file(file: str) -> srv {
    {
        ctxt: build_ctxt(parse::from_file(file))
    }
}

fn build_ctxt(ast: @ast::crate) -> ctxt {

    import rustc::front::config;

    let ast = config::strip_unconfigured_items(ast);

    {
        ast: ast,
        map: ast_map::map_crate(*ast)
    }
}

#[test]
fn should_prune_unconfigured_items() {
    let source = "#[cfg(shut_up_and_leave_me_alone)]fn a() { }";
    let srv = mk_srv_from_str(source);
    exec(srv) {|ctxt|
        assert vec::is_empty(ctxt.ast.node.module.items);
    }
}

fn exec<T>(
    srv: srv,
    f: fn~(ctxt: ctxt) -> T
) -> T {
    f(srv.ctxt)
}

#[cfg(test)]
mod tests {

    #[test]
    fn srv_should_build_ast_map() {
        let source = "fn a() { }";
        let srv = mk_srv_from_str(source);
        exec(srv) {|ctxt|
            assert ctxt.map.size() != 0u
        };
    }

    #[test]
    fn srv_should_return_request_result() {
        let source = "fn a() { }";
        let srv = mk_srv_from_str(source);
        let result = exec(srv) {|_ctxt| 1000};
        assert result == 1000;
    }
}