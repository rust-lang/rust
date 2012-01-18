#[doc = "Provides all access to AST-related, non-sendable info"];

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
    {
        ast: ast,
        map: ast_map::map_crate(*ast)
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