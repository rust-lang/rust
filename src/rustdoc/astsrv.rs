#[doc = "Provides all access to AST-related, non-sendable info"];

import rustc::syntax::ast;
import rustc::middle::ast_map;

export ctxt;
export ctxt_handler;
export srv, seq_srv;
export mk_seq_srv_from_str;
export mk_seq_srv_from_file;

type ctxt = {
    ast: @ast::crate,
    map: ast_map::map
};

type ctxt_handler<T> = fn~(ctxt: ctxt) -> T;

iface srv {
    fn exec<T>(f: ctxt_handler<T>) -> T;
}

#[doc = "The single-task service"]
tag seq_srv = ctxt;

impl seq_srv of srv for seq_srv {
    fn exec<T>(f: ctxt_handler<T>) -> T {
        f(*self)
    }
}

fn mk_seq_srv_from_str(source: str) -> seq_srv {
    seq_srv(build_ctxt(parse::from_str(source)))
}

fn mk_seq_srv_from_file(file: str) -> seq_srv {
    seq_srv(build_ctxt(parse::from_file(file)))
}

fn build_ctxt(ast: @ast::crate) -> ctxt {
    {
        ast: ast,
        map: ast_map::map_crate(*ast)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn seq_srv_should_build_ast_map() {
        let source = "fn a() { }";
        let srv = mk_seq_srv_from_str(source);
        srv.exec {|ctxt|
            assert ctxt.map.size() != 0u
        };
    }

    #[test]
    fn seq_srv_should_return_request_result() {
        let source = "fn a() { }";
        let srv = mk_seq_srv_from_str(source);
        let result = srv.exec {|_ctxt| 1000};
        assert result == 1000;
    }
}