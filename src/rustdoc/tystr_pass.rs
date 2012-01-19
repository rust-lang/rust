#[doc =
  "Pulls type information out of the AST and attaches it to the document"];

import rustc::syntax::ast;
import rustc::syntax::print::pprust;
import rustc::middle::ast_map;

export mk_pass;

fn mk_pass() -> pass {
    run
}

fn run(
    srv: astsrv::srv,
    doc: doc::cratedoc
) -> doc::cratedoc {
    let fold = fold::fold({
        fold_fn: fn~(
            f: fold::fold<astsrv::srv>,
            d: doc::fndoc
        ) -> doc::fndoc {
            fold_fn(f, d)
        }
        with *fold::default_seq_fold(srv)
    });
    fold.fold_crate(fold, doc)
}

fn fold_fn(
    fold: fold::fold<astsrv::srv>,
    doc: doc::fndoc
) -> doc::fndoc {

    let srv = fold.ctxt;

    ~{
        return: merge_ret_ty(srv, doc.id, doc.return),
        args: merge_arg_tys(srv, doc.id, doc.args)
        with *doc
    }
}

fn merge_ret_ty(
    srv: astsrv::srv,
    fn_id: doc::ast_id,
    doc: option<doc::retdoc>
) -> option<doc::retdoc> {
    let ty = get_ret_ty(srv, fn_id);
    alt doc {
      some(doc) {
        fail "unimplemented";
      }
      none. {
        some({
            desc: none,
            ty: some(ty)
        })
      }
    }
}

fn get_ret_ty(srv: astsrv::srv, id: doc::ast_id) -> str {
    astsrv::exec(srv) {|ctxt|
        alt ctxt.map.get(id) {
          ast_map::node_item(@{
            node: ast::item_fn(decl, _, _), _
          }) {
            pprust::ty_to_str(decl.output)
          }
        }
    }
}

#[test]
fn should_add_fn_ret_types() {
    let source = "fn a() -> int { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert option::get(doc.topmod.fns[0].return).ty == some("int");
}

fn merge_arg_tys(
    srv: astsrv::srv,
    fn_id: doc::ast_id,
    args: [doc::argdoc]
) -> [doc::argdoc] {
    let tys = get_arg_tys(srv, fn_id);
    vec::map2(args, tys) {|arg, ty|
        // Sanity check that we're talking about the same args
        assert arg.name == tuple::first(ty);
        ~{
            ty: some(tuple::second(ty))
            with *arg
        }
    }
}

fn get_arg_tys(srv: astsrv::srv, fn_id: doc::ast_id) -> [(str, str)] {
    astsrv::exec(srv) {|ctxt|
        alt ctxt.map.get(fn_id) {
          ast_map::node_item(@{
            node: ast::item_fn(decl, _, _), _
          }) {
            vec::map(decl.inputs) {|arg|
                (arg.ident, pprust::ty_to_str(arg.ty))
            }
          }
        }
    }
}

#[test]
fn should_add_arg_types() {
    let source = "fn a(b: int, c: bool) { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    let fn_ = doc.topmod.fns[0];
    assert fn_.args[0].ty == some("int");
    assert fn_.args[1].ty == some("bool");
}