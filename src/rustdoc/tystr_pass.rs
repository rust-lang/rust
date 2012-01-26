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
        fold_fn: fold_fn,
        fold_const: fold_const,
        fold_enum: fold_enum
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
        args: merge_arg_tys(srv, doc.id, doc.args),
        return: merge_ret_ty(srv, doc.id, doc.return),
        sig: get_fn_sig(srv, doc.id)
        with *doc
    }
}

fn get_fn_sig(srv: astsrv::srv, fn_id: doc::ast_id) -> option<str> {
    astsrv::exec(srv) {|ctxt|
        alt ctxt.ast_map.get(fn_id) {
          ast_map::node_item(@{
            ident: ident,
            node: ast::item_fn(decl, _, blk), _
          }) {
            some(pprust::fun_to_str(decl, ident, []))
          }
        }
    }
}

#[test]
fn should_add_fn_sig() {
    let source = "fn a() -> int { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert doc.topmod.fns[0].sig == some("fn a() -> int");
}

fn merge_ret_ty(
    srv: astsrv::srv,
    fn_id: doc::ast_id,
    doc: doc::retdoc
) -> doc::retdoc {
    alt get_ret_ty(srv, fn_id) {
      some(ty) {
        {
            ty: some(ty)
            with doc
        }
      }
      none { doc }
    }
}

fn get_ret_ty(srv: astsrv::srv, fn_id: doc::ast_id) -> option<str> {
    astsrv::exec(srv) {|ctxt|
        alt ctxt.ast_map.get(fn_id) {
          ast_map::node_item(@{
            node: ast::item_fn(decl, _, _), _
          }) {
            if decl.output.node != ast::ty_nil {
                some(pprust::ty_to_str(decl.output))
            } else {
                // Nil-typed return values are not interesting
                none
            }
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
    assert doc.topmod.fns[0].return.ty == some("int");
}

#[test]
fn should_not_add_nil_ret_type() {
    let source = "fn a() { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert doc.topmod.fns[0].return.ty == none;
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
        alt ctxt.ast_map.get(fn_id) {
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

fn fold_const(
    fold: fold::fold<astsrv::srv>,
    doc: doc::constdoc
) -> doc::constdoc {
    let srv = fold.ctxt;

    ~{
        ty: some(astsrv::exec(srv) {|ctxt|
            alt ctxt.ast_map.get(doc.id) {
              ast_map::node_item(@{
                node: ast::item_const(ty, _), _
              }) {
                pprust::ty_to_str(ty)
              }
            }
        })
        with *doc
    }
}

#[test]
fn should_add_const_types() {
    let source = "const a: bool = true;";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert doc.topmod.consts[0].ty == some("bool");
}

fn fold_enum(
    fold: fold::fold<astsrv::srv>,
    doc: doc::enumdoc
) -> doc::enumdoc {
    let srv = fold.ctxt;

    ~{
        variants: vec::map(doc.variants) {|variant|
            let sig = astsrv::exec(srv) {|ctxt|
                alt ctxt.ast_map.get(doc.id) {
                  ast_map::node_item(@{
                    node: ast::item_enum(ast_variants, _), _
                  }) {
                    let ast_variant = option::get(
                        vec::find(ast_variants) {|v|
                            v.node.name == variant.name
                        });

                    pprust::variant_to_str(ast_variant)
                  }
                }
            };

            ~{
                sig: some(sig)
                with *variant
            }
        }
        with *doc
    }
}

#[test]
fn should_add_variant_sigs() {
    let source = "enum a { b(int) }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert doc.topmod.enums[0].variants[0].sig == some("b(int)");
}
