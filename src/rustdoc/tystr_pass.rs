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
        fold_enum: fold_enum,
        fold_res: fold_res,
        fold_iface: fold_iface,
        fold_impl: fold_impl
        with *fold::default_seq_fold(srv)
    });
    fold.fold_crate(fold, doc)
}

fn fold_fn(
    fold: fold::fold<astsrv::srv>,
    doc: doc::fndoc
) -> doc::fndoc {

    let srv = fold.ctxt;

    {
        args: merge_arg_tys(srv, doc.id, doc.args),
        return: merge_ret_ty(srv, doc.id, doc.return),
        sig: get_fn_sig(srv, doc.id)
        with doc
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
          _ {
            fail "get_fn_sig: undocumented invariant";
          }
        }
    }
}

#[test]
fn should_add_fn_sig() {
    let doc = test::mk_doc("fn a() -> int { }");
    assert doc.topmod.fns()[0].sig == some("fn a() -> int");
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
            ret_ty_to_str(decl)
          }
          _ { fail "get_ret_ty: undocumented invariant"; }
        }
    }
}

fn ret_ty_to_str(decl: ast::fn_decl) -> option<str> {
    if decl.output.node != ast::ty_nil {
        some(pprust::ty_to_str(decl.output))
    } else {
        // Nil-typed return values are not interesting
        none
    }
}

#[test]
fn should_add_fn_ret_types() {
    let doc = test::mk_doc("fn a() -> int { }");
    assert doc.topmod.fns()[0].return.ty == some("int");
}

#[test]
fn should_not_add_nil_ret_type() {
    let doc = test::mk_doc("fn a() { }");
    assert doc.topmod.fns()[0].return.ty == none;
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
        {
            ty: some(tuple::second(ty))
            with arg
        }
    }
}

fn get_arg_tys(srv: astsrv::srv, fn_id: doc::ast_id) -> [(str, str)] {
    astsrv::exec(srv) {|ctxt|
        alt ctxt.ast_map.get(fn_id) {
          ast_map::node_item(@{
            node: ast::item_fn(decl, _, _), _
          }) |
          ast_map::node_item(@{
            node: ast::item_res(decl, _, _, _, _), _
          }) {
            decl_arg_tys(decl)
          }
          _ {
            fail "get_arg_tys: undocumented invariant";
          }
        }
    }
}

fn decl_arg_tys(decl: ast::fn_decl) -> [(str, str)] {
    vec::map(decl.inputs) {|arg|
        (arg.ident, pprust::ty_to_str(arg.ty))
    }
}

#[test]
fn should_add_arg_types() {
    let doc = test::mk_doc("fn a(b: int, c: bool) { }");
    let fn_ = doc.topmod.fns()[0];
    assert fn_.args[0].ty == some("int");
    assert fn_.args[1].ty == some("bool");
}

fn fold_const(
    fold: fold::fold<astsrv::srv>,
    doc: doc::constdoc
) -> doc::constdoc {
    let srv = fold.ctxt;

    {
        ty: some(astsrv::exec(srv) {|ctxt|
            alt ctxt.ast_map.get(doc.id) {
              ast_map::node_item(@{
                node: ast::item_const(ty, _), _
              }) {
                pprust::ty_to_str(ty)
              }
              _ {
                fail "fold_const: undocumented invariant";
              }
            }
        })
        with doc
    }
}

#[test]
fn should_add_const_types() {
    let doc = test::mk_doc("const a: bool = true;");
    assert doc.topmod.consts()[0].ty == some("bool");
}

fn fold_enum(
    fold: fold::fold<astsrv::srv>,
    doc: doc::enumdoc
) -> doc::enumdoc {
    let srv = fold.ctxt;

    {
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
                  _ { fail "fold_enum: undocumented invariant"; }
                }
            };

            {
                sig: some(sig)
                with variant
            }
        }
        with doc
    }
}

#[test]
fn should_add_variant_sigs() {
    let doc = test::mk_doc("enum a { b(int) }");
    assert doc.topmod.enums()[0].variants[0].sig == some("b(int)");
}

fn fold_res(
    fold: fold::fold<astsrv::srv>,
    doc: doc::resdoc
) -> doc::resdoc {
    let srv = fold.ctxt;

    {
        args: merge_arg_tys(srv, doc.id, doc.args),
        sig: some(astsrv::exec(srv) {|ctxt|
            alt ctxt.ast_map.get(doc.id) {
              ast_map::node_item(@{
                node: ast::item_res(decl, _, _, _, _), _
              }) {
                pprust::res_to_str(decl, doc.name, [])
              }
              _ { fail "fold_res: undocumented invariant"; }
            }
        })
        with doc
    }
}

#[test]
fn should_add_resource_sigs() {
    let doc = test::mk_doc("resource r(b: bool) { }");
    assert doc.topmod.resources()[0].sig == some("resource r(b: bool)");
}

#[test]
fn should_add_resource_arg_tys() {
    let doc = test::mk_doc("resource r(a: bool) { }");
    assert doc.topmod.resources()[0].args[0].ty == some("bool");
}

fn fold_iface(
    fold: fold::fold<astsrv::srv>,
    doc: doc::ifacedoc
) -> doc::ifacedoc {
    {
        methods: merge_methods(fold.ctxt, doc.id, doc.methods)
        with doc
    }
}

fn merge_methods(
    srv: astsrv::srv,
    item_id: doc::ast_id,
    docs: [doc::methoddoc]
) -> [doc::methoddoc] {
    vec::map(docs) {|doc|
        {
            args: merge_method_arg_tys(
                srv,
                item_id,
                doc.args,
                doc.name),
            return: merge_method_ret_ty(
                srv,
                item_id,
                doc.return,
                doc.name),
            sig: get_method_sig(srv, item_id, doc.name)
            with doc
        }
    }
}

fn merge_method_ret_ty(
    srv: astsrv::srv,
    item_id: doc::ast_id,
    doc: doc::retdoc,
    method_name: str
) -> doc::retdoc {
    alt get_method_ret_ty(srv, item_id, method_name) {
      some(ty) {
        {
            ty: some(ty)
            with doc
        }
      }
      none { doc }
    }
}

fn get_method_ret_ty(
    srv: astsrv::srv,
    item_id: doc::ast_id,
    method_name: str
) -> option<str> {
    astsrv::exec(srv) {|ctxt|
        alt ctxt.ast_map.get(item_id) {
          ast_map::node_item(@{
            node: ast::item_iface(_, methods), _
          }) {
            alt vec::find(methods) {|method|
                method.ident == method_name
            } {
                some(method) {
                    ret_ty_to_str(method.decl)
                }
                _ { fail "get_method_ret_ty: undocumented invariant"; }
            }
          }
          ast_map::node_item(@{
            node: ast::item_impl(_, _, _, methods), _
          }) {
            alt vec::find(methods) {|method|
                method.ident == method_name
            } {
                some(method) {
                    ret_ty_to_str(method.decl)
                }
                _ { fail "get_method_ret_ty: undocumented invariant"; }
            }
          }
          _ { fail }
        }
    }
}

fn get_method_sig(
    srv: astsrv::srv,
    item_id: doc::ast_id,
    method_name: str
) -> option<str> {
    astsrv::exec(srv) {|ctxt|
        alt ctxt.ast_map.get(item_id) {
          ast_map::node_item(@{
            node: ast::item_iface(_, methods), _
          }) {
            alt vec::find(methods) {|method|
                method.ident == method_name
            } {
                some(method) {
                    some(pprust::fun_to_str(method.decl, method.ident, []))
                }
                _ { fail "get_method_sig: undocumented invariant"; }
            }
          }
          ast_map::node_item(@{
            node: ast::item_impl(_, _, _, methods), _
          }) {
            alt vec::find(methods) {|method|
                method.ident == method_name
            } {
                some(method) {
                    some(pprust::fun_to_str(method.decl, method.ident, []))
                }
                _ { fail "get_method_sig: undocumented invariant"; }
            }
          }
          _ { fail "get_method_sig: undocumented invariant"; }
        }
    }
}

fn merge_method_arg_tys(
    srv: astsrv::srv,
    item_id: doc::ast_id,
    args: [doc::argdoc],
    method_name: str
) -> [doc::argdoc] {
    let tys = get_method_arg_tys(srv, item_id, method_name);
    vec::map2(args, tys) {|arg, ty|
        assert arg.name == tuple::first(ty);
        {
            ty: some(tuple::second(ty))
            with arg
        }
    }
}

fn get_method_arg_tys(
    srv: astsrv::srv,
    item_id: doc::ast_id,
    method_name: str
) -> [(str, str)] {
    astsrv::exec(srv) {|ctxt|
        alt ctxt.ast_map.get(item_id) {
          ast_map::node_item(@{
            node: ast::item_iface(_, methods), _
          }) {
            alt vec::find(methods) {|method|
                method.ident == method_name
            } {
                some(method) {
                    decl_arg_tys(method.decl)
                }
                _ { fail "get_method_arg_tys: expected method"; }
            }
          }
          ast_map::node_item(@{
            node: ast::item_impl(_, _, _, methods), _
          }) {
            alt vec::find(methods) {|method|
                method.ident == method_name
            } {
                some(method) {
                    decl_arg_tys(method.decl)
                }
                _ { fail "get_method_arg_tys: expected method"; }
            }
          }
          _ { fail }
        }
    }
}

#[test]
fn should_add_iface_method_sigs() {
    let doc = test::mk_doc("iface i { fn a() -> int; }");
    assert doc.topmod.ifaces()[0].methods[0].sig == some("fn a() -> int");
}

#[test]
fn should_add_iface_method_ret_types() {
    let doc = test::mk_doc("iface i { fn a() -> int; }");
    assert doc.topmod.ifaces()[0].methods[0].return.ty == some("int");
}

#[test]
fn should_not_add_iface_method_nil_ret_type() {
    let doc = test::mk_doc("iface i { fn a(); }");
    assert doc.topmod.ifaces()[0].methods[0].return.ty == none;
}

#[test]
fn should_add_iface_method_arg_types() {
    let doc = test::mk_doc("iface i { fn a(b: int, c: bool); }");
    let fn_ = doc.topmod.ifaces()[0].methods[0];
    assert fn_.args[0].ty == some("int");
    assert fn_.args[1].ty == some("bool");
}

fn fold_impl(
    fold: fold::fold<astsrv::srv>,
    doc: doc::impldoc
) -> doc::impldoc {

    let srv = fold.ctxt;

    let (iface_ty, self_ty) = astsrv::exec(srv) {|ctxt|
        alt ctxt.ast_map.get(doc.id) {
          ast_map::node_item(@{
            node: ast::item_impl(_, iface_ty, self_ty, _), _
          }) {
            let iface_ty = option::map(iface_ty) {|iface_ty|
                pprust::ty_to_str(iface_ty)
            };
            (iface_ty, some(pprust::ty_to_str(self_ty)))
          }
          _ { fail "expected impl" }
        }
    };

    {
        iface_ty: iface_ty,
        self_ty: self_ty,
        methods: merge_methods(fold.ctxt, doc.id, doc.methods)
        with doc
    }
}

#[test]
fn should_add_impl_iface_ty() {
    let doc = test::mk_doc("impl i of j for int { fn a() { } }");
    assert doc.topmod.impls()[0].iface_ty == some("j");
}

#[test]
fn should_not_add_impl_iface_ty_if_none() {
    let doc = test::mk_doc("impl i for int { fn a() { } }");
    assert doc.topmod.impls()[0].iface_ty == none;
}

#[test]
fn should_add_impl_self_ty() {
    let doc = test::mk_doc("impl i for int { fn a() { } }");
    assert doc.topmod.impls()[0].self_ty == some("int");
}

#[test]
fn should_add_impl_method_sigs() {
    let doc = test::mk_doc("impl i for int { fn a() -> int { fail } }");
    assert doc.topmod.impls()[0].methods[0].sig == some("fn a() -> int");
}

#[test]
fn should_add_impl_method_ret_types() {
    let doc = test::mk_doc("impl i for int { fn a() -> int { fail } }");
    assert doc.topmod.impls()[0].methods[0].return.ty == some("int");
}

#[test]
fn should_not_add_impl_method_nil_ret_type() {
    let doc = test::mk_doc("impl i for int { fn a() { } }");
    assert doc.topmod.impls()[0].methods[0].return.ty == none;
}

#[test]
fn should_add_impl_method_arg_types() {
    let doc = test::mk_doc("impl i for int { fn a(b: int, c: bool) { } }");
    let fn_ = doc.topmod.impls()[0].methods[0];
    assert fn_.args[0].ty == some("int");
    assert fn_.args[1].ty == some("bool");
}

#[cfg(test)]
mod test {
    fn mk_doc(source: str) -> doc::cratedoc {
        let srv = astsrv::mk_srv_from_str(source);
        let doc = extract::from_srv(srv, "");
        run(srv, doc)
    }
}