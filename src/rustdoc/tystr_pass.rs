#[doc =
  "Pulls type information out of the AST and attaches it to the document"];

import rustc::syntax::ast;
import rustc::syntax::print::pprust;
import rustc::middle::ast_map;
import std::map::hashmap;

export mk_pass;

fn mk_pass() -> pass {
    {
        name: "tystr",
        f: run
    }
}

fn run(
    srv: astsrv::srv,
    doc: doc::doc
) -> doc::doc {
    let fold = fold::fold({
        fold_fn: fold_fn,
        fold_const: fold_const,
        fold_enum: fold_enum,
        fold_res: fold_res,
        fold_iface: fold_iface,
        fold_impl: fold_impl,
        fold_type: fold_type
        with *fold::default_any_fold(srv)
    });
    fold.fold_doc(fold, doc)
}

fn fold_fn(
    fold: fold::fold<astsrv::srv>,
    doc: doc::fndoc
) -> doc::fndoc {

    let srv = fold.ctxt;

    {
        sig: get_fn_sig(srv, doc.id())
        with doc
    }
}

fn get_fn_sig(srv: astsrv::srv, fn_id: doc::ast_id) -> option<str> {
    astsrv::exec(srv) {|ctxt|
        alt check ctxt.ast_map.get(fn_id) {
          ast_map::node_item(@{
            ident: ident,
            node: ast::item_fn(decl, _, _), _
          }, _) |
          ast_map::node_native_item(@{
            ident: ident,
            node: ast::native_item_fn(decl, _), _
          }, _, _) {
            some(pprust::fun_to_str(decl, ident, []))
          }
        }
    }
}

#[test]
fn should_add_fn_sig() {
    let doc = test::mk_doc("fn a() -> int { }");
    assert doc.cratemod().fns()[0].sig == some("fn a() -> int");
}

#[test]
fn should_add_native_fn_sig() {
    let doc = test::mk_doc("native mod a { fn a() -> int; }");
    assert doc.cratemod().nmods()[0].fns[0].sig == some("fn a() -> int");
}

fn fold_const(
    fold: fold::fold<astsrv::srv>,
    doc: doc::constdoc
) -> doc::constdoc {
    let srv = fold.ctxt;

    {
        sig: some(astsrv::exec(srv) {|ctxt|
            alt check ctxt.ast_map.get(doc.id()) {
              ast_map::node_item(@{
                node: ast::item_const(ty, _), _
              }, _) {
                pprust::ty_to_str(ty)
              }
            }
        })
        with doc
    }
}

#[test]
fn should_add_const_types() {
    let doc = test::mk_doc("const a: bool = true;");
    assert doc.cratemod().consts()[0].sig == some("bool");
}

fn fold_enum(
    fold: fold::fold<astsrv::srv>,
    doc: doc::enumdoc
) -> doc::enumdoc {
    let doc_id = doc.id();
    let srv = fold.ctxt;

    {
        variants: par::anymap(doc.variants) {|variant|
            let sig = astsrv::exec(srv) {|ctxt|
                alt check ctxt.ast_map.get(doc_id) {
                  ast_map::node_item(@{
                    node: ast::item_enum(ast_variants, _), _
                  }, _) {
                    let ast_variant = option::get(
                        vec::find(ast_variants) {|v|
                            v.node.name == variant.name
                        });

                    pprust::variant_to_str(ast_variant)
                  }
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
    assert doc.cratemod().enums()[0].variants[0].sig == some("b(int)");
}

fn fold_res(
    fold: fold::fold<astsrv::srv>,
    doc: doc::resdoc
) -> doc::resdoc {
    let srv = fold.ctxt;

    {
        sig: some(astsrv::exec(srv) {|ctxt|
            alt check ctxt.ast_map.get(doc.id()) {
              ast_map::node_item(@{
                node: ast::item_res(decl, _, _, _, _), _
              }, _) {
                pprust::res_to_str(decl, doc.name(), [])
              }
            }
        })
        with doc
    }
}

#[test]
fn should_add_resource_sigs() {
    let doc = test::mk_doc("resource r(b: bool) { }");
    assert doc.cratemod().resources()[0].sig == some("resource r(b: bool)");
}

fn fold_iface(
    fold: fold::fold<astsrv::srv>,
    doc: doc::ifacedoc
) -> doc::ifacedoc {
    {
        methods: merge_methods(fold.ctxt, doc.id(), doc.methods)
        with doc
    }
}

fn merge_methods(
    srv: astsrv::srv,
    item_id: doc::ast_id,
    docs: [doc::methoddoc]
) -> [doc::methoddoc] {
    par::anymap(docs) {|doc|
        {
            sig: get_method_sig(srv, item_id, doc.name)
            with doc
        }
    }
}

fn get_method_sig(
    srv: astsrv::srv,
    item_id: doc::ast_id,
    method_name: str
) -> option<str> {
    astsrv::exec(srv) {|ctxt|
        alt check ctxt.ast_map.get(item_id) {
          ast_map::node_item(@{
            node: ast::item_iface(_, methods), _
          }, _) {
            alt check vec::find(methods) {|method|
                method.ident == method_name
            } {
                some(method) {
                    some(pprust::fun_to_str(method.decl, method.ident, []))
                }
            }
          }
          ast_map::node_item(@{
            node: ast::item_impl(_, _, _, methods), _
          }, _) {
            alt check vec::find(methods) {|method|
                method.ident == method_name
            } {
                some(method) {
                    some(pprust::fun_to_str(method.decl, method.ident, []))
                }
            }
          }
        }
    }
}

#[test]
fn should_add_iface_method_sigs() {
    let doc = test::mk_doc("iface i { fn a() -> int; }");
    assert doc.cratemod().ifaces()[0].methods[0].sig == some("fn a() -> int");
}

fn fold_impl(
    fold: fold::fold<astsrv::srv>,
    doc: doc::impldoc
) -> doc::impldoc {

    let srv = fold.ctxt;

    let (iface_ty, self_ty) = astsrv::exec(srv) {|ctxt|
        alt ctxt.ast_map.get(doc.id()) {
          ast_map::node_item(@{
            node: ast::item_impl(_, iface_ty, self_ty, _), _
          }, _) {
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
        methods: merge_methods(fold.ctxt, doc.id(), doc.methods)
        with doc
    }
}

#[test]
fn should_add_impl_iface_ty() {
    let doc = test::mk_doc("impl i of j for int { fn a() { } }");
    assert doc.cratemod().impls()[0].iface_ty == some("j");
}

#[test]
fn should_not_add_impl_iface_ty_if_none() {
    let doc = test::mk_doc("impl i for int { fn a() { } }");
    assert doc.cratemod().impls()[0].iface_ty == none;
}

#[test]
fn should_add_impl_self_ty() {
    let doc = test::mk_doc("impl i for int { fn a() { } }");
    assert doc.cratemod().impls()[0].self_ty == some("int");
}

#[test]
fn should_add_impl_method_sigs() {
    let doc = test::mk_doc("impl i for int { fn a() -> int { fail } }");
    assert doc.cratemod().impls()[0].methods[0].sig == some("fn a() -> int");
}

fn fold_type(
    fold: fold::fold<astsrv::srv>,
    doc: doc::tydoc
) -> doc::tydoc {

    let srv = fold.ctxt;

    {
        sig: astsrv::exec(srv) {|ctxt|
            alt ctxt.ast_map.get(doc.id()) {
              ast_map::node_item(@{
                ident: ident,
                node: ast::item_ty(ty, params), _
              }, _) {
                some(#fmt(
                    "type %s%s = %s",
                    ident,
                    pprust::typarams_to_str(params),
                    pprust::ty_to_str(ty)
                ))
              }
              _ { fail "expected type" }
            }
        }
        with doc
    }
}

#[test]
fn should_add_type_signatures() {
    let doc = test::mk_doc("type t<T> = int;");
    assert doc.cratemod().types()[0].sig == some("type t<T> = int");
}

#[cfg(test)]
mod test {
    fn mk_doc(source: str) -> doc::doc {
        astsrv::from_str(source) {|srv|
            let doc = extract::from_srv(srv, "");
            run(srv, doc)
        }
    }
}
