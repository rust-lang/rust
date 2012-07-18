//! Pulls type information out of the AST and attaches it to the document

import doc::item_utils;
import syntax::ast;
import syntax::print::pprust;
import syntax::ast_map;
import std::map::hashmap;
import extract::to_str;

export mk_pass;

fn mk_pass() -> pass {
    {
        name: ~"tystr",
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
        fold_trait: fold_trait,
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

fn get_fn_sig(srv: astsrv::srv, fn_id: doc::ast_id) -> option<~str> {
    do astsrv::exec(srv) |ctxt| {
        match check ctxt.ast_map.get(fn_id) {
          ast_map::node_item(@{
            ident: ident,
            node: ast::item_fn(decl, tys, _), _
          }, _) |
          ast_map::node_foreign_item(@{
            ident: ident,
            node: ast::foreign_item_fn(decl, tys), _
          }, _, _) => {
            some(pprust::fun_to_str(decl, ident, tys, extract::interner()))
          }
        }
    }
}

#[test]
fn should_add_fn_sig() {
    let doc = test::mk_doc(~"fn a<T>() -> int { }");
    assert doc.cratemod().fns()[0].sig == some(~"fn a<T>() -> int");
}

#[test]
fn should_add_foreign_fn_sig() {
    let doc = test::mk_doc(~"extern mod a { fn a<T>() -> int; }");
    assert doc.cratemod().nmods()[0].fns[0].sig == some(~"fn a<T>() -> int");
}

fn fold_const(
    fold: fold::fold<astsrv::srv>,
    doc: doc::constdoc
) -> doc::constdoc {
    let srv = fold.ctxt;

    {
        sig: some(do astsrv::exec(srv) |ctxt| {
            match check ctxt.ast_map.get(doc.id()) {
              ast_map::node_item(@{
                node: ast::item_const(ty, _), _
              }, _) => {
                pprust::ty_to_str(ty, extract::interner())
              }
            }
        })
        with doc
    }
}

#[test]
fn should_add_const_types() {
    let doc = test::mk_doc(~"const a: bool = true;");
    assert doc.cratemod().consts()[0].sig == some(~"bool");
}

fn fold_enum(
    fold: fold::fold<astsrv::srv>,
    doc: doc::enumdoc
) -> doc::enumdoc {
    let doc_id = doc.id();
    let srv = fold.ctxt;

    {
        variants: do par::map(doc.variants) |variant| {
            let sig = do astsrv::exec(srv) |ctxt| {
                match check ctxt.ast_map.get(doc_id) {
                  ast_map::node_item(@{
                    node: ast::item_enum(enum_definition, _), _
                  }, _) => {
                    let ast_variant = option::get(
                        do vec::find(enum_definition.variants) |v| {
                            to_str(v.node.name) == variant.name
                        });

                    pprust::variant_to_str(ast_variant, extract::interner())
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
    let doc = test::mk_doc(~"enum a { b(int) }");
    assert doc.cratemod().enums()[0].variants[0].sig == some(~"b(int)");
}

fn fold_trait(
    fold: fold::fold<astsrv::srv>,
    doc: doc::traitdoc
) -> doc::traitdoc {
    {
        methods: merge_methods(fold.ctxt, doc.id(), doc.methods)
        with doc
    }
}

fn merge_methods(
    srv: astsrv::srv,
    item_id: doc::ast_id,
    docs: ~[doc::methoddoc]
) -> ~[doc::methoddoc] {
    do par::map(docs) |doc| {
        {
            sig: get_method_sig(srv, item_id, doc.name)
            with doc
        }
    }
}

fn get_method_sig(
    srv: astsrv::srv,
    item_id: doc::ast_id,
    method_name: ~str
) -> option<~str> {
    do astsrv::exec(srv) |ctxt| {
        match check ctxt.ast_map.get(item_id) {
          ast_map::node_item(@{
            node: ast::item_trait(_, _, methods), _
          }, _) => {
            match check vec::find(methods, |method| {
                match method {
                  ast::required(ty_m) => to_str(ty_m.ident) == method_name,
                  ast::provided(m) => to_str(m.ident) == method_name,
                }
            }) {
                some(method) => {
                  match method {
                    ast::required(ty_m) => {
                      some(pprust::fun_to_str(
                          ty_m.decl,
                          ty_m.ident,
                          ty_m.tps,
                          extract::interner()
                      ))
                    }
                    ast::provided(m) => {
                      some(pprust::fun_to_str(
                          m.decl,
                          m.ident,
                          m.tps,
                          extract::interner()
                      ))
                    }
                  }
                }
            }
          }
          ast_map::node_item(@{
            node: ast::item_impl(_, _, _, methods), _
          }, _) => {
            match check vec::find(methods, |method| {
                to_str(method.ident) == method_name
            }) {
                some(method) => {
                    some(pprust::fun_to_str(
                        method.decl,
                        method.ident,
                        method.tps,
                        extract::interner()
                    ))
                }
            }
          }
        }
    }
}

#[test]
fn should_add_trait_method_sigs() {
    let doc = test::mk_doc(~"trait i { fn a<T>() -> int; }");
    assert doc.cratemod().traits()[0].methods[0].sig
        == some(~"fn a<T>() -> int");
}

fn fold_impl(
    fold: fold::fold<astsrv::srv>,
    doc: doc::impldoc
) -> doc::impldoc {

    let srv = fold.ctxt;

    let (trait_types, self_ty) = do astsrv::exec(srv) |ctxt| {
        match ctxt.ast_map.get(doc.id()) {
          ast_map::node_item(@{
            node: ast::item_impl(_, trait_types, self_ty, _), _
          }, _) => {
            let trait_types = vec::map(trait_types, |p| {
                pprust::path_to_str(p.path, extract::interner())
            });
            (trait_types, some(pprust::ty_to_str(self_ty,
                                                 extract::interner())))
          }
          _ => fail ~"expected impl"
        }
    };

    {
        trait_types: trait_types,
        self_ty: self_ty,
        methods: merge_methods(fold.ctxt, doc.id(), doc.methods)
        with doc
    }
}

#[test]
fn should_add_impl_trait_types() {
    let doc = test::mk_doc(~"impl int: j { fn a<T>() { } }");
    assert doc.cratemod().impls()[0].trait_types[0] == ~"j";
}

#[test]
fn should_not_add_impl_trait_types_if_none() {
    let doc = test::mk_doc(~"impl int { fn a() { } }");
    assert vec::len(doc.cratemod().impls()[0].trait_types) == 0;
}

#[test]
fn should_add_impl_self_ty() {
    let doc = test::mk_doc(~"impl int { fn a() { } }");
    assert doc.cratemod().impls()[0].self_ty == some(~"int");
}

#[test]
fn should_add_impl_method_sigs() {
    let doc = test::mk_doc(~"impl int { fn a<T>() -> int { fail } }");
    assert doc.cratemod().impls()[0].methods[0].sig
        == some(~"fn a<T>() -> int");
}

fn fold_type(
    fold: fold::fold<astsrv::srv>,
    doc: doc::tydoc
) -> doc::tydoc {

    let srv = fold.ctxt;

    {
        sig: do astsrv::exec(srv) |ctxt| {
            match ctxt.ast_map.get(doc.id()) {
              ast_map::node_item(@{
                ident: ident,
                node: ast::item_ty(ty, params), _
              }, _) => {
                some(fmt!{
                    "type %s%s = %s",
                    to_str(ident),
                    pprust::typarams_to_str(params, extract::interner()),
                    pprust::ty_to_str(ty, extract::interner())
                })
              }
              _ => fail ~"expected type"
            }
        }
        with doc
    }
}

#[test]
fn should_add_type_signatures() {
    let doc = test::mk_doc(~"type t<T> = int;");
    assert doc.cratemod().types()[0].sig == some(~"type t<T> = int");
}

#[cfg(test)]
mod test {
    fn mk_doc(source: ~str) -> doc::doc {
        do astsrv::from_str(source) |srv| {
            let doc = extract::from_srv(srv, ~"");
            run(srv, doc)
        }
    }
}
