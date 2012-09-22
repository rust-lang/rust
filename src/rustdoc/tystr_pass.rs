//! Pulls type information out of the AST and attaches it to the document

use doc::ItemUtils;
use syntax::ast;
use syntax::print::pprust;
use syntax::ast_map;
use std::map::HashMap;
use extract::to_str;

export mk_pass;

fn mk_pass() -> Pass {
    {
        name: ~"tystr",
        f: run
    }
}

fn run(
    srv: astsrv::Srv,
    doc: doc::Doc
) -> doc::Doc {
    let fold = fold::Fold({
        fold_fn: fold_fn,
        fold_const: fold_const,
        fold_enum: fold_enum,
        fold_trait: fold_trait,
        fold_impl: fold_impl,
        fold_type: fold_type,
        fold_struct: fold_struct,
        .. *fold::default_any_fold(srv)
    });
    fold.fold_doc(fold, doc)
}

fn fold_fn(
    fold: fold::Fold<astsrv::Srv>,
    doc: doc::FnDoc
) -> doc::FnDoc {

    let srv = fold.ctxt;

    {
        sig: get_fn_sig(srv, doc.id()),
        .. doc
    }
}

fn get_fn_sig(srv: astsrv::Srv, fn_id: doc::AstId) -> Option<~str> {
    do astsrv::exec(srv) |ctxt| {
        match ctxt.ast_map.get(fn_id) {
          ast_map::node_item(@{
            ident: ident,
            node: ast::item_fn(decl, _, tys, _), _
          }, _) |
          ast_map::node_foreign_item(@{
            ident: ident,
            node: ast::foreign_item_fn(decl, _, tys), _
          }, _, _) => {
            Some(pprust::fun_to_str(decl, ident, tys, extract::interner()))
          }
          _ => fail ~"get_fn_sig: fn_id not bound to a fn item"
        }
    }
}

#[test]
fn should_add_fn_sig() {
    let doc = test::mk_doc(~"fn a<T>() -> int { }");
    assert doc.cratemod().fns()[0].sig == Some(~"fn a<T>() -> int");
}

#[test]
fn should_add_foreign_fn_sig() {
    let doc = test::mk_doc(~"extern mod a { fn a<T>() -> int; }");
    assert doc.cratemod().nmods()[0].fns[0].sig == Some(~"fn a<T>() -> int");
}

fn fold_const(
    fold: fold::Fold<astsrv::Srv>,
    doc: doc::ConstDoc
) -> doc::ConstDoc {
    let srv = fold.ctxt;

    {
        sig: Some(do astsrv::exec(srv) |ctxt| {
            match ctxt.ast_map.get(doc.id()) {
              ast_map::node_item(@{
                node: ast::item_const(ty, _), _
              }, _) => {
                pprust::ty_to_str(ty, extract::interner())
              }
              _ => fail ~"fold_const: id not bound to a const item"
            }
        }),
        .. doc
    }
}

#[test]
fn should_add_const_types() {
    let doc = test::mk_doc(~"const a: bool = true;");
    assert doc.cratemod().consts()[0].sig == Some(~"bool");
}

fn fold_enum(
    fold: fold::Fold<astsrv::Srv>,
    doc: doc::EnumDoc
) -> doc::EnumDoc {
    let doc_id = doc.id();
    let srv = fold.ctxt;

    {
        variants: do par::map(doc.variants) |variant| {
            let sig = do astsrv::exec(srv) |ctxt| {
                match ctxt.ast_map.get(doc_id) {
                  ast_map::node_item(@{
                    node: ast::item_enum(enum_definition, _), _
                  }, _) => {
                    let ast_variant = option::get(
                        do vec::find(enum_definition.variants) |v| {
                            to_str(v.node.name) == variant.name
                        });

                    pprust::variant_to_str(ast_variant, extract::interner())
                  }
                  _ => fail ~"enum variant not bound to an enum item"
                }
            };

            {
                sig: Some(sig),
                .. variant
            }
        },
        .. doc
    }
}

#[test]
fn should_add_variant_sigs() {
    let doc = test::mk_doc(~"enum a { b(int) }");
    assert doc.cratemod().enums()[0].variants[0].sig == Some(~"b(int)");
}

fn fold_trait(
    fold: fold::Fold<astsrv::Srv>,
    doc: doc::TraitDoc
) -> doc::TraitDoc {
    {
        methods: merge_methods(fold.ctxt, doc.id(), doc.methods),
        .. doc
    }
}

fn merge_methods(
    srv: astsrv::Srv,
    item_id: doc::AstId,
    docs: ~[doc::MethodDoc]
) -> ~[doc::MethodDoc] {
    do par::map(docs) |doc| {
        {
            sig: get_method_sig(srv, item_id, doc.name),
            .. doc
        }
    }
}

fn get_method_sig(
    srv: astsrv::Srv,
    item_id: doc::AstId,
    method_name: ~str
) -> Option<~str> {
    do astsrv::exec(srv) |ctxt| {
        match ctxt.ast_map.get(item_id) {
          ast_map::node_item(@{
            node: ast::item_trait(_, _, methods), _
          }, _) => {
            match vec::find(methods, |method| {
                match method {
                  ast::required(ty_m) => to_str(ty_m.ident) == method_name,
                  ast::provided(m) => to_str(m.ident) == method_name,
                }
            }) {
                Some(method) => {
                  match method {
                    ast::required(ty_m) => {
                      Some(pprust::fun_to_str(
                          ty_m.decl,
                          ty_m.ident,
                          ty_m.tps,
                          extract::interner()
                      ))
                    }
                    ast::provided(m) => {
                      Some(pprust::fun_to_str(
                          m.decl,
                          m.ident,
                          m.tps,
                          extract::interner()
                      ))
                    }
                  }
                }
                _ => fail ~"method not found"
            }
          }
          ast_map::node_item(@{
            node: ast::item_impl(_, _, _, methods), _
          }, _) => {
            match vec::find(methods, |method| {
                to_str(method.ident) == method_name
            }) {
                Some(method) => {
                    Some(pprust::fun_to_str(
                        method.decl,
                        method.ident,
                        method.tps,
                        extract::interner()
                    ))
                }
                None => fail ~"method not found"
            }
          }
          _ => fail ~"get_method_sig: item ID not bound to trait or impl"
        }
    }
}

#[test]
fn should_add_trait_method_sigs() {
    let doc = test::mk_doc(~"trait i { fn a<T>() -> int; }");
    assert doc.cratemod().traits()[0].methods[0].sig
        == Some(~"fn a<T>() -> int");
}

fn fold_impl(
    fold: fold::Fold<astsrv::Srv>,
    doc: doc::ImplDoc
) -> doc::ImplDoc {

    let srv = fold.ctxt;

    let (trait_types, self_ty) = do astsrv::exec(srv) |ctxt| {
        match ctxt.ast_map.get(doc.id()) {
          ast_map::node_item(@{
            node: ast::item_impl(_, opt_trait_type, self_ty, _), _
          }, _) => {
            let trait_types = opt_trait_type.map_default(~[], |p| {
                ~[pprust::path_to_str(p.path, extract::interner())]
            });
            (trait_types, Some(pprust::ty_to_str(self_ty,
                                                 extract::interner())))
          }
          _ => fail ~"expected impl"
        }
    };

    {
        trait_types: trait_types,
        self_ty: self_ty,
        methods: merge_methods(fold.ctxt, doc.id(), doc.methods),
        .. doc
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
    assert doc.cratemod().impls()[0].self_ty == Some(~"int");
}

#[test]
fn should_add_impl_method_sigs() {
    let doc = test::mk_doc(~"impl int { fn a<T>() -> int { fail } }");
    assert doc.cratemod().impls()[0].methods[0].sig
        == Some(~"fn a<T>() -> int");
}

fn fold_type(
    fold: fold::Fold<astsrv::Srv>,
    doc: doc::TyDoc
) -> doc::TyDoc {

    let srv = fold.ctxt;

    {
        sig: do astsrv::exec(srv) |ctxt| {
            match ctxt.ast_map.get(doc.id()) {
              ast_map::node_item(@{
                ident: ident,
                node: ast::item_ty(ty, params), _
              }, _) => {
                Some(fmt!(
                    "type %s%s = %s",
                    to_str(ident),
                    pprust::typarams_to_str(params, extract::interner()),
                    pprust::ty_to_str(ty, extract::interner())
                ))
              }
              _ => fail ~"expected type"
            }
        },
        .. doc
    }
}

#[test]
fn should_add_type_signatures() {
    let doc = test::mk_doc(~"type t<T> = int;");
    assert doc.cratemod().types()[0].sig == Some(~"type t<T> = int");
}

fn fold_struct(
    fold: fold::Fold<astsrv::Srv>,
    doc: doc::StructDoc
) -> doc::StructDoc {
    let srv = fold.ctxt;

    {
        sig: do astsrv::exec(srv) |ctxt| {
            match ctxt.ast_map.get(doc.id()) {
                ast_map::node_item(item, _) => {
                    let item = strip_struct_extra_stuff(item);
                    Some(pprust::item_to_str(item,
                                             extract::interner()))
                }
                _ => fail ~"not an item"
            }
        },
        .. doc
    }
}

/// Removes various things from the struct item definition that
/// shouldn't be displayed in the struct signature. Probably there
/// should be a simple pprust::struct_to_str function that does
/// what I actually want
fn strip_struct_extra_stuff(item: @ast::item) -> @ast::item {
    let node = match item.node {
        ast::item_class(def, tys) => {
            let def = @{
                dtor: None, // Remove the drop { } block
                .. *def
            };
            ast::item_class(def, tys)
        }
        _ => fail ~"not a struct"
    };

    @{
        attrs: ~[], // Remove the attributes
        node: node,
        .. *item
    }
}

#[test]
fn should_add_struct_defs() {
    let doc = test::mk_doc(~"struct S { field: () }");
    assert doc.cratemod().structs()[0].sig.get().contains("struct S {");
}

#[test]
fn should_not_serialize_struct_drop_blocks() {
    // All we care about are the fields
    let doc = test::mk_doc(~"struct S { field: (), drop { } }");
    assert !doc.cratemod().structs()[0].sig.get().contains("drop");
}

#[test]
fn should_not_serialize_struct_attrs() {
    // All we care about are the fields
    let doc = test::mk_doc(~"#[wut] struct S { field: () }");
    assert !doc.cratemod().structs()[0].sig.get().contains("wut");
}

#[cfg(test)]
mod test {
    #[legacy_exports];
    fn mk_doc(source: ~str) -> doc::Doc {
        do astsrv::from_str(source) |srv| {
            let doc = extract::from_srv(srv, ~"");
            run(srv, doc)
        }
    }
}
