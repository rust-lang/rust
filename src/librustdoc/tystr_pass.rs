// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Pulls type information out of the AST and attaches it to the document

use astsrv;
use doc::ItemUtils;
use doc;
use extract::to_str;
use extract;
use fold::Fold;
use fold;
use pass::Pass;

use syntax::ast;
use syntax::print::pprust;
use syntax::ast_map;

pub fn mk_pass() -> Pass {
    Pass {
        name: ~"tystr",
        f: run
    }
}

pub fn run(
    srv: astsrv::Srv,
    doc: doc::Doc
) -> doc::Doc {
    let fold = Fold {
        ctxt: srv.clone(),
        fold_fn: fold_fn,
        fold_const: fold_const,
        fold_enum: fold_enum,
        fold_trait: fold_trait,
        fold_impl: fold_impl,
        fold_type: fold_type,
        fold_struct: fold_struct,
        .. fold::default_any_fold(srv)
    };
    (fold.fold_doc)(&fold, doc)
}

fn fold_fn(
    fold: &fold::Fold<astsrv::Srv>,
    doc: doc::FnDoc
) -> doc::FnDoc {

    let srv = fold.ctxt.clone();

    doc::SimpleItemDoc {
        sig: get_fn_sig(srv, doc.id()),
        .. doc
    }
}

fn get_fn_sig(srv: astsrv::Srv, fn_id: doc::AstId) -> Option<~str> {
    do astsrv::exec(srv) |ctxt| {
        match *ctxt.ast_map.get(&fn_id) {
            ast_map::node_item(@ast::item {
                ident: ident,
                node: ast::item_fn(ref decl, purity, _, ref tys, _), _
            }, _) |
            ast_map::node_foreign_item(@ast::foreign_item {
                ident: ident,
                node: ast::foreign_item_fn(ref decl, purity, ref tys), _
            }, _, _, _) => {
                Some(pprust::fun_to_str(decl, purity, ident, None, tys,
                                        extract::interner()))
            }
            _ => fail!("get_fn_sig: fn_id not bound to a fn item")
        }
    }
}

fn fold_const(
    fold: &fold::Fold<astsrv::Srv>,
    doc: doc::ConstDoc
) -> doc::ConstDoc {
    let srv = fold.ctxt.clone();

    doc::SimpleItemDoc {
        sig: Some({
            let doc = copy doc;
            do astsrv::exec(srv) |ctxt| {
                match *ctxt.ast_map.get(&doc.id()) {
                    ast_map::node_item(@ast::item {
                        node: ast::item_const(ty, _), _
                    }, _) => {
                        pprust::ty_to_str(ty, extract::interner())
                    }
                    _ => fail!("fold_const: id not bound to a const item")
                }
            }}),
        .. doc
    }
}

fn fold_enum(
    fold: &fold::Fold<astsrv::Srv>,
    doc: doc::EnumDoc
) -> doc::EnumDoc {
    let doc_id = doc.id();
    let srv = fold.ctxt.clone();

    doc::EnumDoc {
        variants: do vec::map(doc.variants) |variant| {
            let sig = {
                let variant = copy *variant;
                do astsrv::exec(srv.clone()) |ctxt| {
                    match *ctxt.ast_map.get(&doc_id) {
                        ast_map::node_item(@ast::item {
                            node: ast::item_enum(ref enum_definition, _), _
                        }, _) => {
                            let ast_variant =
                                do vec::find(enum_definition.variants) |v| {
                                to_str(v.node.name) == variant.name
                            }.get();

                            pprust::variant_to_str(
                                ast_variant, extract::interner())
                        }
                        _ => fail!("enum variant not bound to an enum item")
                    }
                }
            };

            doc::VariantDoc {
                sig: Some(sig),
                .. copy *variant
            }
        },
        .. doc
    }
}

fn fold_trait(
    fold: &fold::Fold<astsrv::Srv>,
    doc: doc::TraitDoc
) -> doc::TraitDoc {
    doc::TraitDoc {
        methods: merge_methods(fold.ctxt.clone(), doc.id(), copy doc.methods),
        .. doc
    }
}

fn merge_methods(
    srv: astsrv::Srv,
    item_id: doc::AstId,
    docs: ~[doc::MethodDoc]
) -> ~[doc::MethodDoc] {
    do vec::map(docs) |doc| {
        doc::MethodDoc {
            sig: get_method_sig(srv.clone(), item_id, copy doc.name),
            .. copy *doc
        }
    }
}

fn get_method_sig(
    srv: astsrv::Srv,
    item_id: doc::AstId,
    method_name: ~str
) -> Option<~str> {
    do astsrv::exec(srv) |ctxt| {
        match *ctxt.ast_map.get(&item_id) {
            ast_map::node_item(@ast::item {
                node: ast::item_trait(_, _, ref methods), _
            }, _) => {
                match vec::find(*methods, |method| {
                    match copy *method {
                        ast::required(ty_m) => to_str(ty_m.ident) == method_name,
                        ast::provided(m) => to_str(m.ident) == method_name,
                    }
                }) {
                    Some(method) => {
                        match method {
                            ast::required(ty_m) => {
                                Some(pprust::fun_to_str(
                                    &ty_m.decl,
                                    ty_m.purity,
                                    ty_m.ident,
                                    Some(ty_m.self_ty.node),
                                    &ty_m.generics,
                                    extract::interner()
                                ))
                            }
                            ast::provided(m) => {
                                Some(pprust::fun_to_str(
                                    &m.decl,
                                    m.purity,
                                    m.ident,
                                    Some(m.self_ty.node),
                                    &m.generics,
                                    extract::interner()
                                ))
                            }
                        }
                    }
                    _ => fail!("method not found")
                }
            }
            ast_map::node_item(@ast::item {
                node: ast::item_impl(_, _, _, ref methods), _
            }, _) => {
                match vec::find(*methods, |method| {
                    to_str(method.ident) == method_name
                }) {
                    Some(method) => {
                        Some(pprust::fun_to_str(
                            &method.decl,
                            method.purity,
                            method.ident,
                            Some(method.self_ty.node),
                            &method.generics,
                            extract::interner()
                        ))
                    }
                    None => fail!("method not found")
                }
            }
            _ => fail!("get_method_sig: item ID not bound to trait or impl")
        }
    }
}

fn fold_impl(
    fold: &fold::Fold<astsrv::Srv>,
    doc: doc::ImplDoc
) -> doc::ImplDoc {

    let srv = fold.ctxt.clone();

    let (bounds, trait_types, self_ty) = {
        let doc = copy doc;
        do astsrv::exec(srv) |ctxt| {
            match *ctxt.ast_map.get(&doc.id()) {
                ast_map::node_item(@ast::item {
                    node: ast::item_impl(ref generics, opt_trait_type, self_ty, _), _
                }, _) => {
                    let bounds = pprust::generics_to_str(generics, extract::interner());
                    let bounds = if bounds.is_empty() { None } else { Some(bounds) };
                    let trait_types = opt_trait_type.map_default(~[], |p| {
                        ~[pprust::path_to_str(p.path, extract::interner())]
                    });
                    (bounds,
                     trait_types,
                     Some(pprust::ty_to_str(
                         self_ty, extract::interner())))
                }
                _ => fail!("expected impl")
            }
        }
    };

    doc::ImplDoc {
        bounds_str: bounds,
        trait_types: trait_types,
        self_ty: self_ty,
        methods: merge_methods(fold.ctxt.clone(), doc.id(), copy doc.methods),
        .. doc
    }
}

fn fold_type(
    fold: &fold::Fold<astsrv::Srv>,
    doc: doc::TyDoc
) -> doc::TyDoc {

    let srv = fold.ctxt.clone();

    doc::SimpleItemDoc {
        sig: {
            let doc = copy doc;
            do astsrv::exec(srv) |ctxt| {
                match *ctxt.ast_map.get(&doc.id()) {
                    ast_map::node_item(@ast::item {
                        ident: ident,
                        node: ast::item_ty(ty, ref params), _
                    }, _) => {
                        Some(fmt!(
                            "type %s%s = %s",
                            to_str(ident),
                            pprust::generics_to_str(params,
                                                    extract::interner()),
                            pprust::ty_to_str(ty,
                                              extract::interner())
                        ))
                    }
                    _ => fail!("expected type")
                }
            }
        },
        .. doc
    }
}

fn fold_struct(
    fold: &fold::Fold<astsrv::Srv>,
    doc: doc::StructDoc
) -> doc::StructDoc {
    let srv = fold.ctxt.clone();

    doc::StructDoc {
        sig: {
            let doc = copy doc;
            do astsrv::exec(srv) |ctxt| {
                match *ctxt.ast_map.get(&doc.id()) {
                    ast_map::node_item(item, _) => {
                        let item = strip_struct_extra_stuff(item);
                        Some(pprust::item_to_str(item,
                                                 extract::interner()))
                    }
                    _ => fail!("not an item")
                }
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
    let node = match copy item.node {
        ast::item_struct(def, tys) => ast::item_struct(def, tys),
        _ => fail!("not a struct")
    };

    @ast::item {
        attrs: ~[], // Remove the attributes
        node: node,
        .. copy *item
    }
}

#[cfg(test)]
mod test {
    use astsrv;
    use doc;
    use extract;
    use tystr_pass::run;

    fn mk_doc(source: ~str) -> doc::Doc {
        do astsrv::from_str(copy source) |srv| {
            let doc = extract::from_srv(srv.clone(), ~"");
            run(srv.clone(), doc)
        }
    }

    #[test]
    fn should_add_fn_sig() {
        let doc = mk_doc(~"fn a<T>() -> int { }");
        assert!(doc.cratemod().fns()[0].sig == Some(~"fn a<T>() -> int"));
    }

    #[test]
    fn should_add_foreign_fn_sig() {
        let doc = mk_doc(~"extern { fn a<T>() -> int; }");
        assert!(doc.cratemod().nmods()[0].fns[0].sig ==
                Some(~"fn a<T>() -> int"));
    }

    #[test]
    fn should_add_const_types() {
        let doc = mk_doc(~"static a: bool = true;");
        assert!(doc.cratemod().consts()[0].sig == Some(~"bool"));
    }

    #[test]
    fn should_add_variant_sigs() {
        let doc = mk_doc(~"enum a { b(int) }");
        assert!(doc.cratemod().enums()[0].variants[0].sig ==
                Some(~"b(int)"));
    }

    #[test]
    fn should_add_trait_method_sigs() {
        let doc = mk_doc(~"trait i { fn a<T>(&mut self) -> int; }");
        assert!(doc.cratemod().traits()[0].methods[0].sig
                == Some(~"fn a<T>(&mut self) -> int"));
    }

    #[test]
    fn should_add_impl_bounds() {
        let doc = mk_doc(~"impl<T, U: Copy, V: Copy + Clone> Option<T, U, V> { }");
        assert!(doc.cratemod().impls()[0].bounds_str == Some(~"<T, U: Copy, V: Copy + Clone>"));
    }

    #[test]
    fn should_add_impl_trait_types() {
        let doc = mk_doc(~"impl j for int { fn a<T>() { } }");
        assert!(doc.cratemod().impls()[0].trait_types[0] == ~"j");
    }

    #[test]
    fn should_not_add_impl_trait_types_if_none() {
        let doc = mk_doc(~"impl int { fn a() { } }");
        assert!(vec::len(doc.cratemod().impls()[0].trait_types) == 0);
    }

    #[test]
    fn should_add_impl_self_ty() {
        let doc = mk_doc(~"impl int { fn a() { } }");
        assert!(doc.cratemod().impls()[0].self_ty == Some(~"int"));
    }

    #[test]
    fn should_add_impl_method_sigs() {
        let doc = mk_doc(~"impl int { fn a<T>(&self) -> int { fail!() } }");
        assert!(doc.cratemod().impls()[0].methods[0].sig
                == Some(~"fn a<T>(&self) -> int"));
    }

    #[test]
    fn should_add_type_signatures() {
        let doc = mk_doc(~"type t<T> = int;");
        assert!(doc.cratemod().types()[0].sig == Some(~"type t<T> = int"));
    }

    #[test]
    fn should_add_struct_defs() {
        let doc = mk_doc(~"struct S { field: () }");
        assert!((&doc.cratemod().structs()[0].sig).get().contains(
            "struct S {"));
    }

    #[test]
    fn should_not_serialize_struct_attrs() {
        // All we care about are the fields
        let doc = mk_doc(~"#[wut] struct S { field: () }");
        assert!(!(&doc.cratemod().structs()[0].sig).get().contains("wut"));
    }
}
