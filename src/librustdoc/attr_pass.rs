// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
The attribute parsing pass

Traverses the document tree, pulling relevant documention out of the
corresponding AST nodes. The information gathered here is the basis
of the natural-language documentation for a crate.
*/


use astsrv;
use attr_parser;
use doc::ItemUtils;
use doc;
use extract::to_str;
use fold::Fold;
use fold;
use pass::Pass;

use syntax::ast;
use syntax::ast_map;

pub fn mk_pass() -> Pass {
    Pass {
        name: ~"attr",
        f: run
    }
}

pub fn run(
    srv: astsrv::Srv,
    doc: doc::Doc
) -> doc::Doc {
    let fold = Fold {
        ctxt: srv.clone(),
        fold_crate: fold_crate,
        fold_item: fold_item,
        fold_enum: fold_enum,
        fold_trait: fold_trait,
        fold_impl: fold_impl,
        .. fold::default_any_fold(srv)
    };
    (fold.fold_doc)(&fold, doc)
}

fn fold_crate(
    fold: &fold::Fold<astsrv::Srv>,
    doc: doc::CrateDoc
) -> doc::CrateDoc {

    let srv = fold.ctxt.clone();
    let doc = fold::default_seq_fold_crate(fold, doc);

    let attrs = do astsrv::exec(srv) |ctxt| {
        let attrs = copy ctxt.ast.node.attrs;
        attr_parser::parse_crate(attrs)
    };

    doc::CrateDoc {
        topmod: doc::ModDoc {
            item: doc::ItemDoc {
                name: (copy attrs.name).get_or_default(doc.topmod.name()),
                .. copy doc.topmod.item
            },
            .. copy doc.topmod
        }
    }
}

fn fold_item(
    fold: &fold::Fold<astsrv::Srv>,
    doc: doc::ItemDoc
) -> doc::ItemDoc {

    let srv = fold.ctxt.clone();
    let doc = fold::default_seq_fold_item(fold, doc);

    let desc = if doc.id == ast::crate_node_id {
        // This is the top-level mod, use the crate attributes
        do astsrv::exec(srv) |ctxt| {
            attr_parser::parse_desc(copy ctxt.ast.node.attrs)
        }
    } else {
        parse_item_attrs(srv, doc.id, attr_parser::parse_desc)
    };

    doc::ItemDoc {
        desc: desc,
        .. doc
    }
}

fn parse_item_attrs<T:Send>(
    srv: astsrv::Srv,
    id: doc::AstId,
    parse_attrs: ~fn(a: ~[ast::attribute]) -> T) -> T {
    do astsrv::exec(srv) |ctxt| {
        let attrs = match ctxt.ast_map.get_copy(&id) {
            ast_map::node_item(item, _) => copy item.attrs,
            ast_map::node_foreign_item(item, _, _, _) => copy item.attrs,
            _ => fail!("parse_item_attrs: not an item")
        };
        parse_attrs(attrs)
    }
}

fn fold_enum(
    fold: &fold::Fold<astsrv::Srv>,
    doc: doc::EnumDoc
) -> doc::EnumDoc {

    let srv = fold.ctxt.clone();
    let doc_id = doc.id();
    let doc = fold::default_seq_fold_enum(fold, doc);

    doc::EnumDoc {
        variants: do doc.variants.iter().transform |variant| {
            let variant = copy *variant;
            let desc = {
                let variant = copy variant;
                do astsrv::exec(srv.clone()) |ctxt| {
                    match ctxt.ast_map.get_copy(&doc_id) {
                        ast_map::node_item(@ast::item {
                            node: ast::item_enum(ref enum_definition, _), _
                        }, _) => {
                            let ast_variant =
                                copy *enum_definition.variants.iter().find_(|v| {
                                    to_str(v.node.name) == variant.name
                                }).get();

                            attr_parser::parse_desc(
                                copy ast_variant.node.attrs)
                        }
                        _ => {
                            fail!("Enum variant %s has id that's not bound to an enum item",
                                  variant.name)
                        }
                    }
                }
            };

            doc::VariantDoc {
                desc: desc,
                .. variant
            }
        }.collect(),
        .. doc
    }
}

fn fold_trait(
    fold: &fold::Fold<astsrv::Srv>,
    doc: doc::TraitDoc
) -> doc::TraitDoc {
    let srv = fold.ctxt.clone();
    let doc = fold::default_seq_fold_trait(fold, doc);

    doc::TraitDoc {
        methods: merge_method_attrs(srv, doc.id(), copy doc.methods),
        .. doc
    }
}

fn merge_method_attrs(
    srv: astsrv::Srv,
    item_id: doc::AstId,
    docs: ~[doc::MethodDoc]
) -> ~[doc::MethodDoc] {

    // Create an assoc list from method name to attributes
    let attrs: ~[(~str, Option<~str>)] = do astsrv::exec(srv) |ctxt| {
        match ctxt.ast_map.get_copy(&item_id) {
            ast_map::node_item(@ast::item {
                node: ast::item_trait(_, _, ref methods), _
            }, _) => {
                methods.iter().transform(|method| {
                    match copy *method {
                        ast::required(ty_m) => {
                            (to_str(ty_m.ident),
                             attr_parser::parse_desc(copy ty_m.attrs))
                        }
                        ast::provided(m) => {
                            (to_str(m.ident), attr_parser::parse_desc(copy m.attrs))
                        }
                    }
                }).collect()
            }
            ast_map::node_item(@ast::item {
                node: ast::item_impl(_, _, _, ref methods), _
            }, _) => {
                methods.iter().transform(|method| {
                    (to_str(method.ident),
                     attr_parser::parse_desc(copy method.attrs))
                }).collect()
            }
            _ => fail!("unexpected item")
        }
    };

    do docs.iter().zip(attrs.iter()).transform |(doc, attrs)| {
        assert!(doc.name == attrs.first());
        let desc = attrs.second();

        doc::MethodDoc {
            desc: desc,
            .. copy *doc
        }
    }.collect()
}


fn fold_impl(
    fold: &fold::Fold<astsrv::Srv>,
    doc: doc::ImplDoc
) -> doc::ImplDoc {
    let srv = fold.ctxt.clone();
    let doc = fold::default_seq_fold_impl(fold, doc);

    doc::ImplDoc {
        methods: merge_method_attrs(srv, doc.id(), copy doc.methods),
        .. doc
    }
}

#[cfg(test)]
mod test {

    use astsrv;
    use attr_pass::run;
    use doc;
    use extract;

    fn mk_doc(source: ~str) -> doc::Doc {
        do astsrv::from_str(copy source) |srv| {
            let doc = extract::from_srv(srv.clone(), ~"");
            run(srv.clone(), doc)
        }
    }

    #[test]
    fn should_replace_top_module_name_with_crate_name() {
        let doc = mk_doc(~"#[link(name = \"bond\")];");
        assert!(doc.cratemod().name() == ~"bond");
    }

    #[test]
    fn should_should_extract_mod_attributes() {
        let doc = mk_doc(~"#[doc = \"test\"] mod a { }");
        // hidden __std_macros module at the start.
        assert!(doc.cratemod().mods()[1].desc() == Some(~"test"));
    }

    #[test]
    fn should_extract_top_mod_attributes() {
        let doc = mk_doc(~"#[doc = \"test\"];");
        assert!(doc.cratemod().desc() == Some(~"test"));
    }

    #[test]
    fn should_extract_foreign_fn_attributes() {
        let doc = mk_doc(~"extern { #[doc = \"test\"] fn a(); }");
        assert!(doc.cratemod().nmods()[0].fns[0].desc() == Some(~"test"));
    }

    #[test]
    fn should_extract_fn_attributes() {
        let doc = mk_doc(~"#[doc = \"test\"] fn a() -> int { }");
        assert!(doc.cratemod().fns()[0].desc() == Some(~"test"));
    }

    #[test]
    fn should_extract_enum_docs() {
        let doc = mk_doc(~"#[doc = \"b\"]\
                                 enum a { v }");
        assert!(doc.cratemod().enums()[0].desc() == Some(~"b"));
    }

    #[test]
    fn should_extract_variant_docs() {
        let doc = mk_doc(~"enum a { #[doc = \"c\"] v }");
        assert!(doc.cratemod().enums()[0].variants[0].desc == Some(~"c"));
    }

    #[test]
    fn should_extract_trait_docs() {
        let doc = mk_doc(~"#[doc = \"whatever\"] trait i { fn a(); }");
        assert!(doc.cratemod().traits()[0].desc() == Some(~"whatever"));
    }

    #[test]
    fn should_extract_trait_method_docs() {
        let doc = mk_doc(
            ~"trait i {\
              #[doc = \"desc\"]\
              fn f(a: bool) -> bool;\
              }");
        assert!(doc.cratemod().traits()[0].methods[0].desc == Some(~"desc"));
    }

    #[test]
    fn should_extract_impl_docs() {
        let doc = mk_doc(
            ~"#[doc = \"whatever\"] impl int { fn a() { } }");
        assert!(doc.cratemod().impls()[0].desc() == Some(~"whatever"));
    }

    #[test]
    fn should_extract_impl_method_docs() {
        let doc = mk_doc(
            ~"impl int {\
              #[doc = \"desc\"]\
              fn f(a: bool) -> bool { }\
              }");
        assert!(doc.cratemod().impls()[0].methods[0].desc == Some(~"desc"));
    }
}
