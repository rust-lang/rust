// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Converts the Rust AST to the rustdoc document model


use astsrv;
use doc::ItemUtils;
use doc;

use syntax::ast;
use syntax::parse::token::{ident_interner, ident_to_str};
use syntax::parse::token;

// Hack; rather than thread an interner through everywhere, rely on
// thread-local data
// Hack-Becomes-Feature: using thread-local-state everywhere...
pub fn to_str(id: ast::Ident) -> ~str {
    /* bad */ ident_to_str(&id).to_owned()
}

// get rid of this pointless function:
pub fn interner() -> @ident_interner {
    return token::get_ident_interner();
}

pub fn from_srv(
    srv: astsrv::Srv,
    default_name: ~str
) -> doc::Doc {

    //! Use the AST service to create a document tree

    do astsrv::exec(srv) |ctxt| {
        extract(ctxt.ast, default_name.clone())
    }
}

pub fn extract(
    crate: @ast::Crate,
    default_name: ~str
) -> doc::Doc {
    doc::Doc {
        pages: ~[
            doc::CratePage(doc::CrateDoc {
                topmod: top_moddoc_from_crate(crate, default_name),
            })
        ]
    }
}

fn top_moddoc_from_crate(
    crate: @ast::Crate,
    default_name: ~str
) -> doc::ModDoc {
    moddoc_from_mod(mk_itemdoc(ast::CRATE_NODE_ID, default_name),
                    crate.module.clone())
}

fn mk_itemdoc(id: ast::NodeId, name: ~str) -> doc::ItemDoc {
    doc::ItemDoc {
        id: id,
        name: name,
        path: ~[],
        brief: None,
        desc: None,
        sections: ~[],
        reexport: false
    }
}

fn moddoc_from_mod(
    itemdoc: doc::ItemDoc,
    module_: ast::_mod
) -> doc::ModDoc {
    doc::ModDoc {
        item: itemdoc,
        items: do module_.items.iter().filter_map |item| {
            let ItemDoc = mk_itemdoc(item.id, to_str(item.ident));
            match item.node.clone() {
              ast::item_mod(m) => {
                Some(doc::ModTag(
                    moddoc_from_mod(ItemDoc, m)
                ))
              }
              ast::item_foreign_mod(nm) => {
                Some(doc::NmodTag(
                    nmoddoc_from_mod(ItemDoc, nm)
                ))
              }
              ast::item_fn(*) => {
                Some(doc::FnTag(
                    fndoc_from_fn(ItemDoc)
                ))
              }
              ast::item_static(*) => {
                Some(doc::StaticTag(
                    staticdoc_from_static(ItemDoc)
                ))
              }
              ast::item_enum(enum_definition, _) => {
                Some(doc::EnumTag(
                    enumdoc_from_enum(ItemDoc, enum_definition.variants.clone())
                ))
              }
              ast::item_trait(_, _, methods) => {
                Some(doc::TraitTag(
                    traitdoc_from_trait(ItemDoc, methods)
                ))
              }
              ast::item_impl(_, _, _, methods) => {
                Some(doc::ImplTag(
                    impldoc_from_impl(ItemDoc, methods)
                ))
              }
              ast::item_ty(_, _) => {
                Some(doc::TyTag(
                    tydoc_from_ty(ItemDoc)
                ))
              }
              ast::item_struct(def, _) => {
                Some(doc::StructTag(
                    structdoc_from_struct(ItemDoc, def)
                ))
              }
              _ => None
            }
        }.collect(),
        index: None
    }
}

fn nmoddoc_from_mod(
    itemdoc: doc::ItemDoc,
    module_: ast::foreign_mod
) -> doc::NmodDoc {
    let mut fns = ~[];
    for item in module_.items.iter() {
        let ItemDoc = mk_itemdoc(item.id, to_str(item.ident));
        match item.node {
          ast::foreign_item_fn(*) => {
            fns.push(fndoc_from_fn(ItemDoc));
          }
          ast::foreign_item_static(*) => {} // XXX: Not implemented.
        }
    }
    doc::NmodDoc {
        item: itemdoc,
        fns: fns,
        index: None
    }
}

fn fndoc_from_fn(itemdoc: doc::ItemDoc) -> doc::FnDoc {
    doc::SimpleItemDoc {
        item: itemdoc,
        sig: None
    }
}

fn staticdoc_from_static(itemdoc: doc::ItemDoc) -> doc::StaticDoc {
    doc::SimpleItemDoc {
        item: itemdoc,
        sig: None
    }
}

fn enumdoc_from_enum(
    itemdoc: doc::ItemDoc,
    variants: ~[ast::variant]
) -> doc::EnumDoc {
    doc::EnumDoc {
        item: itemdoc,
        variants: variantdocs_from_variants(variants)
    }
}

fn variantdocs_from_variants(
    variants: ~[ast::variant]
) -> ~[doc::VariantDoc] {
    variants.iter().map(variantdoc_from_variant).collect()
}

fn variantdoc_from_variant(variant: &ast::variant) -> doc::VariantDoc {
    doc::VariantDoc {
        name: to_str(variant.node.name),
        desc: None,
        sig: None
    }
}

fn traitdoc_from_trait(
    itemdoc: doc::ItemDoc,
    methods: ~[ast::trait_method]
) -> doc::TraitDoc {
    doc::TraitDoc {
        item: itemdoc,
        methods: do methods.iter().map |method| {
            match (*method).clone() {
              ast::required(ty_m) => {
                doc::MethodDoc {
                    name: to_str(ty_m.ident),
                    brief: None,
                    desc: None,
                    sections: ~[],
                    sig: None,
                    implementation: doc::Required,
                }
              }
              ast::provided(m) => {
                doc::MethodDoc {
                    name: to_str(m.ident),
                    brief: None,
                    desc: None,
                    sections: ~[],
                    sig: None,
                    implementation: doc::Provided,
                }
              }
            }
        }.collect()
    }
}

fn impldoc_from_impl(
    itemdoc: doc::ItemDoc,
    methods: ~[@ast::method]
) -> doc::ImplDoc {
    doc::ImplDoc {
        item: itemdoc,
        bounds_str: None,
        trait_types: ~[],
        self_ty: None,
        methods: do methods.iter().map |method| {
            doc::MethodDoc {
                name: to_str(method.ident),
                brief: None,
                desc: None,
                sections: ~[],
                sig: None,
                implementation: doc::Provided,
            }
        }.collect()
    }
}

fn tydoc_from_ty(
    itemdoc: doc::ItemDoc
) -> doc::TyDoc {
    doc::SimpleItemDoc {
        item: itemdoc,
        sig: None
    }
}

fn structdoc_from_struct(
    itemdoc: doc::ItemDoc,
    struct_def: @ast::struct_def
) -> doc::StructDoc {
    doc::StructDoc {
        item: itemdoc,
        fields: do struct_def.fields.map |field| {
            match field.node.kind {
                ast::named_field(ident, _) => to_str(ident),
                ast::unnamed_field => ~"(unnamed)",
            }
        },
        sig: None
    }
}

#[cfg(test)]
mod test {
    use astsrv;
    use doc;
    use extract::{extract, from_srv};
    use parse;

    fn mk_doc(source: @str) -> doc::Doc {
        let ast = parse::from_str(source);
        debug!("ast=%?", ast);
        extract(ast, ~"")
    }

    #[test]
    fn extract_empty_crate() {
        let doc = mk_doc(@"");
        assert!(doc.cratemod().mods().is_empty());
        assert!(doc.cratemod().fns().is_empty());
    }

    #[test]
    fn extract_mods() {
        let doc = mk_doc(@"mod a { mod b { } mod c { } }");
        assert!(doc.cratemod().mods()[0].name_() == ~"a");
        assert!(doc.cratemod().mods()[0].mods()[0].name_() == ~"b");
        assert!(doc.cratemod().mods()[0].mods()[1].name_() == ~"c");
    }

    #[test]
    fn extract_fns_from_foreign_mods() {
        let doc = mk_doc(@"extern { fn a(); }");
        assert!(doc.cratemod().nmods()[0].fns[0].name_() == ~"a");
    }

    #[test]
    fn extract_mods_deep() {
        let doc = mk_doc(@"mod a { mod b { mod c { } } }");
        assert!(doc.cratemod().mods()[0].mods()[0].mods()[0].name_() ==
            ~"c");
    }

    #[test]
    fn extract_should_set_mod_ast_id() {
        let doc = mk_doc(@"mod a { }");
        assert!(doc.cratemod().mods()[0].id() != 0);
    }

    #[test]
    fn extract_fns() {
        let doc = mk_doc(
            @"fn a() { } \
              mod b { fn c() {
             } }");
        assert!(doc.cratemod().fns()[0].name_() == ~"a");
        assert!(doc.cratemod().mods()[0].fns()[0].name_() == ~"c");
    }

    #[test]
    fn extract_should_set_fn_ast_id() {
        let doc = mk_doc(@"fn a() { }");
        assert!(doc.cratemod().fns()[0].id() != 0);
    }

    #[test]
    fn extract_should_use_default_crate_name() {
        let source = @"";
        let ast = parse::from_str(source);
        let doc = extract(ast, ~"burp");
        assert!(doc.cratemod().name_() == ~"burp");
    }

    #[test]
    fn extract_from_seq_srv() {
        let source = ~"";
        do astsrv::from_str(source) |srv| {
            let doc = from_srv(srv, ~"name");
            assert!(doc.cratemod().name_() == ~"name");
        }
    }

    #[test]
    fn should_extract_static_name_and_id() {
        let doc = mk_doc(@"static a: int = 0;");
        assert!(doc.cratemod().statics()[0].id() != 0);
        assert!(doc.cratemod().statics()[0].name_() == ~"a");
    }

    #[test]
    fn should_extract_enums() {
        let doc = mk_doc(@"enum e { v }");
        assert!(doc.cratemod().enums()[0].id() != 0);
        assert!(doc.cratemod().enums()[0].name_() == ~"e");
    }

    #[test]
    fn should_extract_enum_variants() {
        let doc = mk_doc(@"enum e { v }");
        assert!(doc.cratemod().enums()[0].variants[0].name == ~"v");
    }

    #[test]
    fn should_extract_traits() {
        let doc = mk_doc(@"trait i { fn f(); }");
        assert!(doc.cratemod().traits()[0].name_() == ~"i");
    }

    #[test]
    fn should_extract_trait_methods() {
        let doc = mk_doc(@"trait i { fn f(); }");
        assert!(doc.cratemod().traits()[0].methods[0].name == ~"f");
    }

    #[test]
    fn should_extract_impl_methods() {
        let doc = mk_doc(@"impl int { fn f() { } }");
        assert!(doc.cratemod().impls()[0].methods[0].name == ~"f");
    }

    #[test]
    fn should_extract_tys() {
        let doc = mk_doc(@"type a = int;");
        assert!(doc.cratemod().types()[0].name_() == ~"a");
    }

    #[test]
    fn should_extract_structs() {
        let doc = mk_doc(@"struct Foo { field: () }");
        assert!(doc.cratemod().structs()[0].name_() == ~"Foo");
    }

    #[test]
    fn should_extract_struct_fields() {
        let doc = mk_doc(@"struct Foo { field: () }");
        assert!(doc.cratemod().structs()[0].fields[0] == ~"field");
    }
}
