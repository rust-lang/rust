//! Converts the Rust AST to the rustdoc document model

import syntax::ast;
import doc::item_utils;

export from_srv, extract, to_str, interner;

// Hack; rather than thread an interner through everywhere, rely on
// thread-local data
fn to_str(id: ast::ident) -> ~str {
    let intr = unsafe{ task::local_data_get(
        syntax::parse::token::interner_key) };

    return *(*intr.get()).get(id);
}

fn interner() -> syntax::parse::token::ident_interner {
    return *(unsafe{ task::local_data_get(
        syntax::parse::token::interner_key) }).get();
}

fn from_srv(
    srv: astsrv::srv,
    default_name: ~str
) -> doc::doc {

    //! Use the AST service to create a document tree

    do astsrv::exec(srv) |ctxt| {
        extract(ctxt.ast, default_name)
    }
}

fn extract(
    crate: @ast::crate,
    default_name: ~str
) -> doc::doc {
    doc::doc_({
        pages: ~[
            doc::cratepage({
                topmod: top_moddoc_from_crate(crate, default_name),
            })
        ]
    })
}

fn top_moddoc_from_crate(
    crate: @ast::crate,
    default_name: ~str
) -> doc::moddoc {
    moddoc_from_mod(mk_itemdoc(ast::crate_node_id, default_name),
                    crate.node.module)
}

fn mk_itemdoc(id: ast::node_id, name: ~str) -> doc::itemdoc {
    {
        id: id,
        name: name,
        path: ~[],
        brief: none,
        desc: none,
        sections: ~[],
        reexport: false
    }
}

fn moddoc_from_mod(
    itemdoc: doc::itemdoc,
    module_: ast::_mod
) -> doc::moddoc {
    doc::moddoc_({
        item: itemdoc,
        items: do vec::filter_map(module_.items) |item| {
            let itemdoc = mk_itemdoc(item.id, to_str(item.ident));
            match item.node {
              ast::item_mod(m) => {
                some(doc::modtag(
                    moddoc_from_mod(itemdoc, m)
                ))
              }
              ast::item_foreign_mod(nm) => {
                some(doc::nmodtag(
                    nmoddoc_from_mod(itemdoc, nm)
                ))
              }
              ast::item_fn(*) => {
                some(doc::fntag(
                    fndoc_from_fn(itemdoc)
                ))
              }
              ast::item_const(_, _) => {
                some(doc::consttag(
                    constdoc_from_const(itemdoc)
                ))
              }
              ast::item_enum(enum_definition, _) => {
                some(doc::enumtag(
                    enumdoc_from_enum(itemdoc, enum_definition.variants)
                ))
              }
              ast::item_trait(_, _, methods) => {
                some(doc::traittag(
                    traitdoc_from_trait(itemdoc, methods)
                ))
              }
              ast::item_impl(_, _, _, methods) => {
                some(doc::impltag(
                    impldoc_from_impl(itemdoc, methods)
                ))
              }
              ast::item_ty(_, _) => {
                some(doc::tytag(
                    tydoc_from_ty(itemdoc)
                ))
              }
              _ => none
            }
        },
        index: none
    })
}

fn nmoddoc_from_mod(
    itemdoc: doc::itemdoc,
    module_: ast::foreign_mod
) -> doc::nmoddoc {
    {
        item: itemdoc,
        fns: do vec::map(module_.items) |item| {
            let itemdoc = mk_itemdoc(item.id, to_str(item.ident));
            match item.node {
              ast::foreign_item_fn(*) => {
                fndoc_from_fn(itemdoc)
              }
            }
        },
        index: none
    }
}

fn fndoc_from_fn(itemdoc: doc::itemdoc) -> doc::fndoc {
    {
        item: itemdoc,
        sig: none
    }
}

fn constdoc_from_const(itemdoc: doc::itemdoc) -> doc::constdoc {
    {
        item: itemdoc,
        sig: none
    }
}

#[test]
fn should_extract_const_name_and_id() {
    let doc = test::mk_doc(~"const a: int = 0;");
    assert doc.cratemod().consts()[0].id() != 0;
    assert doc.cratemod().consts()[0].name() == ~"a";
}

fn enumdoc_from_enum(
    itemdoc: doc::itemdoc,
    variants: ~[ast::variant]
) -> doc::enumdoc {
    {
        item: itemdoc,
        variants: variantdocs_from_variants(variants)
    }
}

fn variantdocs_from_variants(
    variants: ~[ast::variant]
) -> ~[doc::variantdoc] {
    vec::map(variants, variantdoc_from_variant)
}

fn variantdoc_from_variant(variant: ast::variant) -> doc::variantdoc {

    {
        name: to_str(variant.node.name),
        desc: none,
        sig: none
    }
}

#[test]
fn should_extract_enums() {
    let doc = test::mk_doc(~"enum e { v }");
    assert doc.cratemod().enums()[0].id() != 0;
    assert doc.cratemod().enums()[0].name() == ~"e";
}

#[test]
fn should_extract_enum_variants() {
    let doc = test::mk_doc(~"enum e { v }");
    assert doc.cratemod().enums()[0].variants[0].name == ~"v";
}

fn traitdoc_from_trait(
    itemdoc: doc::itemdoc,
    methods: ~[ast::trait_method]
) -> doc::traitdoc {
    {
        item: itemdoc,
        methods: do vec::map(methods) |method| {
            match method {
              ast::required(ty_m) => {
                {
                    name: to_str(ty_m.ident),
                    brief: none,
                    desc: none,
                    sections: ~[],
                    sig: none,
                    implementation: doc::required,
                }
              }
              ast::provided(m) => {
                {
                    name: to_str(m.ident),
                    brief: none,
                    desc: none,
                    sections: ~[],
                    sig: none,
                    implementation: doc::provided,
                }
              }
            }
        }
    }
}

#[test]
fn should_extract_traits() {
    let doc = test::mk_doc(~"trait i { fn f(); }");
    assert doc.cratemod().traits()[0].name() == ~"i";
}

#[test]
fn should_extract_trait_methods() {
    let doc = test::mk_doc(~"trait i { fn f(); }");
    assert doc.cratemod().traits()[0].methods[0].name == ~"f";
}

fn impldoc_from_impl(
    itemdoc: doc::itemdoc,
    methods: ~[@ast::method]
) -> doc::impldoc {
    {
        item: itemdoc,
        trait_types: ~[],
        self_ty: none,
        methods: do vec::map(methods) |method| {
            {
                name: to_str(method.ident),
                brief: none,
                desc: none,
                sections: ~[],
                sig: none,
                implementation: doc::provided,
            }
        }
    }
}

#[test]
fn should_extract_impl_methods() {
    let doc = test::mk_doc(~"impl int { fn f() { } }");
    assert doc.cratemod().impls()[0].methods[0].name == ~"f";
}

fn tydoc_from_ty(
    itemdoc: doc::itemdoc
) -> doc::tydoc {
    {
        item: itemdoc,
        sig: none
    }
}

#[test]
fn should_extract_tys() {
    let doc = test::mk_doc(~"type a = int;");
    assert doc.cratemod().types()[0].name() == ~"a";
}

#[cfg(test)]
mod test {

    fn mk_doc(source: ~str) -> doc::doc {
        let ast = parse::from_str(source);
        extract(ast, ~"")
    }

    #[test]
    fn extract_empty_crate() {
        let doc = mk_doc(~"");
        assert vec::is_empty(doc.cratemod().mods());
        assert vec::is_empty(doc.cratemod().fns());
    }

    #[test]
    fn extract_mods() {
        let doc = mk_doc(~"mod a { mod b { } mod c { } }");
        assert doc.cratemod().mods()[0].name() == ~"a";
        assert doc.cratemod().mods()[0].mods()[0].name() == ~"b";
        assert doc.cratemod().mods()[0].mods()[1].name() == ~"c";
    }

    #[test]
    fn extract_foreign_mods() {
        let doc = mk_doc(~"extern mod a { }");
        assert doc.cratemod().nmods()[0].name() == ~"a";
    }

    #[test]
    fn extract_fns_from_foreign_mods() {
        let doc = mk_doc(~"extern mod a { fn a(); }");
        assert doc.cratemod().nmods()[0].fns[0].name() == ~"a";
    }

    #[test]
    fn extract_mods_deep() {
        let doc = mk_doc(~"mod a { mod b { mod c { } } }");
        assert doc.cratemod().mods()[0].mods()[0].mods()[0].name() == ~"c";
    }

    #[test]
    fn extract_should_set_mod_ast_id() {
        let doc = mk_doc(~"mod a { }");
        assert doc.cratemod().mods()[0].id() != 0;
    }

    #[test]
    fn extract_fns() {
        let doc = mk_doc(
            ~"fn a() { } \
             mod b { fn c() { } }");
        assert doc.cratemod().fns()[0].name() == ~"a";
        assert doc.cratemod().mods()[0].fns()[0].name() == ~"c";
    }

    #[test]
    fn extract_should_set_fn_ast_id() {
        let doc = mk_doc(~"fn a() { }");
        assert doc.cratemod().fns()[0].id() != 0;
    }

    #[test]
    fn extract_should_use_default_crate_name() {
        let source = ~"";
        let ast = parse::from_str(source);
        let doc = extract(ast, ~"burp");
        assert doc.cratemod().name() == ~"burp";
    }

    #[test]
    fn extract_from_seq_srv() {
        let source = ~"";
        do astsrv::from_str(source) |srv| {
            let doc = from_srv(srv, ~"name");
            assert doc.cratemod().name() == ~"name";
        }
    }
}
