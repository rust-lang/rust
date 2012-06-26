#[doc = "Converts the Rust AST to the rustdoc document model"];

import syntax::ast;

export from_srv, extract;

fn from_srv(
    srv: astsrv::srv,
    default_name: str
) -> doc::doc {

    #[doc = "Use the AST service to create a document tree"];

    astsrv::exec(srv) {|ctxt|
        extract(ctxt.ast, default_name)
    }
}

fn extract(
    crate: @ast::crate,
    default_name: str
) -> doc::doc {
    {
        pages: [
            doc::cratepage({
                topmod: top_moddoc_from_crate(crate, default_name),
            })
        ]/~
    }
}

fn top_moddoc_from_crate(
    crate: @ast::crate,
    default_name: str
) -> doc::moddoc {
    moddoc_from_mod(mk_itemdoc(ast::crate_node_id, @default_name),
                    crate.node.module)
}

fn mk_itemdoc(id: ast::node_id, name: ast::ident) -> doc::itemdoc {
    {
        id: id,
        name: *name,
        path: []/~,
        brief: none,
        desc: none,
        sections: []/~,
        reexport: false
    }
}

fn moddoc_from_mod(
    itemdoc: doc::itemdoc,
    module: ast::_mod
) -> doc::moddoc {
    {
        item: itemdoc,
        items: vec::filter_map(module.items) {|item|
            let itemdoc = mk_itemdoc(item.id, item.ident);
            alt item.node {
              ast::item_mod(m) {
                some(doc::modtag(
                    moddoc_from_mod(itemdoc, m)
                ))
              }
              ast::item_foreign_mod(nm) {
                some(doc::nmodtag(
                    nmoddoc_from_mod(itemdoc, nm)
                ))
              }
              ast::item_fn(_, _, _) {
                some(doc::fntag(
                    fndoc_from_fn(itemdoc)
                ))
              }
              ast::item_const(_, _) {
                some(doc::consttag(
                    constdoc_from_const(itemdoc)
                ))
              }
              ast::item_enum(variants, _, _) {
                some(doc::enumtag(
                    enumdoc_from_enum(itemdoc, variants)
                ))
              }
              ast::item_iface(_, _, methods) {
                some(doc::ifacetag(
                    ifacedoc_from_iface(itemdoc, methods)
                ))
              }
              ast::item_impl(_, _, _, _, methods) {
                some(doc::impltag(
                    impldoc_from_impl(itemdoc, methods)
                ))
              }
              ast::item_ty(_, _, _) {
                some(doc::tytag(
                    tydoc_from_ty(itemdoc)
                ))
              }
              _ {
                none
              }
            }
        },
        index: none
    }
}

fn nmoddoc_from_mod(
    itemdoc: doc::itemdoc,
    module: ast::foreign_mod
) -> doc::nmoddoc {
    {
        item: itemdoc,
        fns: par::seqmap(module.items) {|item|
            let itemdoc = mk_itemdoc(item.id, item.ident);
            alt item.node {
              ast::foreign_item_fn(_, _) {
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
    let doc = test::mk_doc("const a: int = 0;");
    assert doc.cratemod().consts()[0].id() != 0;
    assert doc.cratemod().consts()[0].name() == "a";
}

fn enumdoc_from_enum(
    itemdoc: doc::itemdoc,
    variants: [ast::variant]/~
) -> doc::enumdoc {
    {
        item: itemdoc,
        variants: variantdocs_from_variants(variants)
    }
}

fn variantdocs_from_variants(
    variants: [ast::variant]/~
) -> [doc::variantdoc]/~ {
    par::seqmap(variants, variantdoc_from_variant)
}

fn variantdoc_from_variant(variant: ast::variant) -> doc::variantdoc {
    {
        name: *variant.node.name,
        desc: none,
        sig: none
    }
}

#[test]
fn should_extract_enums() {
    let doc = test::mk_doc("enum e { v }");
    assert doc.cratemod().enums()[0].id() != 0;
    assert doc.cratemod().enums()[0].name() == "e";
}

#[test]
fn should_extract_enum_variants() {
    let doc = test::mk_doc("enum e { v }");
    assert doc.cratemod().enums()[0].variants[0].name == "v";
}

fn ifacedoc_from_iface(
    itemdoc: doc::itemdoc,
    methods: [ast::ty_method]/~
) -> doc::ifacedoc {
    {
        item: itemdoc,
        methods: par::seqmap(methods) {|method|
            {
                name: *method.ident,
                brief: none,
                desc: none,
                sections: []/~,
                sig: none
            }
        }
    }
}

#[test]
fn should_extract_ifaces() {
    let doc = test::mk_doc("iface i { fn f(); }");
    assert doc.cratemod().ifaces()[0].name() == "i";
}

#[test]
fn should_extract_iface_methods() {
    let doc = test::mk_doc("iface i { fn f(); }");
    assert doc.cratemod().ifaces()[0].methods[0].name == "f";
}

fn impldoc_from_impl(
    itemdoc: doc::itemdoc,
    methods: [@ast::method]/~
) -> doc::impldoc {
    {
        item: itemdoc,
        iface_ty: none,
        self_ty: none,
        methods: par::seqmap(methods) {|method|
            {
                name: *method.ident,
                brief: none,
                desc: none,
                sections: []/~,
                sig: none
            }
        }
    }
}

#[test]
fn should_extract_impls_with_names() {
    let doc = test::mk_doc("impl i for int { fn a() { } }");
    assert doc.cratemod().impls()[0].name() == "i";
}

#[test]
fn should_extract_impls_without_names() {
    let doc = test::mk_doc("impl of i for int { fn a() { } }");
    assert doc.cratemod().impls()[0].name() == "i";
}

#[test]
fn should_extract_impl_methods() {
    let doc = test::mk_doc("impl i for int { fn f() { } }");
    assert doc.cratemod().impls()[0].methods[0].name == "f";
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
    let doc = test::mk_doc("type a = int;");
    assert doc.cratemod().types()[0].name() == "a";
}

#[cfg(test)]
mod test {

    fn mk_doc(source: str) -> doc::doc {
        let ast = parse::from_str(source);
        extract(ast, "")
    }

    #[test]
    fn extract_empty_crate() {
        let doc = mk_doc("");
        assert vec::is_empty(doc.cratemod().mods());
        assert vec::is_empty(doc.cratemod().fns());
    }

    #[test]
    fn extract_mods() {
        let doc = mk_doc("mod a { mod b { } mod c { } }");
        assert doc.cratemod().mods()[0].name() == "a";
        assert doc.cratemod().mods()[0].mods()[0].name() == "b";
        assert doc.cratemod().mods()[0].mods()[1].name() == "c";
    }

    #[test]
    fn extract_foreign_mods() {
        let doc = mk_doc("native mod a { }");
        assert doc.cratemod().nmods()[0].name() == "a";
    }

    #[test]
    fn extract_fns_from_foreign_mods() {
        let doc = mk_doc("native mod a { fn a(); }");
        assert doc.cratemod().nmods()[0].fns[0].name() == "a";
    }

    #[test]
    fn extract_mods_deep() {
        let doc = mk_doc("mod a { mod b { mod c { } } }");
        assert doc.cratemod().mods()[0].mods()[0].mods()[0].name() == "c";
    }

    #[test]
    fn extract_should_set_mod_ast_id() {
        let doc = mk_doc("mod a { }");
        assert doc.cratemod().mods()[0].id() != 0;
    }

    #[test]
    fn extract_fns() {
        let doc = mk_doc(
            "fn a() { } \
             mod b { fn c() { } }");
        assert doc.cratemod().fns()[0].name() == "a";
        assert doc.cratemod().mods()[0].fns()[0].name() == "c";
    }

    #[test]
    fn extract_should_set_fn_ast_id() {
        let doc = mk_doc("fn a() { }");
        assert doc.cratemod().fns()[0].id() != 0;
    }

    #[test]
    fn extract_should_use_default_crate_name() {
        let source = "";
        let ast = parse::from_str(source);
        let doc = extract(ast, "burp");
        assert doc.cratemod().name() == "burp";
    }

    #[test]
    fn extract_from_seq_srv() {
        let source = "";
        astsrv::from_str(source) {|srv|
            let doc = from_srv(srv, "name");
            assert doc.cratemod().name() == "name";
        }
    }
}
