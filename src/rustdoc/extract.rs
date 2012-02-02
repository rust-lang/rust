#[doc = "Converts the Rust AST to the rustdoc document model"];

import rustc::syntax::ast;

export from_srv, extract;

fn from_srv(
    srv: astsrv::srv,
    default_name: str
) -> doc::cratedoc {

    #[doc = "Use the AST service to create a document tree"];

    astsrv::exec(srv) {|ctxt|
        extract(ctxt.ast, default_name)
    }
}

fn extract(
    crate: @ast::crate,
    default_name: str
) -> doc::cratedoc {
    {
        topmod: top_moddoc_from_crate(crate, default_name),
    }
}

fn top_moddoc_from_crate(
    crate: @ast::crate,
    default_name: str
) -> doc::moddoc {
    moddoc_from_mod(crate.node.module, default_name, ast::crate_node_id)
}

fn moddoc_from_mod(
    module: ast::_mod,
    name: ast::ident,
    id: ast::node_id
) -> doc::moddoc {
    {
        id: id,
        name: name,
        path: [],
        brief: none,
        desc: none,
        items: ~vec::filter_map(module.items) {|item|
            alt item.node {
              ast::item_mod(m) {
                some(doc::modtag(
                    moddoc_from_mod(m, item.ident, item.id)
                ))
              }
              ast::item_fn(decl, _, _) {
                some(doc::fntag(
                    fndoc_from_fn(decl, item.ident, item.id)
                ))
              }
              ast::item_const(_, _) {
                some(doc::consttag(
                    constdoc_from_const(item.ident, item.id)
                ))
              }
              ast::item_enum(variants, _) {
                some(doc::enumtag(
                    enumdoc_from_enum(item.ident, item.id, variants)
                ))
              }
              ast::item_res(decl, _, _, _, _) {
                some(doc::restag(
                    resdoc_from_resource(decl, item.ident, item.id)
                ))
              }
              ast::item_iface(_, methods) {
                some(doc::ifacetag(
                    ifacedoc_from_iface(methods, item.ident, item.id)
                ))
              }
              ast::item_impl(_, _, _, methods) {
                some(doc::impltag(
                    impldoc_from_impl(methods, item.ident, item.id)
                ))
              }
              ast::item_ty(_, _) {
                some(doc::tytag(
                    tydoc_from_ty(item.ident, item.id)
                ))
              }
              _ {
                none
              }
            }
        }
    }
}

fn fndoc_from_fn(
    decl: ast::fn_decl,
    name: ast::ident,
    id: ast::node_id
) -> doc::fndoc {
    {
        id: id,
        name: name,
        brief: none,
        desc: none,
        args: argdocs_from_args(decl.inputs),
        return: {
            desc: none,
            ty: none
        },
        failure: none,
        sig: none
    }
}

#[test]
fn should_extract_fn_args() {
    let source = "fn a(b: int, c: int) { }";
    let ast = parse::from_str(source);
    let doc = extract(ast, "");
    let fn_ = doc.topmod.fns()[0];
    assert fn_.args[0].name == "b";
    assert fn_.args[1].name == "c";
}

fn argdocs_from_args(args: [ast::arg]) -> [doc::argdoc] {
    vec::map(args, argdoc_from_arg)
}

fn argdoc_from_arg(arg: ast::arg) -> doc::argdoc {
    {
        name: arg.ident,
        desc: none,
        ty: none
    }
}

fn constdoc_from_const(
    name: ast::ident,
    id: ast::node_id
) -> doc::constdoc {
    {
        id: id,
        name: name,
        brief: none,
        desc: none,
        ty: none
    }
}

#[test]
fn should_extract_const_name_and_id() {
    let doc = test::mk_doc("const a: int = 0;");
    assert doc.topmod.consts()[0].id != 0;
    assert doc.topmod.consts()[0].name == "a";
}

fn enumdoc_from_enum(
    name: ast::ident,
    id: ast::node_id,
    variants: [ast::variant]
) -> doc::enumdoc {
    {
        id: id,
        name: name,
        brief: none,
        desc: none,
        variants: variantdocs_from_variants(variants)
    }
}

fn variantdocs_from_variants(
    variants: [ast::variant]
) -> [doc::variantdoc] {
    vec::map(variants, variantdoc_from_variant)
}

fn variantdoc_from_variant(variant: ast::variant) -> doc::variantdoc {
    {
        name: variant.node.name,
        desc: none,
        sig: none
    }
}

#[test]
fn should_extract_enums() {
    let doc = test::mk_doc("enum e { v }");
    assert doc.topmod.enums()[0].id != 0;
    assert doc.topmod.enums()[0].name == "e";
}

#[test]
fn should_extract_enum_variants() {
    let doc = test::mk_doc("enum e { v }");
    assert doc.topmod.enums()[0].variants[0].name == "v";
}

fn resdoc_from_resource(
    decl: ast::fn_decl,
    name: str,
    id: ast::node_id
) -> doc::resdoc {
    {
        id: id,
        name: name,
        brief: none,
        desc: none,
        args: argdocs_from_args(decl.inputs),
        sig: none
    }
}

#[test]
fn should_extract_resources() {
    let doc = test::mk_doc("resource r(b: bool) { }");
    assert doc.topmod.resources()[0].id != 0;
    assert doc.topmod.resources()[0].name == "r";
}

#[test]
fn should_extract_resource_args() {
    let doc = test::mk_doc("resource r(b: bool) { }");
    assert doc.topmod.resources()[0].args[0].name == "b";
}

fn ifacedoc_from_iface(
    methods: [ast::ty_method],
    name: str,
    id: ast::node_id
) -> doc::ifacedoc {
    {
        id: id,
        name: name,
        brief: none,
        desc: none,
        methods: vec::map(methods) {|method|
            {
                name: method.ident,
                brief: none,
                desc: none,
                args: argdocs_from_args(method.decl.inputs),
                return: {
                    desc: none,
                    ty: none
                },
                failure: none,
                sig: none
            }
        }
    }
}

#[test]
fn should_extract_ifaces() {
    let doc = test::mk_doc("iface i { fn f(); }");
    assert doc.topmod.ifaces()[0].name == "i";
}

#[test]
fn should_extract_iface_methods() {
    let doc = test::mk_doc("iface i { fn f(); }");
    assert doc.topmod.ifaces()[0].methods[0].name == "f";
}

#[test]
fn should_extract_iface_method_args() {
    let doc = test::mk_doc("iface i { fn f(a: bool); }");
    assert doc.topmod.ifaces()[0].methods[0].args[0].name == "a";
}

fn impldoc_from_impl(
    methods: [@ast::method],
    name: str,
    id: ast::node_id
) -> doc::impldoc {
    {
        id: id,
        name: name,
        brief: none,
        desc: none,
        iface_ty: none,
        self_ty: none,
        methods: vec::map(methods) {|method|
            {
                name: method.ident,
                brief: none,
                desc: none,
                args: argdocs_from_args(method.decl.inputs),
                return: {
                    desc: none,
                    ty: none
                },
                failure: none,
                sig: none
            }
        }
    }
}

#[test]
fn should_extract_impls_with_names() {
    let doc = test::mk_doc("impl i for int { fn a() { } }");
    assert doc.topmod.impls()[0].name == "i";
}

#[test]
fn should_extract_impls_without_names() {
    let doc = test::mk_doc("impl of i for int { fn a() { } }");
    assert doc.topmod.impls()[0].name == "i";
}

#[test]
fn should_extract_impl_methods() {
    let doc = test::mk_doc("impl i for int { fn f() { } }");
    assert doc.topmod.impls()[0].methods[0].name == "f";
}

#[test]
fn should_extract_impl_method_args() {
    let doc = test::mk_doc("impl i for int { fn f(a: bool) { } }");
    assert doc.topmod.impls()[0].methods[0].args[0].name == "a";
}

fn tydoc_from_ty(
    name: str,
    id: ast::node_id
) -> doc::tydoc {
    {
        id: id,
        name: name,
        brief: none,
        desc: none,
        sig: none
    }
}

#[test]
fn should_extract_tys() {
    let doc = test::mk_doc("type a = int;");
    assert doc.topmod.types()[0].name == "a";
}

#[cfg(test)]
mod test {

    fn mk_doc(source: str) -> doc::cratedoc {
        let ast = parse::from_str(source);
        extract(ast, "")
    }

    #[test]
    fn extract_empty_crate() {
        let doc = mk_doc("");
        assert vec::is_empty(doc.topmod.mods());
        assert vec::is_empty(doc.topmod.fns());
    }

    #[test]
    fn extract_mods() {
        let doc = mk_doc("mod a { mod b { } mod c { } }");
        assert doc.topmod.mods()[0].name == "a";
        assert doc.topmod.mods()[0].mods()[0].name == "b";
        assert doc.topmod.mods()[0].mods()[1].name == "c";
    }

    #[test]
    fn extract_mods_deep() {
        let doc = mk_doc("mod a { mod b { mod c { } } }");
        assert doc.topmod.mods()[0].mods()[0].mods()[0].name == "c";
    }

    #[test]
    fn extract_should_set_mod_ast_id() {
        let doc = mk_doc("mod a { }");
        assert doc.topmod.mods()[0].id != 0;
    }

    #[test]
    fn extract_fns() {
        let doc = mk_doc(
            "fn a() { } \
             mod b { fn c() { } }");
        assert doc.topmod.fns()[0].name == "a";
        assert doc.topmod.mods()[0].fns()[0].name == "c";
    }

    #[test]
    fn extract_should_set_fn_ast_id() {
        let doc = mk_doc("fn a() { }");
        assert doc.topmod.fns()[0].id != 0;
    }

    #[test]
    fn extract_should_use_default_crate_name() {
        let source = "";
        let ast = parse::from_str(source);
        let doc = extract(ast, "burp");
        assert doc.topmod.name == "burp";
    }

    #[test]
    fn extract_from_seq_srv() {
        let source = "";
        let srv = astsrv::mk_srv_from_str(source);
        let doc = from_srv(srv, "name");
        assert doc.topmod.name == "name";
    }
}