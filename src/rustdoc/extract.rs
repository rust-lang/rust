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
    ~{
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
    ~{
        id: id,
        name: name,
        path: [],
        brief: none,
        desc: none,
        items: [],
        mods: doc::modlist(
            vec::filter_map(module.items) {|item|
                alt item.node {
                  ast::item_mod(m) {
                    some(moddoc_from_mod(m, item.ident, item.id))
                  }
                  _ {
                    none
                  }
                }
            }),
        fns: doc::fnlist(
            vec::filter_map(module.items) {|item|
                alt item.node {
                  ast::item_fn(decl, _, _) {
                    some(fndoc_from_fn(
                        decl, item.ident, item.id))
                  }
                  _ {
                    none
                  }
                }
            }),
        consts: doc::constlist(
            vec::filter_map(module.items) {|item|
                alt item.node {
                  ast::item_const(_, _) {
                    some(constdoc_from_const(item.ident, item.id))
                  }
                  _ {
                    none
                  }
                }
            }),
        enums: doc::enumlist(
            vec::filter_map(module.items) {|item|
                alt item.node {
                  ast::item_enum(variants, _) {
                    some(enumdoc_from_enum(item.ident, item.id, variants))
                  }
                  _ {
                    none
                  }
                }
            }),
        resources: doc::reslist(
            vec::filter_map(module.items) {|item|
                alt item.node {
                  ast::item_res(decl, _, _, _, _) {
                    some(resdoc_from_resource(decl, item.ident, item.id))
                  }
                  _ {
                    none
                  }
                }
            })
    }
}

fn fndoc_from_fn(
    decl: ast::fn_decl,
    name: ast::ident,
    id: ast::node_id
) -> doc::fndoc {
    ~{
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
    let fn_ = doc.topmod.fns[0];
    assert fn_.args[0].name == "b";
    assert fn_.args[1].name == "c";
}

fn argdocs_from_args(args: [ast::arg]) -> [doc::argdoc] {
    vec::map(args, argdoc_from_arg)
}

fn argdoc_from_arg(arg: ast::arg) -> doc::argdoc {
    ~{
        name: arg.ident,
        desc: none,
        ty: none
    }
}

fn constdoc_from_const(
    name: ast::ident,
    id: ast::node_id
) -> doc::constdoc {
    ~{
        id: id,
        name: name,
        brief: none,
        desc: none,
        ty: none
    }
}

#[test]
fn should_extract_const_name_and_id() {
    let source = "const a: int = 0;";
    let ast = parse::from_str(source);
    let doc = extract(ast, "");
    assert doc.topmod.consts[0].id != 0;
    assert doc.topmod.consts[0].name == "a";
}

fn enumdoc_from_enum(
    name: ast::ident,
    id: ast::node_id,
    variants: [ast::variant]
) -> doc::enumdoc {
    ~{
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
    ~{
        name: variant.node.name,
        desc: none,
        sig: none
    }
}

#[test]
fn should_extract_enums() {
    let source = "enum e { v }";
    let ast = parse::from_str(source);
    let doc = extract(ast, "");
    assert doc.topmod.enums[0].id != 0;
    assert doc.topmod.enums[0].name == "e";
}

#[test]
fn should_extract_enum_variants() {
    let source = "enum e { v }";
    let ast = parse::from_str(source);
    let doc = extract(ast, "");
    assert doc.topmod.enums[0].variants[0].name == "v";
}

fn resdoc_from_resource(
    decl: ast::fn_decl,
    name: str,
    id: ast::node_id
) -> doc::resdoc {
    ~{
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
    let source = "resource r(b: bool) { }";
    let ast = parse::from_str(source);
    let doc = extract(ast, "");
    assert doc.topmod.resources[0].id != 0;
    assert doc.topmod.resources[0].name == "r";
}

#[test]
fn should_extract_resource_args() {
    let source = "resource r(b: bool) { }";
    let ast = parse::from_str(source);
    let doc = extract(ast, "");
    assert doc.topmod.resources[0].args[0].name == "b";
}

#[cfg(test)]
mod tests {

    #[test]
    fn extract_empty_crate() {
        let source = ""; // empty crate
        let ast = parse::from_str(source);
        let doc = extract(ast, "");
        // FIXME #1535: These are boxed to prevent a crash
        assert ~doc.topmod.mods == ~doc::modlist([]);
        assert ~doc.topmod.fns == ~doc::fnlist([]);
    }

    #[test]
    fn extract_mods() {
        let source = "mod a { mod b { } mod c { } }";
        let ast = parse::from_str(source);
        let doc = extract(ast, "");
        assert doc.topmod.mods[0].name == "a";
        assert doc.topmod.mods[0].mods[0].name == "b";
        assert doc.topmod.mods[0].mods[1].name == "c";
    }

    #[test]
    fn extract_mods_deep() {
        let source = "mod a { mod b { mod c { } } }";
        let ast = parse::from_str(source);
        let doc = extract(ast, "");
        assert doc.topmod.mods[0].mods[0].mods[0].name == "c";
    }

    #[test]
    fn extract_should_set_mod_ast_id() {
        let source = "mod a { }";
        let ast = parse::from_str(source);
        let doc = extract(ast, "");
        assert doc.topmod.mods[0].id != 0;
    }

    #[test]
    fn extract_fns() {
        let source =
            "fn a() { } \
             mod b { fn c() { } }";
        let ast = parse::from_str(source);
        let doc = extract(ast, "");
        assert doc.topmod.fns[0].name == "a";
        assert doc.topmod.mods[0].fns[0].name == "c";
    }

    #[test]
    fn extract_should_set_fn_ast_id() {
        let source = "fn a() { }";
        let ast = parse::from_str(source);
        let doc = extract(ast, "");
        assert doc.topmod.fns[0].id != 0;
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