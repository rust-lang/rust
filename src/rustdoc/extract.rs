import rustc::syntax::ast;
import rustc::syntax::visit;

export extract;

#[doc = "Converts the Rust AST to the rustdoc document model"]
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
    moddoc_from_mod(crate.node.module, default_name, crate.node.attrs)
}

fn moddoc_from_mod(
    module: ast::_mod,
    name: ast::ident,
    _attrs: [ast::attribute]
) -> doc::moddoc {
    ~{
        name: name,
        mods: doc::modlist(
            vec::filter_map(module.items) {|item|
                alt item.node {
                  ast::item_mod(m) {
                    some(moddoc_from_mod(m, item.ident, item.attrs))
                  }
                  _ {
                    none
                  }
                }
            }),
        fns: doc::fnlist(
            vec::filter_map(module.items) {|item|
                alt item.node {
                  ast::item_fn(decl, typarams, _) {
                    some(fndoc_from_fn(
                        decl, typarams, item.ident, item.id, item.attrs))
                  }
                  _ {
                    none
                  }
                }
            })
    }
}

fn fndoc_from_fn(
    _decl: ast::fn_decl,
    _typarams: [ast::ty_param],
    name: ast::ident,
    id: ast::node_id,
    attrs: [ast::attribute]
) -> doc::fndoc {
    attr_parser::parse_fn(name, id, attrs)
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
    fn extract_should_use_default_crate_name_if_no_link_name_exists() {
        let source = "";
        let ast = parse::from_str(source);
        let doc = extract(ast, "burp");
        assert doc.topmod.name == "burp";
    }
}