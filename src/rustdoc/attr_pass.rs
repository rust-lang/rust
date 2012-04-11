#[doc(
    brief = "The attribute parsing pass",
    desc =
    "Traverses the document tree, pulling relevant documention out of the \
     corresponding AST nodes. The information gathered here is the basis \
     of the natural-language documentation for a crate."
)];

import rustc::syntax::ast;
import rustc::middle::ast_map;
import std::map::hashmap;

export mk_pass;

fn mk_pass() -> pass {
    {
        name: "attr",
        f: run
    }
}

fn run(
    srv: astsrv::srv,
    doc: doc::doc
) -> doc::doc {
    let fold = fold::fold({
        fold_crate: fold_crate,
        fold_item: fold_item,
        fold_enum: fold_enum,
        fold_iface: fold_iface,
        fold_impl: fold_impl
        with *fold::default_any_fold(srv)
    });
    fold.fold_doc(fold, doc)
}

fn fold_crate(
    fold: fold::fold<astsrv::srv>,
    doc: doc::cratedoc
) -> doc::cratedoc {

    let srv = fold.ctxt;
    let doc = fold::default_seq_fold_crate(fold, doc);

    let attrs = astsrv::exec(srv) {|ctxt|
        let attrs = ctxt.ast.node.attrs;
        attr_parser::parse_crate(attrs)
    };

    {
        topmod: {
            item: {
                name: option::get_default(attrs.name, doc.topmod.name())
                with doc.topmod.item
            }
            with doc.topmod
        }
    }
}

#[test]
fn should_replace_top_module_name_with_crate_name() {
    let doc = test::mk_doc("#[link(name = \"bond\")];");
    assert doc.cratemod().name() == "bond";
}

fn fold_item(
    fold: fold::fold<astsrv::srv>,
    doc: doc::itemdoc
) -> doc::itemdoc {

    let srv = fold.ctxt;
    let doc = fold::default_seq_fold_item(fold, doc);

    let desc = if doc.id == ast::crate_node_id {
        // This is the top-level mod, use the crate attributes
        astsrv::exec(srv) {|ctxt|
            attr_parser::parse_desc(ctxt.ast.node.attrs)
        }
    } else {
        parse_item_attrs(srv, doc.id, attr_parser::parse_desc)
    };

    {
        desc: desc
        with doc
    }
}

fn parse_item_attrs<T:send>(
    srv: astsrv::srv,
    id: doc::ast_id,
    parse_attrs: fn~([ast::attribute]) -> T) -> T {
    astsrv::exec(srv) {|ctxt|
        let attrs = alt ctxt.ast_map.get(id) {
          ast_map::node_item(item, _) { item.attrs }
          ast_map::node_native_item(item, _, _) { item.attrs }
          _ {
            fail "parse_item_attrs: not an item";
          }
        };
        parse_attrs(attrs)
    }
}

#[test]
fn should_should_extract_mod_attributes() {
    let doc = test::mk_doc("#[doc = \"test\"] mod a { }");
    assert doc.cratemod().mods()[0].desc() == some("test");
}

#[test]
fn should_extract_top_mod_attributes() {
    let doc = test::mk_doc("#[doc = \"test\"];");
    assert doc.cratemod().desc() == some("test");
}

#[test]
fn should_extract_native_mod_attributes() {
    let doc = test::mk_doc("#[doc = \"test\"] native mod a { }");
    assert doc.cratemod().nmods()[0].desc() == some("test");
}

#[test]
fn should_extract_native_fn_attributes() {
    let doc = test::mk_doc("native mod a { #[doc = \"test\"] fn a(); }");
    assert doc.cratemod().nmods()[0].fns[0].desc() == some("test");
}

#[test]
fn should_extract_fn_attributes() {
    let doc = test::mk_doc("#[doc = \"test\"] fn a() -> int { }");
    assert doc.cratemod().fns()[0].desc() == some("test");
}

fn fold_enum(
    fold: fold::fold<astsrv::srv>,
    doc: doc::enumdoc
) -> doc::enumdoc {

    let srv = fold.ctxt;
    let doc_id = doc.id();
    let doc = fold::default_seq_fold_enum(fold, doc);

    {
        variants: par::anymap(doc.variants) {|variant|
            let desc = astsrv::exec(srv) {|ctxt|
                alt check ctxt.ast_map.get(doc_id) {
                  ast_map::node_item(@{
                    node: ast::item_enum(ast_variants, _, _), _
                  }, _) {
                    let ast_variant = option::get(
                        vec::find(ast_variants) {|v|
                            v.node.name == variant.name
                        });

                    attr_parser::parse_desc(ast_variant.node.attrs)
                  }
                }
            };

            {
                desc: desc
                with variant
            }
        }
        with doc
    }
}

#[test]
fn should_extract_enum_docs() {
    let doc = test::mk_doc("#[doc = \"b\"]\
                            enum a { v }");
    assert doc.cratemod().enums()[0].desc() == some("b");
}

#[test]
fn should_extract_variant_docs() {
    let doc = test::mk_doc("enum a { #[doc = \"c\"] v }");
    assert doc.cratemod().enums()[0].variants[0].desc == some("c");
}

fn fold_iface(
    fold: fold::fold<astsrv::srv>,
    doc: doc::ifacedoc
) -> doc::ifacedoc {
    let srv = fold.ctxt;
    let doc = fold::default_seq_fold_iface(fold, doc);

    {
        methods: merge_method_attrs(srv, doc.id(), doc.methods)
        with doc
    }
}

fn merge_method_attrs(
    srv: astsrv::srv,
    item_id: doc::ast_id,
    docs: [doc::methoddoc]
) -> [doc::methoddoc] {

    // Create an assoc list from method name to attributes
    let attrs: [(str, option<str>)] = astsrv::exec(srv) {|ctxt|
        alt ctxt.ast_map.get(item_id) {
          ast_map::node_item(@{
            node: ast::item_iface(_, methods), _
          }, _) {
            par::seqmap(methods) {|method|
                (method.ident, attr_parser::parse_desc(method.attrs))
            }
          }
          ast_map::node_item(@{
            node: ast::item_impl(_, _, _, methods), _
          }, _) {
            par::seqmap(methods) {|method|
                (method.ident, attr_parser::parse_desc(method.attrs))
            }
          }
          _ { fail "unexpected item" }
        }
    };

    vec::map2(docs, attrs) {|doc, attrs|
        assert doc.name == tuple::first(attrs);
        let desc = tuple::second(attrs);

        {
            desc: desc
            with doc
        }
    }
}

#[test]
fn should_extract_iface_docs() {
    let doc = test::mk_doc("#[doc = \"whatever\"] iface i { fn a(); }");
    assert doc.cratemod().ifaces()[0].desc() == some("whatever");
}

#[test]
fn should_extract_iface_method_docs() {
    let doc = test::mk_doc(
        "iface i {\
         #[doc = \"desc\"]\
         fn f(a: bool) -> bool;\
         }");
    assert doc.cratemod().ifaces()[0].methods[0].desc == some("desc");
}


fn fold_impl(
    fold: fold::fold<astsrv::srv>,
    doc: doc::impldoc
) -> doc::impldoc {
    let srv = fold.ctxt;
    let doc = fold::default_seq_fold_impl(fold, doc);

    {
        methods: merge_method_attrs(srv, doc.id(), doc.methods)
        with doc
    }
}

#[test]
fn should_extract_impl_docs() {
    let doc = test::mk_doc(
        "#[doc = \"whatever\"] impl i for int { fn a() { } }");
    assert doc.cratemod().impls()[0].desc() == some("whatever");
}

#[test]
fn should_extract_impl_method_docs() {
    let doc = test::mk_doc(
        "impl i for int {\
         #[doc = \"desc\"]\
         fn f(a: bool) -> bool { }\
         }");
    assert doc.cratemod().impls()[0].methods[0].desc == some("desc");
}

#[cfg(test)]
mod test {
    fn mk_doc(source: str) -> doc::doc {
        astsrv::from_str(source) {|srv|
            let doc = extract::from_srv(srv, "");
            run(srv, doc)
        }
    }
}
