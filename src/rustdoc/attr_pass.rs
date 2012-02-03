#[doc(
    brief = "The attribute parsing pass",
    desc =
    "Traverses the document tree, pulling relevant documention out of the \
     corresponding AST nodes. The information gathered here is the basis \
     of the natural-language documentation for a crate."
)];

import rustc::syntax::ast;
import rustc::middle::ast_map;

export mk_pass;

fn mk_pass() -> pass {
    run
}

fn run(
    srv: astsrv::srv,
    doc: doc::cratedoc
) -> doc::cratedoc {
    let fold = fold::fold({
        fold_crate: fold_crate,
        fold_mod: fold_mod,
        fold_fn: fold_fn,
        fold_const: fold_const,
        fold_enum: fold_enum,
        fold_res: fold_res,
        fold_iface: fold_iface,
        fold_impl: fold_impl,
        fold_type: fold_type
        with *fold::default_seq_fold(srv)
    });
    fold.fold_crate(fold, doc)
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
            name: option::from_maybe(doc.topmod.name, attrs.name)
            with doc.topmod
        }
    }
}

#[test]
fn should_replace_top_module_name_with_crate_name() {
    let doc = test::mk_doc("#[link(name = \"bond\")];");
    assert doc.topmod.name == "bond";
}

fn parse_item_attrs<T>(
    srv: astsrv::srv,
    id: doc::ast_id,
    parse_attrs: fn~([ast::attribute]) -> T) -> T {
    astsrv::exec(srv) {|ctxt|
        let attrs = alt ctxt.ast_map.get(id) {
          ast_map::node_item(item, _) { item.attrs }
          _ {
            fail "parse_item_attrs: not an item";
          }
        };
        parse_attrs(attrs)
    }
}

fn fold_mod(fold: fold::fold<astsrv::srv>, doc: doc::moddoc) -> doc::moddoc {
    let srv = fold.ctxt;
    let attrs = if doc.id == ast::crate_node_id {
        // This is the top-level mod, use the crate attributes
        astsrv::exec(srv) {|ctxt|
            attr_parser::parse_mod(ctxt.ast.node.attrs)
        }
    } else {
        parse_item_attrs(srv, doc.id, attr_parser::parse_mod)
    };
    let doc = fold::default_seq_fold_mod(fold, doc);
    ret merge_mod_attrs(doc, attrs);

    fn merge_mod_attrs(
        doc: doc::moddoc,
        attrs: attr_parser::mod_attrs
    ) -> doc::moddoc {
        {
            brief: attrs.brief,
            desc: attrs.desc
            with doc
        }
    }
}

#[test]
fn fold_mod_should_extract_mod_attributes() {
    let doc = test::mk_doc("#[doc = \"test\"] mod a { }");
    assert doc.topmod.mods()[0].desc == some("test");
}

#[test]
fn fold_mod_should_extract_top_mod_attributes() {
    let doc = test::mk_doc("#[doc = \"test\"];");
    assert doc.topmod.desc == some("test");
}

fn fold_fn(
    fold: fold::fold<astsrv::srv>,
    doc: doc::fndoc
) -> doc::fndoc {

    let srv = fold.ctxt;

    let attrs = parse_item_attrs(srv, doc.id, attr_parser::parse_fn);
    ret merge_fn_attrs(doc, attrs);

    fn merge_fn_attrs(
        doc: doc::fndoc,
        attrs: attr_parser::fn_attrs
    ) -> doc::fndoc {
        ret {
            brief: attrs.brief,
            desc: attrs.desc,
            args: merge_arg_attrs(doc.args, attrs.args),
            return: merge_ret_attrs(doc.return, attrs.return),
            failure: attrs.failure
            with doc
        };
    }
}

fn merge_arg_attrs(
    docs: [doc::argdoc],
    attrs: [attr_parser::arg_attrs]
) -> [doc::argdoc] {
    vec::map(docs) {|doc|
        alt vec::find(attrs) {|attr|
            attr.name == doc.name
        } {
            some(attr) {
                {
                    desc: some(attr.desc)
                    with doc
                }
            }
            none { doc }
        }
    }
    // FIXME: Warning when documenting a non-existent arg
}


fn merge_ret_attrs(
    doc: doc::retdoc,
    attrs: option<str>
) -> doc::retdoc {
    {
        desc: attrs
        with doc
    }
}

#[test]
fn fold_fn_should_extract_fn_attributes() {
    let doc = test::mk_doc("#[doc = \"test\"] fn a() -> int { }");
    assert doc.topmod.fns()[0].desc == some("test");
}

#[test]
fn fold_fn_should_extract_arg_attributes() {
    let doc = test::mk_doc("#[doc(args(a = \"b\"))] fn c(a: bool) { }");
    assert doc.topmod.fns()[0].args[0].desc == some("b");
}

#[test]
fn fold_fn_should_extract_return_attributes() {
    let source = "#[doc(return = \"what\")] fn a() -> int { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = tystr_pass::mk_pass()(srv, doc);
    let fold = fold::default_seq_fold(srv);
    let doc = fold_fn(fold, doc.topmod.fns()[0]);
    assert doc.return.desc == some("what");
}

#[test]
fn fold_fn_should_preserve_sig() {
    let source = "fn a() -> int { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = tystr_pass::mk_pass()(srv, doc);
    let fold = fold::default_seq_fold(srv);
    let doc = fold_fn(fold, doc.topmod.fns()[0]);
    assert doc.sig == some("fn a() -> int");
}

#[test]
fn fold_fn_should_extract_failure_conditions() {
    let doc = test::mk_doc("#[doc(failure = \"what\")] fn a() { }");
    assert doc.topmod.fns()[0].failure == some("what");
}

fn fold_const(
    fold: fold::fold<astsrv::srv>,
    doc: doc::constdoc
) -> doc::constdoc {
    let srv = fold.ctxt;
    let attrs = parse_item_attrs(srv, doc.id, attr_parser::parse_const);

    {
        brief: attrs.brief,
        desc: attrs.desc
        with doc
    }
}

#[test]
fn fold_const_should_extract_docs() {
    let doc = test::mk_doc("#[doc(brief = \"foo\", desc = \"bar\")]\
                            const a: bool = true;");
    assert doc.topmod.consts()[0].brief == some("foo");
    assert doc.topmod.consts()[0].desc == some("bar");
}

fn fold_enum(
    fold: fold::fold<astsrv::srv>,
    doc: doc::enumdoc
) -> doc::enumdoc {
    let srv = fold.ctxt;
    let attrs = parse_item_attrs(srv, doc.id, attr_parser::parse_enum);

    {
        brief: attrs.brief,
        desc: attrs.desc,
        variants: vec::map(doc.variants) {|variant|
            let attrs = astsrv::exec(srv) {|ctxt|
                alt ctxt.ast_map.get(doc.id) {
                  ast_map::node_item(@{
                    node: ast::item_enum(ast_variants, _), _
                  }, _) {
                    let ast_variant = option::get(
                        vec::find(ast_variants) {|v|
                            v.node.name == variant.name
                        });

                    attr_parser::parse_variant(ast_variant.node.attrs)
                  }
                  _ { fail "fold_enum: undocumented invariant"; }
                }
            };

            {
                desc: attrs.desc
                with variant
            }
        }
        with doc
    }
}

#[test]
fn fold_enum_should_extract_docs() {
    let doc = test::mk_doc("#[doc(brief = \"a\", desc = \"b\")]\
                            enum a { v }");
    assert doc.topmod.enums()[0].brief == some("a");
    assert doc.topmod.enums()[0].desc == some("b");
}

#[test]
fn fold_enum_should_extract_variant_docs() {
    let doc = test::mk_doc("enum a { #[doc = \"c\"] v }");
    assert doc.topmod.enums()[0].variants[0].desc == some("c");
}

fn fold_res(
    fold: fold::fold<astsrv::srv>,
    doc: doc::resdoc
) -> doc::resdoc {

    let srv = fold.ctxt;
    let attrs = parse_item_attrs(srv, doc.id, attr_parser::parse_fn);

    {
        brief: attrs.brief,
        desc: attrs.desc,
        args: vec::map(doc.args) {|doc|
            alt vec::find(attrs.args) {|attr|
                attr.name == doc.name
            } {
                some(attr) {
                    {
                        desc: some(attr.desc)
                        with doc
                    }
                }
                none { doc }
            }
        }
        with doc
    }
}

#[test]
fn fold_res_should_extract_docs() {
    let doc = test::mk_doc("#[doc(brief = \"a\", desc = \"b\")]\
                            resource r(b: bool) { }");
    assert doc.topmod.resources()[0].brief == some("a");
    assert doc.topmod.resources()[0].desc == some("b");
}

#[test]
fn fold_res_should_extract_arg_docs() {
    let doc = test::mk_doc("#[doc(args(a = \"b\"))]\
                            resource r(a: bool) { }");
    assert doc.topmod.resources()[0].args[0].name == "a";
    assert doc.topmod.resources()[0].args[0].desc == some("b");
}

fn fold_iface(
    fold: fold::fold<astsrv::srv>,
    doc: doc::ifacedoc
) -> doc::ifacedoc {
    let srv = fold.ctxt;
    let doc = fold::default_seq_fold_iface(fold, doc);
    let attrs = parse_item_attrs(srv, doc.id, attr_parser::parse_iface);

    {
        brief: attrs.brief,
        desc: attrs.desc,
        methods: merge_method_attrs(srv, doc.id, doc.methods)
        with doc
    }
}

fn merge_method_attrs(
    srv: astsrv::srv,
    item_id: doc::ast_id,
    docs: [doc::methoddoc]
) -> [doc::methoddoc] {
    // Create an assoc list from method name to attributes
    let attrs = astsrv::exec(srv) {|ctxt|
        alt ctxt.ast_map.get(item_id) {
          ast_map::node_item(@{
            node: ast::item_iface(_, methods), _
          }, _) {
            vec::map(methods) {|method|
                (method.ident, attr_parser::parse_method(method.attrs))
            }
          }
          ast_map::node_item(@{
            node: ast::item_impl(_, _, _, methods), _
          }, _) {
            vec::map(methods) {|method|
                (method.ident, attr_parser::parse_method(method.attrs))
            }
          }
          _ { fail "unexpected item" }
        }
    };

    vec::map2(docs, attrs) {|doc, attrs|
        assert doc.name == tuple::first(attrs);
        let attrs = tuple::second(attrs);

        {
            brief: attrs.brief,
            desc: attrs.desc,
            args: merge_arg_attrs(doc.args, attrs.args),
            return: merge_ret_attrs(doc.return, attrs.return),
            failure: attrs.failure
            with doc
        }
    }
}

#[test]
fn should_extract_iface_docs() {
    let doc = test::mk_doc("#[doc = \"whatever\"] iface i { fn a(); }");
    assert doc.topmod.ifaces()[0].desc == some("whatever");
}

#[test]
fn should_extract_iface_method_docs() {
    let doc = test::mk_doc(
        "iface i {\
         #[doc(\
         brief = \"brief\",\
         desc = \"desc\",\
         args(a = \"a\"),\
         return = \"return\",\
         failure = \"failure\")]\
         fn f(a: bool) -> bool;\
         }");
    assert doc.topmod.ifaces()[0].methods[0].brief == some("brief");
    assert doc.topmod.ifaces()[0].methods[0].desc == some("desc");
    assert doc.topmod.ifaces()[0].methods[0].args[0].desc == some("a");
    assert doc.topmod.ifaces()[0].methods[0].return.desc == some("return");
    assert doc.topmod.ifaces()[0].methods[0].failure == some("failure");
}


fn fold_impl(
    fold: fold::fold<astsrv::srv>,
    doc: doc::impldoc
) -> doc::impldoc {
    let srv = fold.ctxt;
    let doc = fold::default_seq_fold_impl(fold, doc);
    let attrs = parse_item_attrs(srv, doc.id, attr_parser::parse_impl);

    {
        brief: attrs.brief,
        desc: attrs.desc,
        methods: merge_method_attrs(srv, doc.id, doc.methods)
        with doc
    }
}

#[test]
fn should_extract_impl_docs() {
    let doc = test::mk_doc(
        "#[doc = \"whatever\"] impl i for int { fn a() { } }");
    assert doc.topmod.impls()[0].desc == some("whatever");
}

#[test]
fn should_extract_impl_method_docs() {
    let doc = test::mk_doc(
        "impl i for int {\
         #[doc(\
         brief = \"brief\",\
         desc = \"desc\",\
         args(a = \"a\"),\
         return = \"return\",\
         failure = \"failure\")]\
         fn f(a: bool) -> bool { }\
         }");
    assert doc.topmod.impls()[0].methods[0].brief == some("brief");
    assert doc.topmod.impls()[0].methods[0].desc == some("desc");
    assert doc.topmod.impls()[0].methods[0].args[0].desc == some("a");
    assert doc.topmod.impls()[0].methods[0].return.desc == some("return");
    assert doc.topmod.impls()[0].methods[0].failure == some("failure");
}

fn fold_type(
    fold: fold::fold<astsrv::srv>,
    doc: doc::tydoc
) -> doc::tydoc {
    let srv = fold.ctxt;
    let doc = fold::default_seq_fold_type(fold, doc);
    let attrs = parse_item_attrs(srv, doc.id, attr_parser::parse_type);

    {
        brief: attrs.brief,
        desc: attrs.desc
        with doc
    }
}

#[test]
fn should_extract_type_docs() {
    let doc = test::mk_doc(
        "#[doc(brief = \"brief\", desc = \"desc\")]\
         type t = int;");
    assert doc.topmod.types()[0].brief == some("brief");
    assert doc.topmod.types()[0].desc == some("desc");
}

#[cfg(test)]
mod test {
    fn mk_doc(source: str) -> doc::cratedoc {
        let srv = astsrv::mk_srv_from_str(source);
        let doc = extract::from_srv(srv, "");
        run(srv, doc)
    }
}