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
        fold_res: fold_res
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

    ~{
        topmod: ~{
            name: option::from_maybe(doc.topmod.name, attrs.name)
            with *doc.topmod
        }
    }
}

#[test]
fn should_replace_top_module_name_with_crate_name() {
    let source = "#[link(name = \"bond\")];";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let fold = fold::default_seq_fold(srv);
    let doc = fold_crate(fold, doc);
    assert doc.topmod.name == "bond";
}

fn parse_item_attrs<T>(
    srv: astsrv::srv,
    id: doc::ast_id,
    parse_attrs: fn~([ast::attribute]) -> T) -> T {
    astsrv::exec(srv) {|ctxt|
        let attrs = alt ctxt.ast_map.get(id) {
          ast_map::node_item(item) { item.attrs }
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
        ~{
            brief: attrs.brief,
            desc: attrs.desc
            with *doc
        }
    }
}

#[test]
fn fold_mod_should_extract_mod_attributes() {
    let source = "#[doc = \"test\"] mod a { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let fold = fold::default_seq_fold(srv);
    let doc = fold_mod(fold, doc.topmod.mods[0]);
    assert doc.desc == some("test");
}

#[test]
fn fold_mod_should_extract_top_mod_attributes() {
    let source = "#[doc = \"test\"];";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let fold = fold::default_seq_fold(srv);
    let doc = fold_mod(fold, doc.topmod);
    assert doc.desc == some("test");
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
        ret ~{
            brief: attrs.brief,
            desc: attrs.desc,
            args: merge_arg_attrs(doc.args, attrs.args),
            return: merge_ret_attrs(doc.return, attrs.return),
            failure: attrs.failure
            with *doc
        };
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
                    ~{
                        desc: some(attr.desc)
                        with *doc
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
}

#[test]
fn fold_fn_should_extract_fn_attributes() {
    let source = "#[doc = \"test\"] fn a() -> int { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let fold = fold::default_seq_fold(srv);
    let doc = fold_fn(fold, doc.topmod.fns[0]);
    assert doc.desc == some("test");
}

#[test]
fn fold_fn_should_extract_arg_attributes() {
    let source = "#[doc(args(a = \"b\"))] fn c(a: bool) { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let fold = fold::default_seq_fold(srv);
    let doc = fold_fn(fold, doc.topmod.fns[0]);
    assert doc.args[0].desc == some("b");
}

#[test]
fn fold_fn_should_extract_return_attributes() {
    let source = "#[doc(return = \"what\")] fn a() -> int { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = tystr_pass::mk_pass()(srv, doc);
    let fold = fold::default_seq_fold(srv);
    let doc = fold_fn(fold, doc.topmod.fns[0]);
    assert doc.return.desc == some("what");
}

#[test]
fn fold_fn_should_preserve_sig() {
    let source = "fn a() -> int { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = tystr_pass::mk_pass()(srv, doc);
    let fold = fold::default_seq_fold(srv);
    let doc = fold_fn(fold, doc.topmod.fns[0]);
    assert doc.sig == some("fn a() -> int");
}

#[test]
fn fold_fn_should_extract_failure_conditions() {
    let source = "#[doc(failure = \"what\")] fn a() { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let fold = fold::default_seq_fold(srv);
    let doc = fold_fn(fold, doc.topmod.fns[0]);
    assert doc.failure == some("what");
}

fn fold_const(
    fold: fold::fold<astsrv::srv>,
    doc: doc::constdoc
) -> doc::constdoc {
    let srv = fold.ctxt;
    let attrs = parse_item_attrs(srv, doc.id, attr_parser::parse_mod);

    ~{
        brief: attrs.brief,
        desc: attrs.desc
        with *doc
    }
}

#[test]
fn fold_const_should_extract_docs() {
    let source = "#[doc(brief = \"foo\", desc = \"bar\")]\
                  const a: bool = true;";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let fold = fold::default_seq_fold(srv);
    let doc = fold_const(fold, doc.topmod.consts[0]);
    assert doc.brief == some("foo");
    assert doc.desc == some("bar");
}

fn fold_enum(
    fold: fold::fold<astsrv::srv>,
    doc: doc::enumdoc
) -> doc::enumdoc {
    let srv = fold.ctxt;
    let attrs = parse_item_attrs(srv, doc.id, attr_parser::parse_enum);

    ~{
        brief: attrs.brief,
        desc: attrs.desc,
        variants: vec::map(doc.variants) {|variant|
            let attrs = astsrv::exec(srv) {|ctxt|
                alt ctxt.ast_map.get(doc.id) {
                  ast_map::node_item(@{
                    node: ast::item_enum(ast_variants, _), _
                  }) {
                    let ast_variant = option::get(
                        vec::find(ast_variants) {|v|
                            v.node.name == variant.name
                        });

                    attr_parser::parse_variant(ast_variant.node.attrs)
                  }
                }
            };

            ~{
                desc: attrs.desc
                with *variant
            }
        }
        with *doc
    }
}

#[test]
fn fold_enum_should_extract_docs() {
    let source = "#[doc(brief = \"a\", desc = \"b\")]\
                  enum a { v }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let fold = fold::default_seq_fold(srv);
    let doc = fold_enum(fold, doc.topmod.enums[0]);
    assert doc.brief == some("a");
    assert doc.desc == some("b");
}

#[test]
fn fold_enum_should_extract_variant_docs() {
    let source = "enum a { #[doc = \"c\"] v }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let fold = fold::default_seq_fold(srv);
    let doc = fold_enum(fold, doc.topmod.enums[0]);
    assert doc.variants[0].desc == some("c");
}

fn fold_res(
    fold: fold::fold<astsrv::srv>,
    doc: doc::resdoc
) -> doc::resdoc {

    let srv = fold.ctxt;
    let attrs = parse_item_attrs(srv, doc.id, attr_parser::parse_fn);

    ~{
        brief: attrs.brief,
        desc: attrs.desc,
        args: vec::map(doc.args) {|doc|
            alt vec::find(attrs.args) {|attr|
                attr.name == doc.name
            } {
                some(attr) {
                    ~{
                        desc: some(attr.desc)
                        with *doc
                    }
                }
                none { doc }
            }
        }
        with *doc
    }
}

#[test]
fn fold_res_should_extract_docs() {
    let source = "#[doc(brief = \"a\", desc = \"b\")]\
                  resource r(b: bool) { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let fold = fold::default_seq_fold(srv);
    let doc = fold_res(fold, doc.topmod.resources()[0]);
    assert doc.brief == some("a");
    assert doc.desc == some("b");
}

#[test]
fn fold_res_should_extract_arg_docs() {
    let source = "#[doc(args(a = \"b\"))]\
                  resource r(a: bool) { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let fold = fold::default_seq_fold(srv);
    let doc = fold_res(fold, doc.topmod.resources()[0]);
    assert doc.args[0].name == "a";
    assert doc.args[0].desc == some("b");
}