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
        fold_fn: fold_fn
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

fn fold_mod(fold: fold::fold<astsrv::srv>, doc: doc::moddoc) -> doc::moddoc {
    let srv = fold.ctxt;
    let attrs = if doc.id == ast::crate_node_id {
        // This is the top-level mod, use the crate attributes
        astsrv::exec(srv) {|ctxt|
            attr_parser::parse_mod(ctxt.ast.node.attrs)
        }
    } else {
        astsrv::exec(srv) {|ctxt|
            let attrs = alt ctxt.map.get(doc.id) {
              ast_map::node_item(item) { item.attrs }
            };
            attr_parser::parse_mod(attrs)
        }
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

    let attrs = astsrv::exec(srv) {|ctxt|
        let attrs = alt ctxt.map.get(doc.id) {
          ast_map::node_item(item) { item.attrs }
        };
        attr_parser::parse_fn(attrs)
    };
    ret merge_fn_attrs(doc, attrs);

    fn merge_fn_attrs(
        doc: doc::fndoc,
        attrs: attr_parser::fn_attrs
    ) -> doc::fndoc {
        ret ~{
            id: doc.id,
            name: doc.name,
            brief: attrs.brief,
            desc: attrs.desc,
            args: merge_arg_attrs(doc.args, attrs.args),
            return: merge_ret_attrs(doc.return, attrs.return),
            ty: none
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
        doc: option<doc::retdoc>,
        attrs: option<str>
    ) -> option<doc::retdoc> {
        alt doc {
          some(doc) {
            some({
                desc: attrs
                with doc
            })
          }
          none {
            // FIXME: Warning about documenting nil?
            none
          }
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
    assert option::get(doc.return).desc == some("what");
}