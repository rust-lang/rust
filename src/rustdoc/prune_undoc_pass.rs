#[doc = "Prunes branches of the document tree that contain no documentation"];

export mk_pass;

fn mk_pass() -> pass {
    run
}

type ctxt = {
    mutable have_docs: bool
};

fn run(
    _srv: astsrv::srv,
    doc: doc::cratedoc
) -> doc::cratedoc {
    let ctxt = {
        mutable have_docs: true
    };
    let fold = fold::fold({
        fold_mod: fold_mod,
        fold_fn: fold_fn,
        fold_const: fold_const,
        fold_enum: fold_enum,
        fold_res: fold_res
        with *fold::default_seq_fold(ctxt)
    });
    fold.fold_crate(fold, doc)
}

fn fold_mod(
    fold: fold::fold<ctxt>,
    doc: doc::moddoc
) -> doc::moddoc {
    let doc = ~{
        items: vec::filter_map(doc.items) {|itemtag|
            alt itemtag {
              doc::modtag(moddoc) {
                let doc = fold.fold_mod(fold, moddoc);
                if fold.ctxt.have_docs {
                    some(doc::modtag(doc))
                } else {
                    none
                }
              }
              doc::fntag(fndoc) {
                let doc = fold.fold_fn(fold, fndoc);
                if fold.ctxt.have_docs {
                    some(doc::fntag(doc))
                } else {
                    none
                }
              }
              doc::consttag(constdoc) {
                let doc = fold.fold_const(fold, constdoc);
                if fold.ctxt.have_docs {
                    some(doc::consttag(doc))
                } else {
                    none
                }
              }
              doc::enumtag(enumdoc) {
                let doc = fold.fold_enum(fold, enumdoc);
                if fold.ctxt.have_docs {
                    some(doc::enumtag(doc))
                } else {
                    none
                }
              }
              doc::restag(resdoc) {
                let doc = fold.fold_res(fold, resdoc);
                if fold.ctxt.have_docs {
                    some(doc::restag(doc))
                } else {
                    none
                }
              }
              _ { some(itemtag) }
            }
        }
        with *fold::default_seq_fold_mod(fold, doc)
    };
    fold.ctxt.have_docs =
        doc.brief != none
        || doc.desc != none
        || vec::is_not_empty(doc.items);
    ret doc;
}

fn fold_fn(
    fold: fold::fold<ctxt>,
    doc: doc::fndoc
) -> doc::fndoc {
    let have_arg_docs = false;
    let doc = ~{
        args: vec::filter_map(doc.args) {|doc|
            if option::is_some(doc.desc) {
                have_arg_docs = true;
                some(doc)
            } else {
                none
            }
        },
        return: {
            ty: if option::is_some(doc.return.desc) {
                doc.return.ty
            } else {
                none
            }
            with doc.return
        }
        with *doc
    };

    fold.ctxt.have_docs =
        doc.brief != none
        || doc.desc != none
        || have_arg_docs
        || doc.return.desc != none
        || doc.failure != none;
    ret doc;
}

#[test]
fn should_elide_undocumented_arguments() {
    let source = "#[doc = \"hey\"] fn a(b: int) { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert vec::is_empty(doc.topmod.fns()[0].args);
}

#[test]
fn should_not_elide_fns_with_documented_arguments() {
    let source = "#[doc(args(a = \"b\"))] fn a(a: int) { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert vec::is_not_empty(doc.topmod.fns());
}

#[test]
fn should_elide_undocumented_return_values() {
    let source = "#[doc = \"fonz\"] fn a() -> int { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = tystr_pass::mk_pass()(srv, doc);
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert doc.topmod.fns()[0].return.ty == none;
}

#[test]
fn should_not_elide_fns_with_documented_failure_conditions() {
    let source = "#[doc(failure = \"yup\")] fn a() { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert vec::is_not_empty(doc.topmod.fns());
}

#[test]
fn should_elide_undocumented_mods() {
    let source = "mod a { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert vec::is_empty(doc.topmod.mods());
}

#[test]
fn should_not_elide_undocument_mods_with_documented_mods() {
    let source = "mod a { #[doc = \"b\"] mod b { } }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert vec::is_not_empty(doc.topmod.mods());
}

#[test]
fn should_not_elide_undocument_mods_with_documented_fns() {
    let source = "mod a { #[doc = \"b\"] fn b() { } }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert vec::is_not_empty(doc.topmod.mods());
}

#[test]
fn should_elide_undocumented_fns() {
    let source = "fn a() { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert vec::is_empty(doc.topmod.fns());
}

fn fold_const(
    fold: fold::fold<ctxt>,
    doc: doc::constdoc
) -> doc::constdoc {
    let doc = fold::default_seq_fold_const(fold, doc);
    fold.ctxt.have_docs =
        doc.brief != none
        || doc.desc != none;
    ret doc;
}

#[test]
fn should_elide_undocumented_consts() {
    let source = "const a: bool = true;";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert vec::is_empty(doc.topmod.consts());
}

fn fold_enum(fold: fold::fold<ctxt>, doc: doc::enumdoc) -> doc::enumdoc {
    let doc = ~{
        variants: vec::filter_map(doc.variants) {|variant|
            if variant.desc != none {
                some(variant)
            } else {
                none
            }
        }
        with *fold::default_seq_fold_enum(fold, doc)
    };
    fold.ctxt.have_docs =
        doc.brief != none
        || doc.desc != none
        || vec::is_not_empty(doc.variants);
    ret doc;
}

#[test]
fn should_elide_undocumented_enums() {
    let source = "enum a { b }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert vec::is_empty(doc.topmod.enums());
}

#[test]
fn should_elide_undocumented_variants() {
    let source = "#[doc = \"a\"] enum a { b }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert vec::is_empty(doc.topmod.enums()[0].variants);
}

#[test]
fn should_not_elide_enums_with_documented_variants() {
    let source = "enum a { #[doc = \"a\"] b }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert vec::is_not_empty(doc.topmod.enums());
}

fn fold_res(fold: fold::fold<ctxt>, doc: doc::resdoc) -> doc::resdoc {
    let doc = ~{
        args: vec::filter_map(doc.args) {|arg|
            if arg.desc != none {
                some(arg)
            } else {
                none
            }
        }
        with *fold::default_seq_fold_res(fold, doc)
    };
    fold.ctxt.have_docs =
        doc.brief != none
        || doc.desc != none
        || vec::is_not_empty(doc.args);
    ret doc;
}

#[test]
fn should_elide_undocumented_resources() {
    let source = "resource r(a: bool) { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert vec::is_empty(doc.topmod.resources());
}

#[test]
fn should_elide_undocumented_resource_args() {
    let source = "#[doc = \"drunk\"]\
                  resource r(a: bool) { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert vec::is_empty(doc.topmod.resources()[0].args);
}

#[test]
fn should_not_elide_resources_with_documented_args() {
    let source = "#[doc(args(a = \"drunk\"))]\
                  resource r(a: bool) { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert vec::is_not_empty(doc.topmod.resources());
}
