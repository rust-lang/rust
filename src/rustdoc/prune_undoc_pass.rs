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
        fold_modlist: fold_modlist,
        fold_fnlist: fold_fnlist
        with *fold::default_seq_fold(ctxt)
    });
    fold.fold_crate(fold, doc)
}

fn fold_mod(
    fold: fold::fold<ctxt>,
    doc: doc::moddoc
) -> doc::moddoc {
    let doc = fold::default_seq_fold_mod(fold, doc);
    fold.ctxt.have_docs =
        doc.brief != none
        || doc.desc != none
        || vec::is_not_empty(*doc.mods)
        || vec::is_not_empty(*doc.fns);
    ret doc;
}

fn fold_fn(
    fold: fold::fold<ctxt>,
    doc: doc::fndoc
) -> doc::fndoc {
    let doc = ~{
        args: vec::filter_map(doc.args) {|doc|
            if option::is_some(doc.desc) {
                some(doc)
            } else {
                none
            }
        }
        with *doc
    };

    fold.ctxt.have_docs =
        doc.brief != none
        || doc.desc != none
        || doc.return.desc != none;
    ret doc;
}

#[test]
fn should_elide_undocumented_arguments() {
    let source = "#[doc = \"hey\"] fn a(b: int) { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert vec::is_empty(doc.topmod.fns[0].args);
}

fn fold_modlist(
    fold: fold::fold<ctxt>,
    list: doc::modlist
) -> doc::modlist {
    doc::modlist(vec::filter_map(*list) {|doc|
        let doc = fold_mod(fold, doc);
        if fold.ctxt.have_docs {
            some(doc)
        } else {
            none
        }
    })
}

#[test]
fn should_elide_undocumented_mods() {
    let source = "mod a { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert vec::is_empty(*doc.topmod.mods);
}

#[test]
fn should_not_elide_undocument_mods_with_documented_mods() {
    let source = "mod a { #[doc = \"b\"] mod b { } }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert vec::is_not_empty(*doc.topmod.mods);
}

#[test]
fn should_not_elide_undocument_mods_with_documented_fns() {
    let source = "mod a { #[doc = \"b\"] fn b() { } }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert vec::is_not_empty(*doc.topmod.mods);
}

fn fold_fnlist(
    fold: fold::fold<ctxt>,
    list: doc::fnlist
) -> doc::fnlist {
    doc::fnlist(vec::filter_map(*list) {|doc|
        let doc = fold_fn(fold, doc);
        if fold.ctxt.have_docs {
            some(doc)
        } else {
            none
        }
    })
}

#[test]
fn should_elide_undocumented_fns() {
    let source = "fn a() { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert vec::is_empty(*doc.topmod.fns);
}
