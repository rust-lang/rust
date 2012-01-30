#[doc = "Records the full path to items"];

export mk_pass;

fn mk_pass() -> pass { run }

type ctxt = {
    srv: astsrv::srv,
    mutable path: [str]
};

fn run(srv: astsrv::srv, doc: doc::cratedoc) -> doc::cratedoc {
    let ctxt = {
        srv: srv,
        mutable path: []
    };
    let fold = fold::fold({
        fold_mod: fn~(
            f: fold::fold<ctxt>,
            d: doc::moddoc
        ) -> doc::moddoc {
            fold_mod(f, d)
        }
        with *fold::default_seq_fold(ctxt)
    });
    fold.fold_crate(fold, doc)
}

fn fold_mod(fold: fold::fold<ctxt>, doc: doc::moddoc) -> doc::moddoc {
    let is_topmod = doc.id == rustc::syntax::ast::crate_node_id;

    if !is_topmod { vec::push(fold.ctxt.path, doc.name); }
    let doc = fold::default_seq_fold_mod(fold, doc);
    if !is_topmod { vec::pop(fold.ctxt.path); }
    {
        path: fold.ctxt.path
        with doc
    }
}

#[test]
fn should_record_mod_paths() {
    let source = "mod a { mod b { mod c { } } mod d { mod e { } } }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert doc.topmod.mods()[0].mods()[0].mods()[0].path == ["a", "b"];
    assert doc.topmod.mods()[0].mods()[1].mods()[0].path == ["a", "d"];
}