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
        fold_fn: fold_fn,
        fold_fnlist: fold_fnlist
        with *fold::default_seq_fold(ctxt)
    });
    fold.fold_crate(fold, doc)
}

fn fold_fn(
    fold: fold::fold<ctxt>,
    doc: doc::fndoc
) -> doc::fndoc {
    fold.ctxt.have_docs =
        doc.brief != none
        || doc.desc != none
        || doc.return.desc != none;
    ret doc;
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

#[cfg(test)]
mod tests {
    #[test]
    fn should_elide_undocumented_fns() {
        let source = "fn a() { }";
        let srv = astsrv::mk_srv_from_str(source);
        let doc = extract::from_srv(srv, "");
        let doc = run(srv, doc);
        assert vec::is_empty(*doc.topmod.fns);
    }
}