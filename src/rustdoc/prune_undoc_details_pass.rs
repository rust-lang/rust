#[doc = "Prunes args, retvals of the document tree that \
         contain no documentation"];

export mk_pass;

fn mk_pass() -> pass {
    {
        name: "prune_undoc_details",
        f: run
    }
}

fn run(
    _srv: astsrv::srv,
    doc: doc::doc
) -> doc::doc {
    let fold = fold::fold({
        fold_fn: fold_fn,
        fold_res: fold_res,
        fold_iface: fold_iface,
        fold_impl: fold_impl
        with *fold::default_any_fold(())
    });
    fold.fold_doc(fold, doc)
}

fn fold_fn(
    fold: fold::fold<()>,
    doc: doc::fndoc
) -> doc::fndoc {
    let doc = fold::default_seq_fold_fn(fold, doc);

    {
        args: prune_args(doc.args)
        with doc
    }
}

fn prune_args(docs: [doc::argdoc]) -> [doc::argdoc] {
    vec::filter_map(docs) {|doc|
        if option::is_some(doc.desc) {
            some(doc)
        } else {
            none
        }
    }
}

#[test]
fn should_elide_undocumented_arguments() {
    let doc = test::mk_doc("#[doc = \"hey\"] fn a(b: int) { }");
    assert vec::is_empty(doc.cratemod().fns()[0].args);
}

fn fold_res(
    fold: fold::fold<()>,
    doc: doc::resdoc
) -> doc::resdoc {
    let doc = fold::default_seq_fold_res(fold, doc);

    {
        args: prune_args(doc.args)
        with doc
    }
}

#[test]
fn should_elide_undocumented_resource_args() {
    let doc = test::mk_doc("#[doc = \"drunk\"]\
                            resource r(a: bool) { }");
    assert vec::is_empty(doc.cratemod().resources()[0].args);
}

fn fold_iface(
    fold: fold::fold<()>,
    doc: doc::ifacedoc
) -> doc::ifacedoc {
    let doc = fold::default_seq_fold_iface(fold, doc);

    {
        methods: prune_methods(doc.methods)
        with doc
    }
}

fn prune_methods(docs: [doc::methoddoc]) -> [doc::methoddoc] {
    par::anymap(docs) {|doc|
        {
            args: prune_args(doc.args)
            with doc
        }
    }
}

#[test]
fn should_elide_undocumented_iface_method_args() {
    let doc = test::mk_doc("#[doc = \"hey\"] iface i { fn a(); }");
    assert vec::is_empty(doc.cratemod().ifaces()[0].methods[0].args);
}

fn fold_impl(
    fold: fold::fold<()>,
    doc: doc::impldoc
) -> doc::impldoc {
    let doc = fold::default_seq_fold_impl(fold, doc);

    {
        methods: prune_methods(doc.methods)
        with doc
    }
}

#[test]
fn should_elide_undocumented_impl_method_args() {
    let doc = test::mk_doc(
        "#[doc = \"hey\"] impl i for int { fn a(b: bool) { } }");
    assert vec::is_empty(doc.cratemod().impls()[0].methods[0].args);
}

#[cfg(test)]
mod test {
    fn mk_doc(source: str) -> doc::doc {
        astsrv::from_str(source) {|srv|
            let doc = extract::from_srv(srv, "");
            let doc = attr_pass::mk_pass().f(srv, doc);
            run(srv, doc)
        }
    }
}
