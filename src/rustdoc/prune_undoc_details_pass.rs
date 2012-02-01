#[doc = "Prunes args, retvals of the document tree that \
         contain no documentation"];

export mk_pass;

fn mk_pass() -> pass {
    run
}

fn run(
    _srv: astsrv::srv,
    doc: doc::cratedoc
) -> doc::cratedoc {
    let fold = fold::fold({
        fold_fn: fold_fn,
        fold_res: fold_res,
        fold_iface: fold_iface,
        fold_impl: fold_impl
        with *fold::default_seq_fold(())
    });
    fold.fold_crate(fold, doc)
}

fn fold_fn(
    fold: fold::fold<()>,
    doc: doc::fndoc
) -> doc::fndoc {
    let doc = fold::default_seq_fold_fn(fold, doc);

    {
        args: prune_args(doc.args),
        return: prune_return(doc.return)
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

fn prune_return(doc: doc::retdoc) -> doc::retdoc {
    {
        ty: if option::is_some(doc.desc) {
            doc.ty
        } else {
            none
        }
        with doc
    }
}

#[test]
fn should_elide_undocumented_arguments() {
    let doc = test::mk_doc("#[doc = \"hey\"] fn a(b: int) { }");
    assert vec::is_empty(doc.topmod.fns()[0].args);
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
    assert vec::is_empty(doc.topmod.resources()[0].args);
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
    vec::map(docs) {|doc|
        {
            args: prune_args(doc.args),
            return: prune_return(doc.return)
            with doc
        }
    }
}

#[test]
fn should_elide_undocumented_iface_method_args() {
    let doc = test::mk_doc("#[doc = \"hey\"] iface i { fn a(); }");
    assert vec::is_empty(doc.topmod.ifaces()[0].methods[0].args);
}

#[test]
fn should_elide_undocumented_iface_method_return_values() {
    let doc = test::mk_doc("#[doc = \"hey\"] iface i { fn a() -> int; }");
    assert doc.topmod.ifaces()[0].methods[0].return.ty == none;
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
    assert vec::is_empty(doc.topmod.impls()[0].methods[0].args);
}

#[test]
fn should_elide_undocumented_impl_method_return_values() {
    let doc = test::mk_doc(
        "#[doc = \"hey\"] impl i for int { fn a() -> int { } }");
    assert doc.topmod.impls()[0].methods[0].return.ty == none;
}

#[cfg(test)]
mod test {
    fn mk_doc(source: str) -> doc::cratedoc {
        let srv = astsrv::mk_srv_from_str(source);
        let doc = extract::from_srv(srv, "");
        let doc = attr_pass::mk_pass()(srv, doc);
        run(srv, doc)
    }
}
