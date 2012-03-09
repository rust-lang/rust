#[doc = "Generic pass for performing an operation on all descriptions"];

export mk_pass;

fn mk_pass(name: str, op: fn~(str) -> str) -> pass {
    {
        name: name,
        f: fn~(srv: astsrv::srv, doc: doc::doc) -> doc::doc {
            run(srv, doc, op)
        }
    }
}

type op = fn~(str) -> str;

fn run(
    _srv: astsrv::srv,
    doc: doc::doc,
    op: op
) -> doc::doc {
    let fold = fold::fold({
        fold_item: fold_item,
        fold_fn: fold_fn,
        fold_enum: fold_enum,
        fold_res: fold_res,
        fold_iface: fold_iface,
        fold_impl: fold_impl
        with *fold::default_any_fold(op)
    });
    fold.fold_doc(fold, doc)
}

fn maybe_apply_op(op: op, s: option<str>) -> option<str> {
    option::map(s) {|s| op(s) }
}

fn fold_item(fold: fold::fold<op>, doc: doc::itemdoc) -> doc::itemdoc {
    let doc = fold::default_seq_fold_item(fold, doc);

    {
        brief: maybe_apply_op(fold.ctxt, doc.brief),
        desc: maybe_apply_op(fold.ctxt, doc.desc)
        with doc
    }
}

fn fold_fn(fold: fold::fold<op>, doc: doc::fndoc) -> doc::fndoc {
    let fold_ctxt = fold.ctxt;
    let doc = fold::default_seq_fold_fn(fold, doc);

    {
        args: par::anymap(doc.args) {|doc|
            {
                desc: maybe_apply_op(fold_ctxt, doc.desc)
                with doc
            }
        },
        return: {
            desc: maybe_apply_op(fold.ctxt, doc.return.desc)
            with doc.return
        },
        failure: maybe_apply_op(fold.ctxt, doc.failure)
        with doc
    }
}

fn fold_enum(fold: fold::fold<op>, doc: doc::enumdoc) -> doc::enumdoc {
    let fold_ctxt = fold.ctxt;
    let doc = fold::default_seq_fold_enum(fold, doc);

    {
        variants: par::anymap(doc.variants) {|variant|
            {
                desc: maybe_apply_op(fold_ctxt, variant.desc)
                with variant
            }
        }
        with doc
    }
}

fn fold_res(fold: fold::fold<op>, doc: doc::resdoc) -> doc::resdoc {
    let fold_ctxt = fold.ctxt;
    let doc = fold::default_seq_fold_res(fold, doc);

    {
        args: par::anymap(doc.args) {|arg|
            {
                desc: maybe_apply_op(fold_ctxt, arg.desc)
                with arg
            }
        }
        with doc
    }
}

fn fold_iface(fold: fold::fold<op>, doc: doc::ifacedoc) -> doc::ifacedoc {
    let doc = fold::default_seq_fold_iface(fold, doc);

    {
        methods: apply_to_methods(fold.ctxt, doc.methods)
        with doc
    }
}

fn apply_to_methods(op: op, docs: [doc::methoddoc]) -> [doc::methoddoc] {
    par::anymap(docs) {|doc|
        {
            brief: maybe_apply_op(op, doc.brief),
            desc: maybe_apply_op(op, doc.desc),
            args: par::anymap(doc.args) {|doc|
                {
                    desc: maybe_apply_op(op, doc.desc)
                    with doc
                }
            },
            return: {
                desc: maybe_apply_op(op, doc.return.desc)
                with doc.return
            },
            failure: maybe_apply_op(op, doc.failure)
            with doc
        }
    }
}

fn fold_impl(fold: fold::fold<op>, doc: doc::impldoc) -> doc::impldoc {
    let doc = fold::default_seq_fold_impl(fold, doc);

    {
        methods: apply_to_methods(fold.ctxt, doc.methods)
        with doc
    }
}

#[test]
fn should_execute_op_on_enum_brief() {
    let doc = test::mk_doc("#[doc(brief = \" a \")] enum a { b }");
    assert doc.cratemod().enums()[0].brief() == some("a");
}

#[test]
fn should_execute_op_on_enum_desc() {
    let doc = test::mk_doc("#[doc(desc = \" a \")] enum a { b }");
    assert doc.cratemod().enums()[0].desc() == some("a");
}

#[test]
fn should_execute_op_on_variant_desc() {
    let doc = test::mk_doc("enum a { #[doc = \" a \"] b }");
    assert doc.cratemod().enums()[0].variants[0].desc == some("a");
}

#[test]
fn should_execute_op_on_resource_brief() {
    let doc = test::mk_doc("#[doc(brief = \" a \")] resource r(a: bool) { }");
    assert doc.cratemod().resources()[0].brief() == some("a");
}

#[test]
fn should_execute_op_on_resource_desc() {
    let doc = test::mk_doc("#[doc(desc = \" a \")] resource r(a: bool) { }");
    assert doc.cratemod().resources()[0].desc() == some("a");
}

#[test]
fn should_execute_op_on_resource_args() {
    let doc = test::mk_doc(
        "#[doc(args(a = \" a \"))] resource r(a: bool) { }");
    assert doc.cratemod().resources()[0].args[0].desc == some("a");
}

#[test]
fn should_execute_op_on_iface_brief() {
    let doc = test::mk_doc(
        "#[doc(brief = \" a \")] iface i { fn a(); }");
    assert doc.cratemod().ifaces()[0].brief() == some("a");
}

#[test]
fn should_execute_op_on_iface_desc() {
    let doc = test::mk_doc(
        "#[doc(desc = \" a \")] iface i { fn a(); }");
    assert doc.cratemod().ifaces()[0].desc() == some("a");
}

#[test]
fn should_execute_op_on_iface_method_brief() {
    let doc = test::mk_doc(
        "iface i { #[doc(brief = \" a \")] fn a(); }");
    assert doc.cratemod().ifaces()[0].methods[0].brief == some("a");
}

#[test]
fn should_execute_op_on_iface_method_desc() {
    let doc = test::mk_doc(
        "iface i { #[doc(desc = \" a \")] fn a(); }");
    assert doc.cratemod().ifaces()[0].methods[0].desc == some("a");
}

#[test]
fn should_execute_op_on_iface_method_args() {
    let doc = test::mk_doc(
        "iface i { #[doc(args(a = \" a \"))] fn a(a: bool); }");
    assert doc.cratemod().ifaces()[0].methods[0].args[0].desc == some("a");
}

#[test]
fn should_execute_op_on_iface_method_return() {
    let doc = test::mk_doc(
        "iface i { #[doc(return = \" a \")] fn a() -> int; }");
    assert doc.cratemod().ifaces()[0].methods[0].return.desc == some("a");
}

#[test]
fn should_execute_op_on_iface_method_failure_condition() {
    let doc = test::mk_doc("iface i { #[doc(failure = \" a \")] fn a(); }");
    assert doc.cratemod().ifaces()[0].methods[0].failure == some("a");
}

#[test]
fn should_execute_op_on_impl_brief() {
    let doc = test::mk_doc(
        "#[doc(brief = \" a \")] impl i for int { fn a() { } }");
    assert doc.cratemod().impls()[0].brief() == some("a");
}

#[test]
fn should_execute_op_on_impl_desc() {
    let doc = test::mk_doc(
        "#[doc(desc = \" a \")] impl i for int { fn a() { } }");
    assert doc.cratemod().impls()[0].desc() == some("a");
}

#[test]
fn should_execute_op_on_impl_method_brief() {
    let doc = test::mk_doc(
        "impl i for int { #[doc(brief = \" a \")] fn a() { } }");
    assert doc.cratemod().impls()[0].methods[0].brief == some("a");
}

#[test]
fn should_execute_op_on_impl_method_desc() {
    let doc = test::mk_doc(
        "impl i for int { #[doc(desc = \" a \")] fn a() { } }");
    assert doc.cratemod().impls()[0].methods[0].desc == some("a");
}

#[test]
fn should_execute_op_on_impl_method_args() {
    let doc = test::mk_doc(
        "impl i for int { #[doc(args(a = \" a \"))] fn a(a: bool) { } }");
    assert doc.cratemod().impls()[0].methods[0].args[0].desc == some("a");
}

#[test]
fn should_execute_op_on_impl_method_return() {
    let doc = test::mk_doc(
        "impl i for int { #[doc(return = \" a \")] fn a() -> int { fail } }");
    assert doc.cratemod().impls()[0].methods[0].return.desc == some("a");
}

#[test]
fn should_execute_op_on_impl_method_failure_condition() {
    let doc = test::mk_doc(
        "impl i for int { #[doc(failure = \" a \")] fn a() { } }");
    assert doc.cratemod().impls()[0].methods[0].failure == some("a");
}


#[test]
fn should_execute_op_on_type_brief() {
    let doc = test::mk_doc(
        "#[doc(brief = \" a \")] type t = int;");
    assert doc.cratemod().types()[0].brief() == some("a");
}

#[test]
fn should_execute_op_on_type_desc() {
    let doc = test::mk_doc(
        "#[doc(desc = \" a \")] type t = int;");
    assert doc.cratemod().types()[0].desc() == some("a");
}

#[cfg(test)]
mod test {
    fn mk_doc(source: str) -> doc::doc {
        astsrv::from_str(source) {|srv|
            let doc = extract::from_srv(srv, "");
            let doc = attr_pass::mk_pass().f(srv, doc);
            mk_pass("", {|s| str::trim(s)}).f(srv, doc)
        }
    }
}