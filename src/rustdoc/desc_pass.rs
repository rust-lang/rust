#[doc = "Generic pass for performing an operation on all descriptions"];

export mk_pass;

fn mk_pass(op: fn~(str) -> str) -> pass {
    fn~(srv: astsrv::srv, doc: doc::cratedoc) -> doc::cratedoc {
        run(srv, doc, op)
    }
}

type op = fn~(str) -> str;

fn run(
    _srv: astsrv::srv,
    doc: doc::cratedoc,
    op: op
) -> doc::cratedoc {
    let fold = fold::fold({
        fold_mod: fold_mod,
        fold_const: fold_const,
        fold_fn: fold_fn,
        fold_enum: fold_enum,
        fold_res: fold_res,
        fold_iface: fold_iface,
        fold_impl: fold_impl
        with *fold::default_seq_fold(op)
    });
    fold.fold_crate(fold, doc)
}

fn maybe_apply_op(op: op, s: option<str>) -> option<str> {
    option::map(s) {|s| op(s) }
}

fn fold_mod(fold: fold::fold<op>, doc: doc::moddoc) -> doc::moddoc {
    let doc = fold::default_seq_fold_mod(fold, doc);

    {
        brief: maybe_apply_op(fold.ctxt, doc.brief),
        desc: maybe_apply_op(fold.ctxt, doc.desc)
        with doc
    }
}

fn fold_const(fold: fold::fold<op>, doc: doc::constdoc) -> doc::constdoc {
    let doc = fold::default_seq_fold_const(fold, doc);

    {
        brief: maybe_apply_op(fold.ctxt, doc.brief),
        desc: maybe_apply_op(fold.ctxt, doc.desc)
        with doc
    }
}

fn fold_fn(fold: fold::fold<op>, doc: doc::fndoc) -> doc::fndoc {
    let doc = fold::default_seq_fold_fn(fold, doc);

    {
        brief: maybe_apply_op(fold.ctxt, doc.brief),
        desc: maybe_apply_op(fold.ctxt, doc.desc),
        args: vec::map(doc.args) {|doc|
            {
                desc: maybe_apply_op(fold.ctxt, doc.desc)
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
    {
        brief: maybe_apply_op(fold.ctxt, doc.brief),
        desc: maybe_apply_op(fold.ctxt, doc.desc),
        variants: vec::map(doc.variants) {|variant|
            {
                desc: maybe_apply_op(fold.ctxt, variant.desc)
                with variant
            }
        }
        with doc
    }
}

fn fold_res(fold: fold::fold<op>, doc: doc::resdoc) -> doc::resdoc {
    {
        brief: maybe_apply_op(fold.ctxt, doc.brief),
        desc: maybe_apply_op(fold.ctxt, doc.desc),
        args: vec::map(doc.args) {|arg|
            {
                desc: maybe_apply_op(fold.ctxt, arg.desc)
                with arg
            }
        }
        with doc
    }
}

fn fold_iface(fold: fold::fold<op>, doc: doc::ifacedoc) -> doc::ifacedoc {
    {
        brief: maybe_apply_op(fold.ctxt, doc.brief),
        desc: maybe_apply_op(fold.ctxt, doc.desc),
        methods: apply_to_methods(fold.ctxt, doc.methods)
        with doc
    }
}

fn apply_to_methods(op: op, docs: [doc::methoddoc]) -> [doc::methoddoc] {
    vec::map(docs) {|doc|
        {
            brief: maybe_apply_op(op, doc.brief),
            desc: maybe_apply_op(op, doc.desc),
            args: vec::map(doc.args) {|doc|
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
    {
        brief: maybe_apply_op(fold.ctxt, doc.brief),
        desc: maybe_apply_op(fold.ctxt, doc.desc),
        methods: apply_to_methods(fold.ctxt, doc.methods)
        with doc
    }
}

#[test]
fn should_execute_op_on_enum_brief() {
    let doc = test::mk_doc("#[doc(brief = \" a \")] enum a { b }");
    assert doc.topmod.enums()[0].brief == some("a");
}

#[test]
fn should_execute_op_on_enum_desc() {
    let doc = test::mk_doc("#[doc(desc = \" a \")] enum a { b }");
    assert doc.topmod.enums()[0].desc == some("a");
}

#[test]
fn should_execute_op_on_variant_desc() {
    let doc = test::mk_doc("enum a { #[doc = \" a \"] b }");
    assert doc.topmod.enums()[0].variants[0].desc == some("a");
}

#[test]
fn should_execute_op_on_resource_brief() {
    let doc = test::mk_doc("#[doc(brief = \" a \")] resource r(a: bool) { }");
    assert doc.topmod.resources()[0].brief == some("a");
}

#[test]
fn should_execute_op_on_resource_desc() {
    let doc = test::mk_doc("#[doc(desc = \" a \")] resource r(a: bool) { }");
    assert doc.topmod.resources()[0].desc == some("a");
}

#[test]
fn should_execute_op_on_resource_args() {
    let doc = test::mk_doc(
        "#[doc(args(a = \" a \"))] resource r(a: bool) { }");
    assert doc.topmod.resources()[0].args[0].desc == some("a");
}

#[test]
fn should_execute_op_on_iface_brief() {
    let doc = test::mk_doc(
        "#[doc(brief = \" a \")] iface i { fn a(); }");
    assert doc.topmod.ifaces()[0].brief == some("a");
}

#[test]
fn should_execute_op_on_iface_desc() {
    let doc = test::mk_doc(
        "#[doc(desc = \" a \")] iface i { fn a(); }");
    assert doc.topmod.ifaces()[0].desc == some("a");
}

#[test]
fn should_execute_op_on_iface_method_brief() {
    let doc = test::mk_doc(
        "iface i { #[doc(brief = \" a \")] fn a(); }");
    assert doc.topmod.ifaces()[0].methods[0].brief == some("a");
}

#[test]
fn should_execute_op_on_iface_method_desc() {
    let doc = test::mk_doc(
        "iface i { #[doc(desc = \" a \")] fn a(); }");
    assert doc.topmod.ifaces()[0].methods[0].desc == some("a");
}

#[test]
fn should_execute_op_on_iface_method_args() {
    let doc = test::mk_doc(
        "iface i { #[doc(args(a = \" a \"))] fn a(a: bool); }");
    assert doc.topmod.ifaces()[0].methods[0].args[0].desc == some("a");
}

#[test]
fn should_execute_op_on_iface_method_return() {
    let doc = test::mk_doc(
        "iface i { #[doc(return = \" a \")] fn a() -> int; }");
    assert doc.topmod.ifaces()[0].methods[0].return.desc == some("a");
}

#[test]
fn should_execute_op_on_iface_method_failure_condition() {
    let doc = test::mk_doc("iface i { #[doc(failure = \" a \")] fn a(); }");
    assert doc.topmod.ifaces()[0].methods[0].failure == some("a");
}

#[test]
fn should_execute_op_on_impl_brief() {
    let doc = test::mk_doc(
        "#[doc(brief = \" a \")] impl i for int { fn a() { } }");
    assert doc.topmod.impls()[0].brief == some("a");
}

#[test]
fn should_execute_op_on_impl_desc() {
    let doc = test::mk_doc(
        "#[doc(desc = \" a \")] impl i for int { fn a() { } }");
    assert doc.topmod.impls()[0].desc == some("a");
}

#[test]
fn should_execute_op_on_impl_method_brief() {
    let doc = test::mk_doc(
        "impl i for int { #[doc(brief = \" a \")] fn a() { } }");
    assert doc.topmod.impls()[0].methods[0].brief == some("a");
}

#[test]
fn should_execute_op_on_impl_method_desc() {
    let doc = test::mk_doc(
        "impl i for int { #[doc(desc = \" a \")] fn a() { } }");
    assert doc.topmod.impls()[0].methods[0].desc == some("a");
}

#[test]
fn should_execute_op_on_impl_method_args() {
    let doc = test::mk_doc(
        "impl i for int { #[doc(args(a = \" a \"))] fn a(a: bool) { } }");
    assert doc.topmod.impls()[0].methods[0].args[0].desc == some("a");
}

#[test]
fn should_execute_op_on_impl_method_return() {
    let doc = test::mk_doc(
        "impl i for int { #[doc(return = \" a \")] fn a() -> int { fail } }");
    assert doc.topmod.impls()[0].methods[0].return.desc == some("a");
}

#[test]
fn should_execute_op_on_impl_method_failure_condition() {
    let doc = test::mk_doc(
        "impl i for int { #[doc(failure = \" a \")] fn a() { } }");
    assert doc.topmod.impls()[0].methods[0].failure == some("a");
}

#[cfg(test)]
mod test {
    fn mk_doc(source: str) -> doc::cratedoc {
        let srv = astsrv::mk_srv_from_str(source);
        let doc = extract::from_srv(srv, "");
        let doc = attr_pass::mk_pass()(srv, doc);
        mk_pass(str::trim)(srv, doc)
    }
}