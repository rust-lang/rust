//! Generic pass for performing an operation on all descriptions

use doc::ItemUtils;

pub fn mk_pass(name: ~str, +op: fn~(~str) -> ~str) -> Pass {
    {
        name: name,
        f: fn~(move op, srv: astsrv::Srv, +doc: doc::Doc) -> doc::Doc {
            run(srv, doc, copy op)
        }
    }
}

type Op = fn~(~str) -> ~str;

#[allow(non_implicitly_copyable_typarams)]
fn run(
    _srv: astsrv::Srv,
    +doc: doc::Doc,
    +op: Op
) -> doc::Doc {
    let fold = fold::Fold({
        fold_item: fold_item,
        fold_enum: fold_enum,
        fold_trait: fold_trait,
        fold_impl: fold_impl,
        .. *fold::default_any_fold(move op)
    });
    fold.fold_doc(&fold, doc)
}

fn maybe_apply_op(op: Op, s: Option<~str>) -> Option<~str> {
    s.map(|s| op(*s) )
}

fn fold_item(fold: &fold::Fold<Op>, +doc: doc::ItemDoc) -> doc::ItemDoc {
    let doc = fold::default_seq_fold_item(fold, doc);

    {
        brief: maybe_apply_op(fold.ctxt, doc.brief),
        desc: maybe_apply_op(fold.ctxt, doc.desc),
        sections: apply_to_sections(fold.ctxt, doc.sections),
        .. doc
    }
}

fn apply_to_sections(op: Op, sections: ~[doc::Section]) -> ~[doc::Section] {
    par::map(sections, |section, copy op| {
        header: op(section.header),
        body: op(section.body)
    })
}

fn fold_enum(fold: &fold::Fold<Op>, +doc: doc::EnumDoc) -> doc::EnumDoc {
    let doc = fold::default_seq_fold_enum(fold, doc);
    let fold_copy = copy *fold;

    {
        variants: do par::map(doc.variants) |variant, copy fold_copy| {
            {
                desc: maybe_apply_op(fold_copy.ctxt, variant.desc),
                .. *variant
            }
        },
        .. doc
    }
}

fn fold_trait(fold: &fold::Fold<Op>, +doc: doc::TraitDoc) -> doc::TraitDoc {
    let doc = fold::default_seq_fold_trait(fold, doc);

    {
        methods: apply_to_methods(fold.ctxt, doc.methods),
        .. doc
    }
}

fn apply_to_methods(op: Op, docs: ~[doc::MethodDoc]) -> ~[doc::MethodDoc] {
    do par::map(docs) |doc, copy op| {
        {
            brief: maybe_apply_op(op, doc.brief),
            desc: maybe_apply_op(op, doc.desc),
            sections: apply_to_sections(op, doc.sections),
            .. *doc
        }
    }
}

fn fold_impl(fold: &fold::Fold<Op>, +doc: doc::ImplDoc) -> doc::ImplDoc {
    let doc = fold::default_seq_fold_impl(fold, doc);

    {
        methods: apply_to_methods(fold.ctxt, doc.methods),
        .. doc
    }
}

#[test]
fn should_execute_op_on_enum_brief() {
    let doc = test::mk_doc(~"#[doc = \" a \"] enum a { b }");
    assert doc.cratemod().enums()[0].brief() == Some(~"a");
}

#[test]
fn should_execute_op_on_enum_desc() {
    let doc = test::mk_doc(~"#[doc = \" a \"] enum a { b }");
    assert doc.cratemod().enums()[0].desc() == Some(~"a");
}

#[test]
fn should_execute_op_on_variant_desc() {
    let doc = test::mk_doc(~"enum a { #[doc = \" a \"] b }");
    assert doc.cratemod().enums()[0].variants[0].desc == Some(~"a");
}

#[test]
fn should_execute_op_on_trait_brief() {
    let doc = test::mk_doc(
        ~"#[doc = \" a \"] trait i { fn a(); }");
    assert doc.cratemod().traits()[0].brief() == Some(~"a");
}

#[test]
fn should_execute_op_on_trait_desc() {
    let doc = test::mk_doc(
        ~"#[doc = \" a \"] trait i { fn a(); }");
    assert doc.cratemod().traits()[0].desc() == Some(~"a");
}

#[test]
fn should_execute_op_on_trait_method_brief() {
    let doc = test::mk_doc(
        ~"trait i { #[doc = \" a \"] fn a(); }");
    assert doc.cratemod().traits()[0].methods[0].brief == Some(~"a");
}

#[test]
fn should_execute_op_on_trait_method_desc() {
    let doc = test::mk_doc(
        ~"trait i { #[doc = \" a \"] fn a(); }");
    assert doc.cratemod().traits()[0].methods[0].desc == Some(~"a");
}

#[test]
fn should_execute_op_on_impl_brief() {
    let doc = test::mk_doc(
        ~"#[doc = \" a \"] impl int { fn a() { } }");
    assert doc.cratemod().impls()[0].brief() == Some(~"a");
}

#[test]
fn should_execute_op_on_impl_desc() {
    let doc = test::mk_doc(
        ~"#[doc = \" a \"] impl int { fn a() { } }");
    assert doc.cratemod().impls()[0].desc() == Some(~"a");
}

#[test]
fn should_execute_op_on_impl_method_brief() {
    let doc = test::mk_doc(
        ~"impl int { #[doc = \" a \"] fn a() { } }");
    assert doc.cratemod().impls()[0].methods[0].brief == Some(~"a");
}

#[test]
fn should_execute_op_on_impl_method_desc() {
    let doc = test::mk_doc(
        ~"impl int { #[doc = \" a \"] fn a() { } }");
    assert doc.cratemod().impls()[0].methods[0].desc == Some(~"a");
}

#[test]
fn should_execute_op_on_type_brief() {
    let doc = test::mk_doc(
        ~"#[doc = \" a \"] type t = int;");
    assert doc.cratemod().types()[0].brief() == Some(~"a");
}

#[test]
fn should_execute_op_on_type_desc() {
    let doc = test::mk_doc(
        ~"#[doc = \" a \"] type t = int;");
    assert doc.cratemod().types()[0].desc() == Some(~"a");
}

#[test]
fn should_execute_on_item_section_headers() {
    let doc = test::mk_doc(
        ~"#[doc = \"\
         #    Header    \n\
         Body\"]\
         fn a() { }");
    assert doc.cratemod().fns()[0].sections()[0].header == ~"Header";
}

#[test]
fn should_execute_on_item_section_bodies() {
    let doc = test::mk_doc(
        ~"#[doc = \"\
         # Header\n\
         Body      \"]\
         fn a() { }");
    assert doc.cratemod().fns()[0].sections()[0].body == ~"Body";
}

#[test]
fn should_execute_on_trait_method_section_headers() {
    let doc = test::mk_doc(
        ~"trait i {
         #[doc = \"\
         # Header    \n\
         Body\"]\
         fn a(); }");
    assert doc.cratemod().traits()[0].methods[0].sections[0].header
        == ~"Header";
}

#[test]
fn should_execute_on_trait_method_section_bodies() {
    let doc = test::mk_doc(
        ~"trait i {
         #[doc = \"\
         # Header\n\
         Body     \"]\
         fn a(); }");
    assert doc.cratemod().traits()[0].methods[0].sections[0].body == ~"Body";
}

#[test]
fn should_execute_on_impl_method_section_headers() {
    let doc = test::mk_doc(
        ~"impl bool {
         #[doc = \"\
         # Header   \n\
         Body\"]\
         fn a() { } }");
    assert doc.cratemod().impls()[0].methods[0].sections[0].header
        == ~"Header";
}

#[test]
fn should_execute_on_impl_method_section_bodies() {
    let doc = test::mk_doc(
        ~"impl bool {
         #[doc = \"\
         # Header\n\
         Body    \"]\
         fn a() { } }");
    assert doc.cratemod().impls()[0].methods[0].sections[0].body == ~"Body";
}

#[cfg(test)]
mod test {
    #[legacy_exports];
    fn mk_doc(source: ~str) -> doc::Doc {
        do astsrv::from_str(source) |srv| {
            let doc = extract::from_srv(srv, ~"");
            let doc = attr_pass::mk_pass().f(srv, doc);
            let doc = desc_to_brief_pass::mk_pass().f(srv, doc);
            let doc = sectionalize_pass::mk_pass().f(srv, doc);
            mk_pass(~"", |s| str::trim(s) ).f(srv, doc)
        }
    }
}
