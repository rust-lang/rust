//! Generic pass for performing an operation on all descriptions

import doc::item_utils;

export mk_pass;

fn mk_pass(name: ~str, +op: fn~(~str) -> ~str) -> pass {
    {
        name: name,
        f: fn~(srv: astsrv::srv, doc: doc::doc) -> doc::doc {
            run(srv, doc, op)
        }
    }
}

type op = fn~(~str) -> ~str;

#[allow(non_implicitly_copyable_typarams)]
fn run(
    _srv: astsrv::srv,
    doc: doc::doc,
    op: op
) -> doc::doc {
    let fold = fold::fold({
        fold_item: fold_item,
        fold_enum: fold_enum,
        fold_trait: fold_trait,
        fold_impl: fold_impl
        with *fold::default_any_fold(op)
    });
    fold.fold_doc(fold, doc)
}

fn maybe_apply_op(op: op, s: option<~str>) -> option<~str> {
    option::map(s, |s| op(s) )
}

fn fold_item(fold: fold::fold<op>, doc: doc::itemdoc) -> doc::itemdoc {
    let doc = fold::default_seq_fold_item(fold, doc);

    {
        brief: maybe_apply_op(fold.ctxt, doc.brief),
        desc: maybe_apply_op(fold.ctxt, doc.desc),
        sections: apply_to_sections(fold.ctxt, doc.sections)
        with doc
    }
}

fn apply_to_sections(op: op, sections: ~[doc::section]) -> ~[doc::section] {
    par::map(sections, |section, copy op| {
        header: op(section.header),
        body: op(section.body)
    })
}

fn fold_enum(fold: fold::fold<op>, doc: doc::enumdoc) -> doc::enumdoc {
    let doc = fold::default_seq_fold_enum(fold, doc);

    {
        variants: do par::map(doc.variants) |variant, copy fold| {
            {
                desc: maybe_apply_op(fold.ctxt, variant.desc)
                with variant
            }
        }
        with doc
    }
}

fn fold_trait(fold: fold::fold<op>, doc: doc::traitdoc) -> doc::traitdoc {
    let doc = fold::default_seq_fold_trait(fold, doc);

    {
        methods: apply_to_methods(fold.ctxt, doc.methods)
        with doc
    }
}

fn apply_to_methods(op: op, docs: ~[doc::methoddoc]) -> ~[doc::methoddoc] {
    do par::map(docs) |doc, copy op| {
        {
            brief: maybe_apply_op(op, doc.brief),
            desc: maybe_apply_op(op, doc.desc),
            sections: apply_to_sections(op, doc.sections)
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
    let doc = test::mk_doc(~"#[doc = \" a \"] enum a { b }");
    assert doc.cratemod().enums()[0].brief() == some(~"a");
}

#[test]
fn should_execute_op_on_enum_desc() {
    let doc = test::mk_doc(~"#[doc = \" a \"] enum a { b }");
    assert doc.cratemod().enums()[0].desc() == some(~"a");
}

#[test]
fn should_execute_op_on_variant_desc() {
    let doc = test::mk_doc(~"enum a { #[doc = \" a \"] b }");
    assert doc.cratemod().enums()[0].variants[0].desc == some(~"a");
}

#[test]
fn should_execute_op_on_trait_brief() {
    let doc = test::mk_doc(
        ~"#[doc = \" a \"] trait i { fn a(); }");
    assert doc.cratemod().traits()[0].brief() == some(~"a");
}

#[test]
fn should_execute_op_on_trait_desc() {
    let doc = test::mk_doc(
        ~"#[doc = \" a \"] trait i { fn a(); }");
    assert doc.cratemod().traits()[0].desc() == some(~"a");
}

#[test]
fn should_execute_op_on_trait_method_brief() {
    let doc = test::mk_doc(
        ~"trait i { #[doc = \" a \"] fn a(); }");
    assert doc.cratemod().traits()[0].methods[0].brief == some(~"a");
}

#[test]
fn should_execute_op_on_trait_method_desc() {
    let doc = test::mk_doc(
        ~"trait i { #[doc = \" a \"] fn a(); }");
    assert doc.cratemod().traits()[0].methods[0].desc == some(~"a");
}

#[test]
fn should_execute_op_on_impl_brief() {
    let doc = test::mk_doc(
        ~"#[doc = \" a \"] impl int { fn a() { } }");
    assert doc.cratemod().impls()[0].brief() == some(~"a");
}

#[test]
fn should_execute_op_on_impl_desc() {
    let doc = test::mk_doc(
        ~"#[doc = \" a \"] impl int { fn a() { } }");
    assert doc.cratemod().impls()[0].desc() == some(~"a");
}

#[test]
fn should_execute_op_on_impl_method_brief() {
    let doc = test::mk_doc(
        ~"impl int { #[doc = \" a \"] fn a() { } }");
    assert doc.cratemod().impls()[0].methods[0].brief == some(~"a");
}

#[test]
fn should_execute_op_on_impl_method_desc() {
    let doc = test::mk_doc(
        ~"impl int { #[doc = \" a \"] fn a() { } }");
    assert doc.cratemod().impls()[0].methods[0].desc == some(~"a");
}

#[test]
fn should_execute_op_on_type_brief() {
    let doc = test::mk_doc(
        ~"#[doc = \" a \"] type t = int;");
    assert doc.cratemod().types()[0].brief() == some(~"a");
}

#[test]
fn should_execute_op_on_type_desc() {
    let doc = test::mk_doc(
        ~"#[doc = \" a \"] type t = int;");
    assert doc.cratemod().types()[0].desc() == some(~"a");
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
    fn mk_doc(source: ~str) -> doc::doc {
        do astsrv::from_str(source) |srv| {
            let doc = extract::from_srv(srv, ~"");
            let doc = attr_pass::mk_pass().f(srv, doc);
            let doc = desc_to_brief_pass::mk_pass().f(srv, doc);
            let doc = sectionalize_pass::mk_pass().f(srv, doc);
            mk_pass(~"", |s| str::trim(s) ).f(srv, doc)
        }
    }
}
