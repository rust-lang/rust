//! Breaks rustdocs into sections according to their headers

export mk_pass;

fn mk_pass() -> pass {
    {
        name: ~"sectionalize",
        f: run
    }
}

fn run(_srv: astsrv::srv, doc: doc::doc) -> doc::doc {
    let fold = fold::fold({
        fold_item: fold_item,
        fold_trait: fold_trait,
        fold_impl: fold_impl
        with *fold::default_any_fold(())
    });
    fold.fold_doc(fold, doc)
}

fn fold_item(fold: fold::fold<()>, doc: doc::itemdoc) -> doc::itemdoc {
    let doc = fold::default_seq_fold_item(fold, doc);
    let (desc, sections) = sectionalize(doc.desc);

    {
        desc: desc,
        sections: sections
        with doc
    }
}

fn fold_trait(fold: fold::fold<()>, doc: doc::traitdoc) -> doc::traitdoc {
    let doc = fold::default_seq_fold_trait(fold, doc);

    {
        methods: do par::map(doc.methods) |method| {
            let (desc, sections) = sectionalize(method.desc);

            {
                desc: desc,
                sections: sections
                with method
            }
        }
        with doc
    }
}

fn fold_impl(fold: fold::fold<()>, doc: doc::impldoc) -> doc::impldoc {
    let doc = fold::default_seq_fold_impl(fold, doc);

    {
        methods: do par::map(doc.methods) |method| {
            let (desc, sections) = sectionalize(method.desc);

            {
                desc: desc,
                sections: sections
                with method
            }
        }
        with doc
    }
}

fn sectionalize(desc: option<~str>) -> (option<~str>, ~[doc::section]) {

    /*!
     * Take a description of the form
     *
     *     General text
     *
     *     # Section header
     *
     *     Section text
     *
     *     # Section header
     *
     *     Section text
     *
     * and remove each header and accompanying text into section records.
     */

    if option::is_none(desc) {
        ret (none, ~[]);
    }

    let lines = str::lines(option::get(desc));

    let mut new_desc = none::<~str>;
    let mut current_section = none;
    let mut sections = ~[];

    for lines.each |line| {
        alt parse_header(line) {
          some(header) {
            if option::is_some(current_section) {
                sections += ~[option::get(current_section)];
            }
            current_section = some({
                header: header,
                body: ~""
            });
          }
          none {
            alt copy current_section {
              some(section) {
                current_section = some({
                    body: section.body + ~"\n" + line
                    with section
                });
              }
              none {
                new_desc = alt new_desc {
                  some(desc) {
                    some(desc + ~"\n" + line)
                  }
                  none {
                    some(line)
                  }
                };
              }
            }
          }
        }
    }

    if option::is_some(current_section) {
        sections += ~[option::get(current_section)];
    }

    (new_desc, sections)
}

fn parse_header(line: ~str) -> option<~str> {
    if str::starts_with(line, ~"# ") {
        some(str::slice(line, 2u, str::len(line)))
    } else {
        none
    }
}

#[test]
fn should_create_section_headers() {
    let doc = test::mk_doc(
        ~"#[doc = \"\
         # Header\n\
         Body\"]\
         mod a { }");
    assert str::contains(
        doc.cratemod().mods()[0].item.sections[0].header,
        ~"Header");
}

#[test]
fn should_create_section_bodies() {
    let doc = test::mk_doc(
        ~"#[doc = \"\
         # Header\n\
         Body\"]\
         mod a { }");
    assert str::contains(
        doc.cratemod().mods()[0].item.sections[0].body,
        ~"Body");
}

#[test]
fn should_not_create_sections_from_indented_headers() {
    let doc = test::mk_doc(
        ~"#[doc = \"\n\
         Text\n             # Header\n\
         Body\"]\
         mod a { }");
    assert vec::is_empty(doc.cratemod().mods()[0].item.sections);
}

#[test]
fn should_remove_section_text_from_main_desc() {
    let doc = test::mk_doc(
        ~"#[doc = \"\
         Description\n\n\
         # Header\n\
         Body\"]\
         mod a { }");
    assert !str::contains(
        option::get(doc.cratemod().mods()[0].desc()),
        ~"Header");
    assert !str::contains(
        option::get(doc.cratemod().mods()[0].desc()),
        ~"Body");
}

#[test]
fn should_eliminate_desc_if_it_is_just_whitespace() {
    let doc = test::mk_doc(
        ~"#[doc = \"\
         # Header\n\
         Body\"]\
         mod a { }");
    assert doc.cratemod().mods()[0].desc() == none;
}

#[test]
fn should_sectionalize_trait_methods() {
    let doc = test::mk_doc(
        ~"iface i {
         #[doc = \"\
         # Header\n\
         Body\"]\
         fn a(); }");
    assert doc.cratemod().traits()[0].methods[0].sections.len() == 1u;
}

#[test]
fn should_sectionalize_impl_methods() {
    let doc = test::mk_doc(
        ~"impl i for bool {
         #[doc = \"\
         # Header\n\
         Body\"]\
         fn a() { } }");
    assert doc.cratemod().impls()[0].methods[0].sections.len() == 1u;
}

#[cfg(test)]
mod test {
    fn mk_doc(source: ~str) -> doc::doc {
        do astsrv::from_str(source) |srv| {
            let doc = extract::from_srv(srv, ~"");
            let doc = attr_pass::mk_pass().f(srv, doc);
            run(srv, doc)
        }
    }
}
