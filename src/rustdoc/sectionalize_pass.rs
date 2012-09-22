//! Breaks rustdocs into sections according to their headers

use doc::ItemUtils;

export mk_pass;

fn mk_pass() -> Pass {
    {
        name: ~"sectionalize",
        f: run
    }
}

fn run(_srv: astsrv::Srv, doc: doc::Doc) -> doc::Doc {
    let fold = fold::Fold({
        fold_item: fold_item,
        fold_trait: fold_trait,
        fold_impl: fold_impl,
        .. *fold::default_any_fold(())
    });
    fold.fold_doc(fold, doc)
}

fn fold_item(fold: fold::Fold<()>, doc: doc::ItemDoc) -> doc::ItemDoc {
    let doc = fold::default_seq_fold_item(fold, doc);
    let (desc, sections) = sectionalize(doc.desc);

    {
        desc: desc,
        sections: sections,
        .. doc
    }
}

fn fold_trait(fold: fold::Fold<()>, doc: doc::TraitDoc) -> doc::TraitDoc {
    let doc = fold::default_seq_fold_trait(fold, doc);

    {
        methods: do par::map(doc.methods) |method| {
            let (desc, sections) = sectionalize(method.desc);

            {
                desc: desc,
                sections: sections,
                ..method
            }
        },
        .. doc
    }
}

fn fold_impl(fold: fold::Fold<()>, doc: doc::ImplDoc) -> doc::ImplDoc {
    let doc = fold::default_seq_fold_impl(fold, doc);

    {
        methods: do par::map(doc.methods) |method| {
            let (desc, sections) = sectionalize(method.desc);

            {
                desc: desc,
                sections: sections,
                .. method
            }
        },
        .. doc
    }
}

fn sectionalize(desc: Option<~str>) -> (Option<~str>, ~[doc::Section]) {

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
        return (None, ~[]);
    }

    let lines = str::lines(option::get(desc));

    let mut new_desc = None::<~str>;
    let mut current_section = None;
    let mut sections = ~[];

    for lines.each |line| {
        match parse_header(*line) {
          Some(header) => {
            if option::is_some(current_section) {
                sections += ~[option::get(current_section)];
            }
            current_section = Some({
                header: header,
                body: ~""
            });
          }
          None => {
            match copy current_section {
              Some(section) => {
                current_section = Some({
                    body: section.body + ~"\n" + *line,
                    .. section
                });
              }
              None => {
                new_desc = match new_desc {
                  Some(desc) => {
                    Some(desc + ~"\n" + *line)
                  }
                  None => {
                    Some(*line)
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

fn parse_header(line: ~str) -> Option<~str> {
    if str::starts_with(line, ~"# ") {
        Some(str::slice(line, 2u, str::len(line)))
    } else {
        None
    }
}

#[test]
fn should_create_section_headers() {
    let doc = test::mk_doc(
        ~"#[doc = \"\
         # Header\n\
         Body\"]\
         mod a {
             #[legacy_exports]; }");
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
         mod a {
             #[legacy_exports]; }");
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
         mod a {
             #[legacy_exports]; }");
    assert vec::is_empty(doc.cratemod().mods()[0].item.sections);
}

#[test]
fn should_remove_section_text_from_main_desc() {
    let doc = test::mk_doc(
        ~"#[doc = \"\
         Description\n\n\
         # Header\n\
         Body\"]\
         mod a {
             #[legacy_exports]; }");
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
         mod a {
             #[legacy_exports]; }");
    assert doc.cratemod().mods()[0].desc() == None;
}

#[test]
fn should_sectionalize_trait_methods() {
    let doc = test::mk_doc(
        ~"trait i {
         #[doc = \"\
         # Header\n\
         Body\"]\
         fn a(); }");
    assert doc.cratemod().traits()[0].methods[0].sections.len() == 1u;
}

#[test]
fn should_sectionalize_impl_methods() {
    let doc = test::mk_doc(
        ~"impl bool {
         #[doc = \"\
         # Header\n\
         Body\"]\
         fn a() { } }");
    assert doc.cratemod().impls()[0].methods[0].sections.len() == 1u;
}

#[cfg(test)]
mod test {
    #[legacy_exports];
    fn mk_doc(source: ~str) -> doc::Doc {
        do astsrv::from_str(source) |srv| {
            let doc = extract::from_srv(srv, ~"");
            let doc = attr_pass::mk_pass().f(srv, doc);
            run(srv, doc)
        }
    }
}
