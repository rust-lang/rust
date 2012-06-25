#[doc = "Generate markdown from a document tree"];

import markdown_writer::writer;
import markdown_writer::writer_util;
import markdown_writer::writer_factory;

export mk_pass;
export header_kind, header_name, header_text;

fn mk_pass(+writer_factory: writer_factory) -> pass {
    let f = fn~(srv: astsrv::srv, doc: doc::doc) -> doc::doc {
        run(srv, doc, copy writer_factory)
    };

    {
        name: "markdown",
        f: f
    }
}

fn run(
    srv: astsrv::srv,
    doc: doc::doc,
    +writer_factory: writer_factory
) -> doc::doc {

    fn mods_last(item1: doc::itemtag, item2: doc::itemtag) -> bool {
        fn is_mod(item: doc::itemtag) -> bool {
            alt item {
              doc::modtag(_) { true }
              _ { false }
            }
        }

        let lteq = !is_mod(item1) || is_mod(item2);
        lteq
    }

    // Sort the items so mods come last. All mods will be
    // output at the same header level so sorting mods last
    // makes the headers come out nested correctly.
    let sorted_doc = sort_pass::mk_pass(
        "mods last", mods_last
    ).f(srv, doc);

    write_markdown(sorted_doc, writer_factory);

    ret doc;
}

#[test]
fn should_write_modules_last() {
    /*
    Because the markdown pass writes all modules at the same level of
    indentation (it doesn't 'nest' them), we need to make sure that we
    write all of the modules contained in each module after all other
    types of items, or else the header nesting will end up wrong, with
    modules appearing to contain items that they do not.
    */
    let markdown = test::render(
        "mod a { }\
         fn b() { }\
         mod c { }\
         fn d() { }"
    );

    let idx_a = option::get(str::find_str(markdown, "# Module `a`"));
    let idx_b = option::get(str::find_str(markdown, "## Function `b`"));
    let idx_c = option::get(str::find_str(markdown, "# Module `c`"));
    let idx_d = option::get(str::find_str(markdown, "## Function `d`"));

    assert idx_b < idx_d;
    assert idx_d < idx_a;
    assert idx_a < idx_c;
}

type ctxt = {
    w: writer
};

fn write_markdown(
    doc: doc::doc,
    +writer_factory: writer_factory
) {
    par::anymap(doc.pages) {|page|
        let ctxt = {
            w: writer_factory(page)
        };
        write_page(ctxt, page)
    };
}

fn write_page(ctxt: ctxt, page: doc::page) {
    write_title(ctxt, page);
    alt page {
      doc::cratepage(doc) {
        write_crate(ctxt, doc);
      }
      doc::itempage(doc) {
        // We don't write a header for item's pages because their
        // header in the html output is created by the page title
        write_item_no_header(ctxt, doc);
      }
    }
    ctxt.w.write_done();
}

#[test]
fn should_request_new_writer_for_each_page() {
    // This port will send us a (page, str) pair for every writer
    // that was created
    let (writer_factory, po) = markdown_writer::future_writer_factory();
    let (srv, doc) = test::create_doc_srv("mod a { }");
    // Split the document up into pages
    let doc = page_pass::mk_pass(config::doc_per_mod).f(srv, doc);
    write_markdown(doc, writer_factory);
    // We expect two pages to have been written
    iter::repeat(2u) {||
        comm::recv(po);
    }
}

fn write_title(ctxt: ctxt, page: doc::page) {
    ctxt.w.write_line(#fmt("%% %s", make_title(page)));
    ctxt.w.write_line("");
}

fn make_title(page: doc::page) -> str {
    let item = alt page {
      doc::cratepage(cratedoc) {
        doc::modtag(cratedoc.topmod)
      }
      doc::itempage(itemtag) {
        itemtag
      }
    };
    let title = markdown_pass::header_text(item);
    let title = str::replace(title, "`", "");
    ret title;
}

#[test]
fn should_write_title_for_each_page() {
    let (writer_factory, po) = markdown_writer::future_writer_factory();
    let (srv, doc) = test::create_doc_srv(
        "#[link(name = \"core\")]; mod a { }");
    let doc = page_pass::mk_pass(config::doc_per_mod).f(srv, doc);
    write_markdown(doc, writer_factory);
    iter::repeat(2u) {||
        let (page, markdown) = comm::recv(po);
        alt page {
          doc::cratepage(_) {
            assert str::contains(markdown, "% Crate core");
          }
          doc::itempage(_) {
            assert str::contains(markdown, "% Module a");
          }
        }
    }
}

enum hlvl {
    h1 = 1,
    h2 = 2,
    h3 = 3,
    h4 = 4
}

fn write_header(ctxt: ctxt, lvl: hlvl, doc: doc::itemtag) {
    let text = header_text(doc);
    write_header_(ctxt, lvl, text);
}

fn write_header_(ctxt: ctxt, lvl: hlvl, title: str) {
    let hashes = str::from_chars(vec::from_elem(lvl as uint, '#'));
    ctxt.w.write_line(#fmt("%s %s", hashes, title));
    ctxt.w.write_line("");
}

fn header_kind(doc: doc::itemtag) -> str {
    alt doc {
      doc::modtag(_) {
        if doc.id() == syntax::ast::crate_node_id {
            "Crate"
        } else {
            "Module"
        }
      }
      doc::nmodtag(_) {
        "Native module"
      }
      doc::fntag(_) {
        "Function"
      }
      doc::consttag(_) {
        "Const"
      }
      doc::enumtag(_) {
        "Enum"
      }
      doc::ifacetag(_) {
        "Interface"
      }
      doc::impltag(doc) {
        "Implementation"
      }
      doc::tytag(_) {
        "Type"
      }
    }
}

fn header_name(doc: doc::itemtag) -> str {
    let fullpath = str::connect(doc.path() + [doc.name()], "::");
    alt doc {
      doc::modtag(_) if doc.id() != syntax::ast::crate_node_id {
        fullpath
      }
      doc::nmodtag(_) {
        fullpath
      }
      doc::impltag(doc) {
        assert option::is_some(doc.self_ty);
        let self_ty = option::get(doc.self_ty);
        alt doc.iface_ty {
          some(iface_ty) {
            #fmt("%s of %s for %s", doc.name(), iface_ty, self_ty)
          }
          none {
            #fmt("%s for %s", doc.name(), self_ty)
          }
        }
      }
      _ {
        doc.name()
      }
    }
}

fn header_text(doc: doc::itemtag) -> str {
    header_text_(header_kind(doc), header_name(doc))
}

fn header_text_(kind: str, name: str) -> str {
    #fmt("%s `%s`", kind, name)
}

fn write_crate(
    ctxt: ctxt,
    doc: doc::cratedoc
) {
    write_top_module(ctxt, doc.topmod);
}

fn write_top_module(
    ctxt: ctxt,
    moddoc: doc::moddoc
) {
    write_mod_contents(ctxt, moddoc);
}

fn write_mod(
    ctxt: ctxt,
    moddoc: doc::moddoc
) {
    write_mod_contents(ctxt, moddoc);
}

#[test]
fn should_write_full_path_to_mod() {
    let markdown = test::render("mod a { mod b { mod c { } } }");
    assert str::contains(markdown, "# Module `a::b::c`");
}

fn write_common(
    ctxt: ctxt,
    desc: option<str>,
    sections: [doc::section]
) {
    write_desc(ctxt, desc);
    write_sections(ctxt, sections);
}

fn write_desc(
    ctxt: ctxt,
    desc: option<str>
) {
    alt desc {
        some(desc) {
            ctxt.w.write_line(desc);
            ctxt.w.write_line("");
        }
        none { }
    }
}

fn write_sections(ctxt: ctxt, sections: [doc::section]) {
    vec::iter(sections) {|section|
        write_section(ctxt, section);
    }
}

fn write_section(ctxt: ctxt, section: doc::section) {
    write_header_(ctxt, h4, section.header);
    ctxt.w.write_line(section.body);
    ctxt.w.write_line("");
}

#[test]
fn should_write_sections() {
    let markdown = test::render(
        "#[doc = \"\
         # Header\n\
         Body\"]\
         mod a { }");
    assert str::contains(markdown, "#### Header\n\nBody\n\n");
}

fn write_mod_contents(
    ctxt: ctxt,
    doc: doc::moddoc
) {
    write_common(ctxt, doc.desc(), doc.sections());
    if option::is_some(doc.index) {
        write_index(ctxt, option::get(doc.index));
    }

    for doc.items.each {|itemtag|
        write_item(ctxt, itemtag);
    }
}

fn write_item(ctxt: ctxt, doc: doc::itemtag) {
    write_item_(ctxt, doc, true);
}

fn write_item_no_header(ctxt: ctxt, doc: doc::itemtag) {
    write_item_(ctxt, doc, false);
}

fn write_item_(ctxt: ctxt, doc: doc::itemtag, write_header: bool) {
    if write_header {
        write_item_header(ctxt, doc);
    }

    alt doc {
      doc::modtag(moddoc) { write_mod(ctxt, moddoc) }
      doc::nmodtag(nmoddoc) { write_nmod(ctxt, nmoddoc) }
      doc::fntag(fndoc) { write_fn(ctxt, fndoc) }
      doc::consttag(constdoc) { write_const(ctxt, constdoc) }
      doc::enumtag(enumdoc) { write_enum(ctxt, enumdoc) }
      doc::ifacetag(ifacedoc) { write_iface(ctxt, ifacedoc) }
      doc::impltag(impldoc) { write_impl(ctxt, impldoc) }
      doc::tytag(tydoc) { write_type(ctxt, tydoc) }
    }
}

fn write_item_header(ctxt: ctxt, doc: doc::itemtag) {
    write_header(ctxt, item_header_lvl(doc), doc);
}

fn item_header_lvl(doc: doc::itemtag) -> hlvl {
    alt doc {
      doc::modtag(_) | doc::nmodtag(_) { h1 }
      _ { h2 }
    }
}

#[test]
fn should_write_crate_description() {
    let markdown = test::render("#[doc = \"this is the crate\"];");
    assert str::contains(markdown, "this is the crate");
}

fn write_index(ctxt: ctxt, index: doc::index) {
    if vec::is_empty(index.entries) {
        ret;
    }

    for index.entries.each {|entry|
        let header = header_text_(entry.kind, entry.name);
        let id = entry.link;
        if option::is_some(entry.brief) {
            ctxt.w.write_line(#fmt("* [%s](%s) - %s",
                                   header, id, option::get(entry.brief)));
        } else {
            ctxt.w.write_line(#fmt("* [%s](%s)", header, id));
        }
    }
    ctxt.w.write_line("");
}

#[test]
fn should_write_index() {
    let markdown = test::render("mod a { } mod b { }");
    assert str::contains(
        markdown,
        "\n\n* [Module `a`](#module-a)\n\
         * [Module `b`](#module-b)\n\n"
    );
}

#[test]
fn should_write_index_brief() {
    let markdown = test::render("#[doc = \"test\"] mod a { }");
    assert str::contains(markdown, "(#module-a) - test\n");
}

#[test]
fn should_not_write_index_if_no_entries() {
    let markdown = test::render("");
    assert !str::contains(markdown, "\n\n\n");
}

#[test]
fn should_write_index_for_native_mods() {
    let markdown = test::render("native mod a { fn a(); }");
    assert str::contains(
        markdown,
        "\n\n* [Function `a`](#function-a)\n\n"
    );
}

fn write_nmod(ctxt: ctxt, doc: doc::nmoddoc) {
    write_common(ctxt, doc.desc(), doc.sections());
    if option::is_some(doc.index) {
        write_index(ctxt, option::get(doc.index));
    }

    for doc.fns.each {|fndoc|
        write_item_header(ctxt, doc::fntag(fndoc));
        write_fn(ctxt, fndoc);
    }
}

#[test]
fn should_write_native_mods() {
    let markdown = test::render("#[doc = \"test\"] native mod a { }");
    assert str::contains(markdown, "Native module `a`");
    assert str::contains(markdown, "test");
}

#[test]
fn should_write_native_fns() {
    let markdown = test::render("native mod a { #[doc = \"test\"] fn a(); }");
    assert str::contains(markdown, "test");
}

#[test]
fn should_write_native_fn_headers() {
    let markdown = test::render("native mod a { #[doc = \"test\"] fn a(); }");
    assert str::contains(markdown, "## Function `a`");
}

fn write_fn(
    ctxt: ctxt,
    doc: doc::fndoc
) {
    write_fnlike(
        ctxt,
        doc.sig,
        doc.desc(),
        doc.sections()
    );
}

fn write_fnlike(
    ctxt: ctxt,
    sig: option<str>,
    desc: option<str>,
    sections: [doc::section]
) {
    write_sig(ctxt, sig);
    write_common(ctxt, desc, sections);
}

fn write_sig(ctxt: ctxt, sig: option<str>) {
    alt sig {
      some(sig) {
        ctxt.w.write_line(code_block_indent(sig));
        ctxt.w.write_line("");
      }
      none { fail "unimplemented" }
    }
}

fn code_block_indent(s: str) -> str {
    let lines = str::lines_any(s);
    let indented = par::seqmap(lines, { |line| #fmt("    %s", line) });
    str::connect(indented, "\n")
}

#[test]
fn write_markdown_should_write_function_header() {
    let markdown = test::render("fn func() { }");
    assert str::contains(markdown, "## Function `func`");
}

#[test]
fn should_write_the_function_signature() {
    let markdown = test::render("#[doc = \"f\"] fn a() { }");
    assert str::contains(markdown, "\n    fn a()\n");
}

#[test]
fn should_insert_blank_line_after_fn_signature() {
    let markdown = test::render("#[doc = \"f\"] fn a() { }");
    assert str::contains(markdown, "fn a()\n\n");
}

#[test]
fn should_correctly_indent_fn_signature() {
    let doc = test::create_doc("fn a() { }");
    let doc = {
        pages: [
            doc::cratepage({
                topmod: {
                    items: [doc::fntag({
                        sig: some("line 1\nline 2")
                        with doc.cratemod().fns()[0]
                    })]
                    with doc.cratemod()
                }
                with doc.cratedoc()
            })
        ]
    };
    let markdown = test::write_markdown_str(doc);
    assert str::contains(markdown, "    line 1\n    line 2");
}

#[test]
fn should_leave_blank_line_between_fn_header_and_sig() {
    let markdown = test::render("fn a() { }");
    assert str::contains(markdown, "Function `a`\n\n    fn a()");
}

fn write_const(
    ctxt: ctxt,
    doc: doc::constdoc
) {
    write_sig(ctxt, doc.sig);
    write_common(ctxt, doc.desc(), doc.sections());
}

#[test]
fn should_write_const_header() {
    let markdown = test::render("const a: bool = true;");
    assert str::contains(markdown, "## Const `a`\n\n");
}

#[test]
fn should_write_const_description() {
    let markdown = test::render(
        "#[doc = \"b\"]\
         const a: bool = true;");
    assert str::contains(markdown, "\n\nb\n\n");
}

fn write_enum(
    ctxt: ctxt,
    doc: doc::enumdoc
) {
    write_common(ctxt, doc.desc(), doc.sections());
    write_variants(ctxt, doc.variants);
}

#[test]
fn should_write_enum_header() {
    let markdown = test::render("enum a { b }");
    assert str::contains(markdown, "## Enum `a`\n\n");
}

#[test]
fn should_write_enum_description() {
    let markdown = test::render(
        "#[doc = \"b\"] enum a { b }");
    assert str::contains(markdown, "\n\nb\n\n");
}

fn write_variants(
    ctxt: ctxt,
    docs: [doc::variantdoc]
) {
    if vec::is_empty(docs) {
        ret;
    }

    write_header_(ctxt, h4, "Variants");

    vec::iter(docs, {|variant| write_variant(ctxt, variant) });

    ctxt.w.write_line("");
}

fn write_variant(ctxt: ctxt, doc: doc::variantdoc) {
    assert option::is_some(doc.sig);
    let sig = option::get(doc.sig);
    alt doc.desc {
      some(desc) {
        ctxt.w.write_line(#fmt("* `%s` - %s", sig, desc));
      }
      none {
        ctxt.w.write_line(#fmt("* `%s`", sig));
      }
    }
}

#[test]
fn should_write_variant_list() {
    let markdown = test::render(
        "enum a { \
         #[doc = \"test\"] b, \
         #[doc = \"test\"] c }");
    assert str::contains(
        markdown,
        "\n\n#### Variants\n\
         \n* `b` - test\
         \n* `c` - test\n\n");
}

#[test]
fn should_write_variant_list_without_descs() {
    let markdown = test::render("enum a { b, c }");
    assert str::contains(
        markdown,
        "\n\n#### Variants\n\
         \n* `b`\
         \n* `c`\n\n");
}

#[test]
fn should_write_variant_list_with_signatures() {
    let markdown = test::render("enum a { b(int), #[doc = \"a\"] c(int) }");
    assert str::contains(
        markdown,
        "\n\n#### Variants\n\
         \n* `b(int)`\
         \n* `c(int)` - a\n\n");
}

fn write_iface(ctxt: ctxt, doc: doc::ifacedoc) {
    write_common(ctxt, doc.desc(), doc.sections());
    write_methods(ctxt, doc.methods);
}

fn write_methods(ctxt: ctxt, docs: [doc::methoddoc]) {
    vec::iter(docs) {|doc| write_method(ctxt, doc) }
}

fn write_method(ctxt: ctxt, doc: doc::methoddoc) {
    write_header_(ctxt, h3, header_text_("Method", doc.name));
    write_fnlike(
        ctxt,
        doc.sig,
        doc.desc,
        doc.sections
    );
}

#[test]
fn should_write_iface_header() {
    let markdown = test::render("iface i { fn a(); }");
    assert str::contains(markdown, "## Interface `i`");
}

#[test]
fn should_write_iface_desc() {
    let markdown = test::render(
        "#[doc = \"desc\"] iface i { fn a(); }");
    assert str::contains(markdown, "desc");
}

#[test]
fn should_write_iface_method_header() {
    let markdown = test::render(
        "iface i { fn a(); }");
    assert str::contains(markdown, "### Method `a`");
}

#[test]
fn should_write_iface_method_signature() {
    let markdown = test::render(
        "iface i { fn a(); }");
    assert str::contains(markdown, "\n    fn a()");
}

fn write_impl(ctxt: ctxt, doc: doc::impldoc) {
    write_common(ctxt, doc.desc(), doc.sections());
    write_methods(ctxt, doc.methods);
}

#[test]
fn should_write_impl_header() {
    let markdown = test::render("impl i for int { fn a() { } }");
    assert str::contains(markdown, "## Implementation `i for int`");
}

#[test]
fn should_write_impl_header_with_iface() {
    let markdown = test::render("impl i of j for int { fn a() { } }");
    assert str::contains(markdown, "## Implementation `i of j for int`");
}

#[test]
fn should_write_impl_desc() {
    let markdown = test::render(
        "#[doc = \"desc\"] impl i for int { fn a() { } }");
    assert str::contains(markdown, "desc");
}

#[test]
fn should_write_impl_method_header() {
    let markdown = test::render(
        "impl i for int { fn a() { } }");
    assert str::contains(markdown, "### Method `a`");
}

#[test]
fn should_write_impl_method_signature() {
    let markdown = test::render(
        "impl i for int { fn a() { } }");
    assert str::contains(markdown, "\n    fn a()");
}

fn write_type(
    ctxt: ctxt,
    doc: doc::tydoc
) {
    write_sig(ctxt, doc.sig);
    write_common(ctxt, doc.desc(), doc.sections());
}

#[test]
fn should_write_type_header() {
    let markdown = test::render("type t = int;");
    assert str::contains(markdown, "## Type `t`");
}

#[test]
fn should_write_type_desc() {
    let markdown = test::render(
        "#[doc = \"desc\"] type t = int;");
    assert str::contains(markdown, "\n\ndesc\n\n");
}

#[test]
fn should_write_type_signature() {
    let markdown = test::render("type t = int;");
    assert str::contains(markdown, "\n\n    type t = int\n\n");
}

#[cfg(test)]
mod test {
    fn render(source: str) -> str {
        let (srv, doc) = create_doc_srv(source);
        let markdown = write_markdown_str_srv(srv, doc);
        #debug("markdown: %s", markdown);
        markdown
    }

    fn create_doc_srv(source: str) -> (astsrv::srv, doc::doc) {
        astsrv::from_str(source) {|srv|

            let config = {
                output_style: config::doc_per_crate
                with config::default_config("whatever")
            };

            let doc = extract::from_srv(srv, "");
            #debug("doc (extract): %?", doc);
            let doc = tystr_pass::mk_pass().f(srv, doc);
            #debug("doc (tystr): %?", doc);
            let doc = path_pass::mk_pass().f(srv, doc);
            #debug("doc (path): %?", doc);
            let doc = attr_pass::mk_pass().f(srv, doc);
            #debug("doc (attr): %?", doc);
            let doc = desc_to_brief_pass::mk_pass().f(srv, doc);
            #debug("doc (desc_to_brief): %?", doc);
            let doc = unindent_pass::mk_pass().f(srv, doc);
            #debug("doc (unindent): %?", doc);
            let doc = sectionalize_pass::mk_pass().f(srv, doc);
            #debug("doc (trim): %?", doc);
            let doc = trim_pass::mk_pass().f(srv, doc);
            #debug("doc (sectionalize): %?", doc);
            let doc = markdown_index_pass::mk_pass(config).f(srv, doc);
            #debug("doc (index): %?", doc);
            (srv, doc)
        }
    }

    fn create_doc(source: str) -> doc::doc {
        let (_, doc) = create_doc_srv(source);
        doc
    }

    fn write_markdown_str(
        doc: doc::doc
    ) -> str {
        let (writer_factory, po) = markdown_writer::future_writer_factory();
        write_markdown(doc, writer_factory);
        ret tuple::second(comm::recv(po));
    }

    fn write_markdown_str_srv(
        srv: astsrv::srv,
        doc: doc::doc
    ) -> str {
        let (writer_factory, po) = markdown_writer::future_writer_factory();
        let pass = mk_pass(writer_factory);
        pass.f(srv, doc);
        ret tuple::second(comm::recv(po));
    }

    #[test]
    fn write_markdown_should_write_mod_headers() {
        let markdown = render("mod moo { }");
        assert str::contains(markdown, "# Module `moo`");
    }

    #[test]
    fn should_leave_blank_line_after_header() {
        let markdown = render("mod morp { }");
        assert str::contains(markdown, "Module `morp`\n\n");
    }
}
