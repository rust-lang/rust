// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Generate markdown from a document tree

use core::prelude::*;

use astsrv;
use doc::ItemUtils;
use doc;
use markdown_pass;
use markdown_writer::Writer;
use markdown_writer::WriterUtils;
use markdown_writer::WriterFactory;
use pass::Pass;
use sort_pass;

use core::cell::Cell;
use syntax;

pub fn mk_pass(writer_factory: WriterFactory) -> Pass {
    let writer_factory = Cell(writer_factory);
    Pass {
        name: ~"markdown",
        f: |srv, doc| run(srv, doc, writer_factory.take())
    }
}

fn run(
    srv: astsrv::Srv,
    doc: doc::Doc,
    writer_factory: WriterFactory
) -> doc::Doc {

    fn mods_last(item1: &doc::ItemTag, item2: &doc::ItemTag) -> bool {
        fn is_mod(item: &doc::ItemTag) -> bool {
            match *item {
                doc::ModTag(_) => true,
                _ => false
            }
        }

        let lteq = !is_mod(item1) || is_mod(item2);
        lteq
    }

    // Sort the items so mods come last. All mods will be
    // output at the same header level so sorting mods last
    // makes the headers come out nested correctly.
    let sorted_doc = (sort_pass::mk_pass(
        ~"mods last", mods_last
    ).f)(srv, copy doc);

    write_markdown(sorted_doc, writer_factory);

    return doc;
}

struct Ctxt {
    w: Writer
}

pub fn write_markdown(
    doc: doc::Doc,
    writer_factory: WriterFactory
) {
    // There is easy parallelism to be had here, but
    // we don't want to spawn too many pandoc processes.
    // (See #2484, which is closed.)
    do doc.pages.map |page| {
        let ctxt = Ctxt {
            w: writer_factory(copy *page)
        };
        write_page(&ctxt, page)
    };
}

fn write_page(ctxt: &Ctxt, page: &doc::Page) {
    write_title(ctxt, copy *page);
    match copy *page {
        doc::CratePage(doc) => {
            write_crate(ctxt, doc);
        }
        doc::ItemPage(doc) => {
            // We don't write a header for item's pages because their
            // header in the html output is created by the page title
            write_item_no_header(ctxt, doc);
        }
    }
    ctxt.w.put_done();
}

fn write_title(ctxt: &Ctxt, page: doc::Page) {
    ctxt.w.put_line(fmt!("%% %s", make_title(page)));
    ctxt.w.put_line(~"");
}

fn make_title(page: doc::Page) -> ~str {
    let item = match page {
        doc::CratePage(CrateDoc) => {
            doc::ModTag(copy CrateDoc.topmod)
        }
        doc::ItemPage(ItemTag) => {
            ItemTag
        }
    };
    let title = markdown_pass::header_text(item);
    let title = str::replace(title, "`", "");
    return title;
}

enum Hlvl {
    H1 = 1,
    H2 = 2,
    H3 = 3,
    H4 = 4
}

fn write_header(ctxt: &Ctxt, lvl: Hlvl, doc: doc::ItemTag) {
    let text = header_text(doc);
    write_header_(ctxt, lvl, text);
}

fn write_header_(ctxt: &Ctxt, lvl: Hlvl, title: ~str) {
    let hashes = str::from_chars(vec::from_elem(lvl as uint, '#'));
    ctxt.w.put_line(fmt!("%s %s", hashes, title));
    ctxt.w.put_line(~"");
}

pub fn header_kind(doc: doc::ItemTag) -> ~str {
    match doc {
        doc::ModTag(_) => {
            if doc.id() == syntax::ast::crate_node_id {
                ~"Crate"
            } else {
                ~"Module"
            }
        }
        doc::NmodTag(_) => {
            ~"Foreign module"
        }
        doc::FnTag(_) => {
            ~"Function"
        }
        doc::ConstTag(_) => {
            ~"Const"
        }
        doc::EnumTag(_) => {
            ~"Enum"
        }
        doc::TraitTag(_) => {
            ~"Trait"
        }
        doc::ImplTag(_) => {
            ~"Implementation"
        }
        doc::TyTag(_) => {
            ~"Type"
        }
        doc::StructTag(_) => {
            ~"Struct"
        }
    }
}

pub fn header_name(doc: doc::ItemTag) -> ~str {
    let fullpath = str::connect(doc.path() + [doc.name()], "::");
    match &doc {
        &doc::ModTag(_) if doc.id() != syntax::ast::crate_node_id => {
            fullpath
        }
        &doc::NmodTag(_) => {
            fullpath
        }
        &doc::ImplTag(ref doc) => {
            assert!(doc.self_ty.is_some());
            let bounds = if (&doc.bounds_str).is_some() {
                fmt!(" where %s", (&doc.bounds_str).get())
            } else {
                ~""
            };
            let self_ty = (&doc.self_ty).get();
            let mut trait_part = ~"";
            for doc.trait_types.eachi |i, trait_type| {
                if i == 0 {
                    trait_part += " of ";
                } else {
                    trait_part += ", ";
                }
                trait_part += *trait_type;
            }
            fmt!("%s for %s%s", trait_part, self_ty, bounds)
        }
        _ => {
            doc.name()
        }
    }
}

pub fn header_text(doc: doc::ItemTag) -> ~str {
    match &doc {
        &doc::ImplTag(ref ImplDoc) => {
            let header_kind = header_kind(copy doc);
            let bounds = if (&ImplDoc.bounds_str).is_some() {
                fmt!(" where `%s`", (&ImplDoc.bounds_str).get())
            } else {
                ~""
            };
            let desc = if ImplDoc.trait_types.is_empty() {
                fmt!("for `%s`%s", (&ImplDoc.self_ty).get(), bounds)
            } else {
                fmt!("of `%s` for `%s`%s",
                     ImplDoc.trait_types[0],
                     (&ImplDoc.self_ty).get(),
                     bounds)
            };
            return fmt!("%s %s", header_kind, desc);
        }
        _ => {}
    }

    header_text_(header_kind(copy doc),
                 header_name(doc))
}

fn header_text_(kind: &str, name: &str) -> ~str {
    fmt!("%s `%s`", kind, name)
}

fn write_crate(
    ctxt: &Ctxt,
    doc: doc::CrateDoc
) {
    write_top_module(ctxt, copy doc.topmod);
}

fn write_top_module(
    ctxt: &Ctxt,
    ModDoc: doc::ModDoc
) {
    write_mod_contents(ctxt, ModDoc);
}

fn write_mod(
    ctxt: &Ctxt,
    ModDoc: doc::ModDoc
) {
    write_mod_contents(ctxt, ModDoc);
}

fn write_common(
    ctxt: &Ctxt,
    desc: Option<~str>,
    sections: &[doc::Section]
) {
    write_desc(ctxt, desc);
    write_sections(ctxt, sections);
}

fn write_desc(
    ctxt: &Ctxt,
    desc: Option<~str>
) {
    match desc {
        Some(desc) => {
            ctxt.w.put_line(desc);
            ctxt.w.put_line(~"");
        }
        None => ()
    }
}

fn write_sections(ctxt: &Ctxt, sections: &[doc::Section]) {
    for sections.each |section| {
        write_section(ctxt, copy *section);
    }
}

fn write_section(ctxt: &Ctxt, section: doc::Section) {
    write_header_(ctxt, H4, copy section.header);
    ctxt.w.put_line(copy section.body);
    ctxt.w.put_line(~"");
}

fn write_mod_contents(
    ctxt: &Ctxt,
    doc: doc::ModDoc
) {
    write_common(ctxt, doc.desc(), doc.sections());
    if doc.index.is_some() {
        write_index(ctxt, (&doc.index).get());
    }

    for doc.items.each |itemTag| {
        write_item(ctxt, copy *itemTag);
    }
}

fn write_item(ctxt: &Ctxt, doc: doc::ItemTag) {
    write_item_(ctxt, doc, true);
}

fn write_item_no_header(ctxt: &Ctxt, doc: doc::ItemTag) {
    write_item_(ctxt, doc, false);
}

fn write_item_(ctxt: &Ctxt, doc: doc::ItemTag, write_header: bool) {
    if write_header {
        write_item_header(ctxt, copy doc);
    }

    match doc {
        doc::ModTag(ModDoc) => write_mod(ctxt, ModDoc),
        doc::NmodTag(nModDoc) => write_nmod(ctxt, nModDoc),
        doc::FnTag(FnDoc) => write_fn(ctxt, FnDoc),
        doc::ConstTag(ConstDoc) => write_const(ctxt, ConstDoc),
        doc::EnumTag(EnumDoc) => write_enum(ctxt, EnumDoc),
        doc::TraitTag(TraitDoc) => write_trait(ctxt, TraitDoc),
        doc::ImplTag(ImplDoc) => write_impl(ctxt, ImplDoc),
        doc::TyTag(TyDoc) => write_type(ctxt, TyDoc),
        doc::StructTag(StructDoc) => put_struct(ctxt, StructDoc),
    }
}

fn write_item_header(ctxt: &Ctxt, doc: doc::ItemTag) {
    write_header(ctxt, item_header_lvl(&doc), doc);
}

fn item_header_lvl(doc: &doc::ItemTag) -> Hlvl {
    match doc {
        &doc::ModTag(_) | &doc::NmodTag(_) => H1,
        _ => H2
    }
}

fn write_index(ctxt: &Ctxt, index: doc::Index) {
    if vec::is_empty(index.entries) {
        return;
    }

    ctxt.w.put_line(~"<div class='index'>");
    ctxt.w.put_line(~"");

    for index.entries.each |entry| {
        let header = header_text_(entry.kind, entry.name);
        let id = copy entry.link;
        if entry.brief.is_some() {
            ctxt.w.put_line(fmt!("* [%s](%s) - %s",
                                 header, id, (&entry.brief).get()));
        } else {
            ctxt.w.put_line(fmt!("* [%s](%s)", header, id));
        }
    }
    ctxt.w.put_line(~"");
    ctxt.w.put_line(~"</div>");
    ctxt.w.put_line(~"");
}

fn write_nmod(ctxt: &Ctxt, doc: doc::NmodDoc) {
    write_common(ctxt, doc.desc(), doc.sections());
    if doc.index.is_some() {
        write_index(ctxt, (&doc.index).get());
    }

    for doc.fns.each |FnDoc| {
        write_item_header(ctxt, doc::FnTag(copy *FnDoc));
        write_fn(ctxt, copy *FnDoc);
    }
}

fn write_fn(
    ctxt: &Ctxt,
    doc: doc::FnDoc
) {
    write_fnlike(
        ctxt,
        copy doc.sig,
        doc.desc(),
        doc.sections()
    );
}

fn write_fnlike(
    ctxt: &Ctxt,
    sig: Option<~str>,
    desc: Option<~str>,
    sections: &[doc::Section]
) {
    write_sig(ctxt, sig);
    write_common(ctxt, desc, sections);
}

fn write_sig(ctxt: &Ctxt, sig: Option<~str>) {
    match sig {
        Some(sig) => {
            ctxt.w.put_line(code_block_indent(sig));
            ctxt.w.put_line(~"");
        }
        None => fail!("unimplemented")
    }
}

fn code_block_indent(s: ~str) -> ~str {
    let mut indented = ~[];
    for str::each_line_any(s) |line| {
        indented.push(fmt!("    %s", line));
    }
    str::connect(indented, "\n")
}

fn write_const(
    ctxt: &Ctxt,
    doc: doc::ConstDoc
) {
    write_sig(ctxt, copy doc.sig);
    write_common(ctxt, doc.desc(), doc.sections());
}

fn write_enum(
    ctxt: &Ctxt,
    doc: doc::EnumDoc
) {
    write_common(ctxt, doc.desc(), doc.sections());
    write_variants(ctxt, doc.variants);
}

fn write_variants(
    ctxt: &Ctxt,
    docs: &[doc::VariantDoc]
) {
    if vec::is_empty(docs) {
        return;
    }

    write_header_(ctxt, H4, ~"Variants");

    for docs.each |variant| {
        write_variant(ctxt, copy *variant);
    }

    ctxt.w.put_line(~"");
}

fn write_variant(ctxt: &Ctxt, doc: doc::VariantDoc) {
    assert!(doc.sig.is_some());
    let sig = (&doc.sig).get();

    // space out list items so they all end up within paragraph elements
    ctxt.w.put_line(~"");

    match copy doc.desc {
        Some(desc) => {
            ctxt.w.put_line(list_item_indent(fmt!("* `%s` - %s", sig, desc)));
        }
        None => {
            ctxt.w.put_line(fmt!("* `%s`", sig));
        }
    }
}

fn list_item_indent(item: &str) -> ~str {
    let mut indented = ~[];
    for str::each_line_any(item) |line| {
        indented.push(line);
    }

    // separate markdown elements within `*` lists must be indented by four
    // spaces, or they will escape the list context. indenting everything
    // seems fine though.
    str::connect_slices(indented, "\n    ")
}

fn write_trait(ctxt: &Ctxt, doc: doc::TraitDoc) {
    write_common(ctxt, doc.desc(), doc.sections());
    write_methods(ctxt, doc.methods);
}

fn write_methods(ctxt: &Ctxt, docs: &[doc::MethodDoc]) {
    for docs.each |doc| {
        write_method(ctxt, copy *doc);
    }
}

fn write_method(ctxt: &Ctxt, doc: doc::MethodDoc) {
    write_header_(ctxt, H3, header_text_("Method", doc.name));
    write_fnlike(
        ctxt,
        copy doc.sig,
        copy doc.desc,
        doc.sections
    );
}

fn write_impl(ctxt: &Ctxt, doc: doc::ImplDoc) {
    write_common(ctxt, doc.desc(), doc.sections());
    write_methods(ctxt, doc.methods);
}

fn write_type(
    ctxt: &Ctxt,
    doc: doc::TyDoc
) {
    write_sig(ctxt, copy doc.sig);
    write_common(ctxt, doc.desc(), doc.sections());
}

fn put_struct(
    ctxt: &Ctxt,
    doc: doc::StructDoc
) {
    write_sig(ctxt, copy doc.sig);
    write_common(ctxt, doc.desc(), doc.sections());
}

#[cfg(test)]
mod test {
    use core::prelude::*;

    use astsrv;
    use attr_pass;
    use config;
    use desc_to_brief_pass;
    use doc;
    use extract;
    use markdown_index_pass;
    use markdown_pass::{mk_pass, write_markdown};
    use markdown_writer;
    use path_pass;
    use page_pass;
    use sectionalize_pass;
    use trim_pass;
    use tystr_pass;
    use unindent_pass;

    fn render(source: ~str) -> ~str {
        let (srv, doc) = create_doc_srv(source);
        let markdown = write_markdown_str_srv(srv, doc);
        debug!("markdown: %s", markdown);
        markdown
    }

    fn create_doc_srv(source: ~str) -> (astsrv::Srv, doc::Doc) {
        do astsrv::from_str(source) |srv| {

            let config = config::Config {
                output_style: config::DocPerCrate,
                .. config::default_config(&Path("whatever"))
            };

            let doc = extract::from_srv(srv.clone(), ~"");
            debug!("doc (extract): %?", doc);
            let doc = (tystr_pass::mk_pass().f)(srv.clone(), doc);
            debug!("doc (tystr): %?", doc);
            let doc = (path_pass::mk_pass().f)(srv.clone(), doc);
            debug!("doc (path): %?", doc);
            let doc = (attr_pass::mk_pass().f)(srv.clone(), doc);
            debug!("doc (attr): %?", doc);
            let doc = (desc_to_brief_pass::mk_pass().f)(srv.clone(), doc);
            debug!("doc (desc_to_brief): %?", doc);
            let doc = (unindent_pass::mk_pass().f)(srv.clone(), doc);
            debug!("doc (unindent): %?", doc);
            let doc = (sectionalize_pass::mk_pass().f)(srv.clone(), doc);
            debug!("doc (trim): %?", doc);
            let doc = (trim_pass::mk_pass().f)(srv.clone(), doc);
            debug!("doc (sectionalize): %?", doc);
            let doc = (markdown_index_pass::mk_pass(config).f)(
                srv.clone(), doc);
            debug!("doc (index): %?", doc);
            (srv.clone(), doc)
        }
    }

    fn create_doc(source: ~str) -> doc::Doc {
        let (_, doc) = create_doc_srv(source);
        doc
    }

    fn write_markdown_str(
        doc: doc::Doc
    ) -> ~str {
        let (writer_factory, po) = markdown_writer::future_writer_factory();
        write_markdown(doc, writer_factory);
        return po.recv().second();
    }

    fn write_markdown_str_srv(
        srv: astsrv::Srv,
        doc: doc::Doc
    ) -> ~str {
        let (writer_factory, po) = markdown_writer::future_writer_factory();
        let pass = mk_pass(writer_factory);
        (pass.f)(srv, doc);
        return po.recv().second();
    }

    #[test]
    fn write_markdown_should_write_mod_headers() {
        let markdown = render(~"mod moo { }");
        assert!(str::contains(markdown, "# Module `moo`"));
    }

    #[test]
    fn should_leave_blank_line_after_header() {
        let markdown = render(~"mod morp { }");
        assert!(str::contains(markdown, "Module `morp`\n\n"));
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
        let markdown = render(
            ~"mod a { }\
              fn b() { }\
              mod c {
}\
              fn d() { }"
        );

        let idx_a = str::find_str(markdown, "# Module `a`").get();
        let idx_b = str::find_str(markdown, "## Function `b`").get();
        let idx_c = str::find_str(markdown, "# Module `c`").get();
        let idx_d = str::find_str(markdown, "## Function `d`").get();

        assert!(idx_b < idx_d);
        assert!(idx_d < idx_a);
        assert!(idx_a < idx_c);
    }

    #[test]
    fn should_request_new_writer_for_each_page() {
        // This port will send us a (page, str) pair for every writer
        // that was created
        let (writer_factory, po) = markdown_writer::future_writer_factory();
        let (srv, doc) = create_doc_srv(~"mod a { }");
        // Split the document up into pages
        let doc = (page_pass::mk_pass(config::DocPerMod).f)(srv, doc);
        write_markdown(doc, writer_factory);
        // We expect two pages to have been written
        for 2.times {
            po.recv();
        }
    }

    #[test]
    fn should_write_title_for_each_page() {
        let (writer_factory, po) = markdown_writer::future_writer_factory();
        let (srv, doc) = create_doc_srv(
            ~"#[link(name = \"core\")]; mod a { }");
        let doc = (page_pass::mk_pass(config::DocPerMod).f)(srv, doc);
        write_markdown(doc, writer_factory);
        for 2.times {
            let (page, markdown) = po.recv();
            match page {
                doc::CratePage(_) => {
                    assert!(str::contains(markdown, "% Crate core"));
                }
                doc::ItemPage(_) => {
                    assert!(str::contains(markdown, "% Module a"));
                }
            }
        }
    }

    #[test]
    fn should_write_full_path_to_mod() {
        let markdown = render(~"mod a { mod b { mod c { } } }");
        assert!(str::contains(markdown, "# Module `a::b::c`"));
    }

    #[test]
    fn should_write_sections() {
        let markdown = render(
            ~"#[doc = \"\
              # Header\n\
              Body\"]\
              mod a {
}");
        assert!(str::contains(markdown, "#### Header\n\nBody\n\n"));
    }

    #[test]
    fn should_write_crate_description() {
        let markdown = render(~"#[doc = \"this is the crate\"];");
        assert!(str::contains(markdown, "this is the crate"));
    }


    #[test]
    fn should_write_index() {
        let markdown = render(~"mod a { } mod b { }");
        assert!(str::contains(
            markdown,
            "\n\n* [Module `a`](#module-a)\n\
             * [Module `b`](#module-b)\n\n"
        ));
    }

    #[test]
    fn should_write_index_brief() {
        let markdown = render(~"#[doc = \"test\"] mod a { }");
        assert!(str::contains(markdown, "(#module-a) - test\n"));
    }

    #[test]
    fn should_not_write_index_if_no_entries() {
        let markdown = render(~"");
        assert!(!str::contains(markdown, "\n\n\n"));
    }

    #[test]
    fn should_write_index_for_foreign_mods() {
        let markdown = render(~"extern { fn a(); }");
        assert!(str::contains(
            markdown,
            "\n\n* [Function `a`](#function-a)\n\n"
        ));
    }

    #[test]
    fn should_write_foreign_fns() {
        let markdown = render(
            ~"extern { #[doc = \"test\"] fn a(); }");
        assert!(str::contains(markdown, "test"));
    }

    #[test]
    fn should_write_foreign_fn_headers() {
        let markdown = render(
            ~"extern { #[doc = \"test\"] fn a(); }");
        assert!(str::contains(markdown, "## Function `a`"));
    }

    #[test]
    fn write_markdown_should_write_function_header() {
        let markdown = render(~"fn func() { }");
        assert!(str::contains(markdown, "## Function `func`"));
    }

    #[test]
    fn should_write_the_function_signature() {
        let markdown = render(~"#[doc = \"f\"] fn a() { }");
        assert!(str::contains(markdown, "\n    fn a()\n"));
    }

    #[test]
    fn should_insert_blank_line_after_fn_signature() {
        let markdown = render(~"#[doc = \"f\"] fn a() { }");
        assert!(str::contains(markdown, "fn a()\n\n"));
    }

    #[test]
    fn should_correctly_indent_fn_signature() {
        let doc = create_doc(~"fn a() { }");
        let doc = doc::Doc{
            pages: ~[
                doc::CratePage(doc::CrateDoc{
                    topmod: doc::ModDoc{
                        items: ~[doc::FnTag(doc::SimpleItemDoc{
                            sig: Some(~"line 1\nline 2"),
                            .. copy doc.cratemod().fns()[0]
                        })],
                        .. doc.cratemod()
                    },
                    .. doc.CrateDoc()
                })
            ]
        };
        let markdown = write_markdown_str(doc);
        assert!(str::contains(markdown, "    line 1\n    line 2"));
    }

    #[test]
    fn should_leave_blank_line_between_fn_header_and_sig() {
        let markdown = render(~"fn a() { }");
        assert!(str::contains(markdown, "Function `a`\n\n    fn a()"));
    }

    #[test]
    fn should_write_const_header() {
        let markdown = render(~"static a: bool = true;");
        assert!(str::contains(markdown, "## Const `a`\n\n"));
    }

    #[test]
    fn should_write_const_description() {
        let markdown = render(
            ~"#[doc = \"b\"]\
              static a: bool = true;");
        assert!(str::contains(markdown, "\n\nb\n\n"));
    }

    #[test]
    fn should_write_enum_header() {
        let markdown = render(~"enum a { b }");
        assert!(str::contains(markdown, "## Enum `a`\n\n"));
    }

    #[test]
    fn should_write_enum_description() {
        let markdown = render(~"#[doc = \"b\"] enum a { b }");
        assert!(str::contains(markdown, "\n\nb\n\n"));
    }

    #[test]
    fn should_write_variant_list() {
        let markdown = render(
            ~"enum a { \
              #[doc = \"test\"] b, \
              #[doc = \"test\"] c }");
        assert!(str::contains(
            markdown,
            "\n\n#### Variants\n\
             \n\
             \n* `b` - test\
             \n\
             \n* `c` - test\n\n"));
    }

    #[test]
    fn should_write_variant_list_without_descs() {
        let markdown = render(~"enum a { b, c }");
        assert!(str::contains(
            markdown,
            "\n\n#### Variants\n\
             \n\
             \n* `b`\
             \n\
             \n* `c`\n\n"));
    }

    #[test]
    fn should_write_variant_list_with_indent() {
        let markdown = render(
            ~"enum a { #[doc = \"line 1\\n\\nline 2\"] b, c }");
        assert!(str::contains(
            markdown,
            "\n\n#### Variants\n\
             \n\
             \n* `b` - line 1\
             \n    \
             \n    line 2\
             \n\
             \n* `c`\n\n"));
    }

    #[test]
    fn should_write_variant_list_with_signatures() {
        let markdown = render(~"enum a { b(int), #[doc = \"a\"] c(int) }");
        assert!(str::contains(
            markdown,
            "\n\n#### Variants\n\
             \n\
             \n* `b(int)`\
             \n\
             \n* `c(int)` - a\n\n"));
    }

    #[test]
    fn should_write_trait_header() {
        let markdown = render(~"trait i { fn a(); }");
        assert!(str::contains(markdown, "## Trait `i`"));
    }

    #[test]
    fn should_write_trait_desc() {
        let markdown = render(~"#[doc = \"desc\"] trait i { fn a(); }");
        assert!(str::contains(markdown, "desc"));
    }

    #[test]
    fn should_write_trait_method_header() {
        let markdown = render(~"trait i { fn a(); }");
        assert!(str::contains(markdown, "### Method `a`"));
    }

    #[test]
    fn should_write_trait_method_signature() {
        let markdown = render(~"trait i { fn a(&self); }");
        assert!(str::contains(markdown, "\n    fn a(&self)"));
    }

    #[test]
    fn should_write_impl_header() {
        let markdown = render(~"impl int { fn a() { } }");
        assert!(str::contains(markdown, "## Implementation for `int`"));
    }

    #[test]
    fn should_write_impl_header_with_bounds() {
        let markdown = render(~"impl <T> int<T> { }");
        assert!(str::contains(markdown, "## Implementation for `int<T>` where `<T>`"));
    }

    #[test]
    fn should_write_impl_header_with_trait() {
        let markdown = render(~"impl j for int { fn a() { } }");
        assert!(str::contains(markdown,
                              "## Implementation of `j` for `int`"));
    }

    #[test]
    fn should_write_impl_desc() {
        let markdown = render(
            ~"#[doc = \"desc\"] impl int { fn a() { } }");
        assert!(str::contains(markdown, "desc"));
    }

    #[test]
    fn should_write_impl_method_header() {
        let markdown = render(
            ~"impl int { fn a() { } }");
        assert!(str::contains(markdown, "### Method `a`"));
    }

    #[test]
    fn should_write_impl_method_signature() {
        let markdown = render(
            ~"impl int { fn a(&mut self) { } }");
        assert!(str::contains(markdown, "\n    fn a(&mut self)"));
    }

    #[test]
    fn should_write_type_header() {
        let markdown = render(~"type t = int;");
        assert!(str::contains(markdown, "## Type `t`"));
    }

    #[test]
    fn should_write_type_desc() {
        let markdown = render(
            ~"#[doc = \"desc\"] type t = int;");
        assert!(str::contains(markdown, "\n\ndesc\n\n"));
    }

    #[test]
    fn should_write_type_signature() {
        let markdown = render(~"type t = int;");
        assert!(str::contains(markdown, "\n\n    type t = int\n\n"));
    }

    #[test]
    fn should_put_struct_header() {
        let markdown = render(~"struct S { field: () }");
        assert!(str::contains(markdown, "## Struct `S`\n\n"));
    }
}
