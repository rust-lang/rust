#[doc = "Generate markdown from a document tree"];

import markdown_writer::writer;
import markdown_writer::writer_util;
import markdown_writer::writer_factory;

export mk_pass;
export header_kind, header_name, header_text;

fn mk_pass(writer_factory: writer_factory) -> pass {
    let f = fn~(srv: astsrv::srv, doc: doc::doc) -> doc::doc {
        run(srv, doc, writer_factory)
    };

    {
        name: "markdown",
        f: f
    }
}

fn run(
    srv: astsrv::srv,
    doc: doc::doc,
    writer_factory: writer_factory
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
    writer_factory: writer_factory
) {
    par::anymap(doc.pages) {|page|
        let ctxt = {
            w: writer_factory(page)
        };
        write_page(ctxt, page)
    };
}

fn write_page(ctxt: ctxt, page: doc::page) {
    alt page {
      doc::cratepage(doc) {
        write_crate(ctxt, doc);
      }
      doc::itempage(doc) {
        write_item(ctxt, doc);
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

enum hlvl {
    h1 = 1,
    h2 = 2,
    h3 = 3
}

fn write_header(ctxt: ctxt, lvl: hlvl, doc: doc::itemtag) {
    let text = header_text(doc);
    write_header_(ctxt, lvl, text);
}

fn write_header_(ctxt: ctxt, lvl: hlvl, title: str) {
    let hashes = str::from_chars(vec::init_elt(lvl as uint, '#'));
    ctxt.w.write_line(#fmt("%s %s", hashes, title));
    ctxt.w.write_line("");
}

fn header_kind(doc: doc::itemtag) -> str {
    alt doc {
      doc::modtag(_) {
        if doc.id() == rustc::syntax::ast::crate_node_id {
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
      doc::restag(_) {
        "Resource"
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
      doc::modtag(_) if doc.id() != rustc::syntax::ast::crate_node_id {
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
    write_header(ctxt, h1, doc::modtag(doc.topmod));
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
    write_header(ctxt, h1, doc::modtag(moddoc));
    write_mod_contents(ctxt, moddoc);
}

#[test]
fn should_write_full_path_to_mod() {
    let markdown = test::render("mod a { mod b { mod c { } } }");
    assert str::contains(markdown, "# Module `a::b::c`");
}

fn write_mod_contents(
    ctxt: ctxt,
    doc: doc::moddoc
) {
    write_brief(ctxt, doc.brief());
    write_desc(ctxt, doc.desc());
    if option::is_some(doc.index) {
        write_index(ctxt, option::get(doc.index));
    }

    for itemtag in doc.items {
        write_item(ctxt, itemtag);
    }
}

fn write_item(ctxt: ctxt, doc: doc::itemtag) {
    alt doc {
      doc::modtag(moddoc) { write_mod(ctxt, moddoc) }
      doc::nmodtag(nmoddoc) { write_nmod(ctxt, nmoddoc) }
      doc::fntag(fndoc) { write_fn(ctxt, fndoc) }
      doc::consttag(constdoc) { write_const(ctxt, constdoc) }
      doc::enumtag(enumdoc) { write_enum(ctxt, enumdoc) }
      doc::restag(resdoc) { write_res(ctxt, resdoc) }
      doc::ifacetag(ifacedoc) { write_iface(ctxt, ifacedoc) }
      doc::impltag(impldoc) { write_impl(ctxt, impldoc) }
      doc::tytag(tydoc) { write_type(ctxt, tydoc) }
    }
}

#[test]
fn should_write_crate_brief_description() {
    let markdown = test::render("#[doc(brief = \"this is the crate\")];");
    assert str::contains(markdown, "this is the crate");
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

    for entry in index.entries {
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
    let markdown = test::render("#[doc(brief = \"test\")] mod a { }");
    assert str::contains(markdown, "(#module-a) - test\n");
}

#[test]
fn should_not_write_index_if_no_entries() {
    let markdown = test::render("");
    assert !str::contains(markdown, "\n\n\n");
}

fn write_nmod(ctxt: ctxt, doc: doc::nmoddoc) {
    write_header(ctxt, h1, doc::nmodtag(doc));

    write_brief(ctxt, doc.brief());
    write_desc(ctxt, doc.desc());

    for fndoc in doc.fns {
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

fn write_fn(
    ctxt: ctxt,
    doc: doc::fndoc
) {
    write_header(ctxt, h2, doc::fntag(doc));
    write_fnlike(
        ctxt,
        doc.sig,
        doc.brief(),
        doc.desc(),
        doc.args,
        doc.return,
        doc.failure
    );
}

fn write_fnlike(
    ctxt: ctxt,
    sig: option<str>,
    brief: option<str>,
    desc: option<str>,
    args: [doc::argdoc],
    return: doc::retdoc,
    failure: option<str>
) {
    write_sig(ctxt, sig);
    write_brief(ctxt, brief);
    write_desc(ctxt, desc);
    write_args(ctxt, args);
    write_return(ctxt, return);
    write_failure(ctxt, failure);
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
    let markdown = test::render("#[doc(brief = \"brief\")] fn a() { }");
    assert str::contains(markdown, "Function `a`\n\n    fn a()");
}

fn write_brief(
    ctxt: ctxt,
    brief: option<str>
) {
    alt brief {
      some(brief) {
        ctxt.w.write_line(brief);
        ctxt.w.write_line("");
      }
      none { }
    }
}

#[test]
fn should_leave_blank_line_after_brief() {
    let markdown = test::render("#[doc(brief = \"brief\")] fn a() { }");
    assert str::contains(markdown, "brief\n\n");
}

#[test]
fn should_leave_blank_line_between_brief_and_desc() {
    let markdown = test::render(
        "#[doc(brief = \"brief\", desc = \"desc\")] fn a() { }"
    );
    assert str::contains(markdown, "brief\n\ndesc");
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

fn write_args(
    ctxt: ctxt,
    args: [doc::argdoc]
) {
    if vec::is_not_empty(args) {
        ctxt.w.write_line("Arguments:");
        ctxt.w.write_line("");
        vec::iter(args) {|arg| write_arg(ctxt, arg) };
        ctxt.w.write_line("");
    }
}

fn write_arg(ctxt: ctxt, arg: doc::argdoc) {
    ctxt.w.write_str(#fmt(
        "* `%s`",
        arg.name
    ));
    alt arg.desc {
      some(desc) {
        ctxt.w.write_str(#fmt(" - %s", desc));
      }
      none { }
    }
    ctxt.w.write_line("");
}

#[test]
fn should_write_argument_list() {
    let source = "fn a(b: int, c: int) { }";
    let markdown = test::render(source);
    assert str::contains(
        markdown,
        "Arguments:\n\
         \n\
         * `b`\n\
         * `c`\n\
         \n"
    );
}

#[test]
fn should_not_write_arguments_if_none() {
    let source = "fn a() { } fn b() { }";
    let markdown = test::render(source);
    assert !str::contains(markdown, "Arguments");
}

#[test]
fn should_write_argument_description() {
    let source = "#[doc(args(a = \"milk\"))] fn f(a: bool) { }";
    let markdown = test::render(source);
    assert str::contains(markdown, "`a` - milk");
}

fn write_return(
    ctxt: ctxt,
    doc: doc::retdoc
) {
    alt doc.desc {
      some(d) {
        ctxt.w.write_line(#fmt("Return value: %s", d));
        ctxt.w.write_line("");
      }
      none { }
    }
}

#[test]
fn should_write_return_type_on_new_line() {
    let markdown = test::render(
        "#[doc(return = \"test\")] fn a() -> int { }");
    assert str::contains(markdown, "\nReturn value: test");
}

#[test]
fn should_write_blank_line_between_return_type_and_next_header() {
    let markdown = test::render(
        "#[doc(return = \"test\")] fn a() -> int { } \
         fn b() -> int { }"
    );
    assert str::contains(markdown, "Return value: test\n\n##");
}

#[test]
fn should_not_write_return_type_when_there_is_none() {
    let markdown = test::render("fn a() { }");
    assert !str::contains(markdown, "Return value");
}

#[test]
fn should_write_blank_line_after_return_description() {
    let markdown = test::render(
        "#[doc(return = \"blorp\")] fn a() -> int { }"
    );
    assert str::contains(markdown, "blorp\n\n");
}

fn write_failure(ctxt: ctxt, str: option<str>) {
    alt str {
      some(str) {
        ctxt.w.write_line(#fmt("Failure conditions: %s", str));
        ctxt.w.write_line("");
      }
      none { }
    }
}

#[test]
fn should_write_failure_conditions() {
    let markdown = test::render(
        "#[doc(failure = \"it's the fail\")] fn a () { }");
    assert str::contains(
        markdown,
        "\n\nFailure conditions: it's the fail\n\n");
}

fn write_const(
    ctxt: ctxt,
    doc: doc::constdoc
) {
    write_header(ctxt, h2, doc::consttag(doc));
    write_sig(ctxt, doc.ty);
    write_brief(ctxt, doc.brief());
    write_desc(ctxt, doc.desc());
}

#[test]
fn should_write_const_header() {
    let markdown = test::render("const a: bool = true;");
    assert str::contains(markdown, "## Const `a`\n\n");
}

#[test]
fn should_write_const_description() {
    let markdown = test::render(
        "#[doc(brief = \"a\", desc = \"b\")]\
         const a: bool = true;");
    assert str::contains(markdown, "\n\na\n\nb\n\n");
}

fn write_enum(
    ctxt: ctxt,
    doc: doc::enumdoc
) {
    write_header(ctxt, h2, doc::enumtag(doc));
    write_brief(ctxt, doc.brief());
    write_desc(ctxt, doc.desc());
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
        "#[doc(brief = \"a\", desc = \"b\")] enum a { b }");
    assert str::contains(markdown, "\n\na\n\nb\n\n");
}

fn write_variants(
    ctxt: ctxt,
    docs: [doc::variantdoc]
) {
    if vec::is_empty(docs) {
        ret;
    }

    ctxt.w.write_line("Variants:");
    ctxt.w.write_line("");

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
        "\n\nVariants:\n\
         \n* `b` - test\
         \n* `c` - test\n\n");
}

#[test]
fn should_write_variant_list_without_descs() {
    let markdown = test::render("enum a { b, c }");
    assert str::contains(
        markdown,
        "\n\nVariants:\n\
         \n* `b`\
         \n* `c`\n\n");
}

#[test]
fn should_write_variant_list_with_signatures() {
    let markdown = test::render("enum a { b(int), #[doc = \"a\"] c(int) }");
    assert str::contains(
        markdown,
        "\n\nVariants:\n\
         \n* `b(int)`\
         \n* `c(int)` - a\n\n");
}

fn write_res(ctxt: ctxt, doc: doc::resdoc) {
    write_header(ctxt, h2, doc::restag(doc));
    write_sig(ctxt, doc.sig);
    write_brief(ctxt, doc.brief());
    write_desc(ctxt, doc.desc());
    write_args(ctxt, doc.args);
}

#[test]
fn should_write_resource_header() {
    let markdown = test::render("resource r(a: bool) { }");
    assert str::contains(markdown, "## Resource `r`");
}

#[test]
fn should_write_resource_signature() {
    let markdown = test::render("resource r(a: bool) { }");
    assert str::contains(markdown, "\n    resource r(a: bool)\n");
}

#[test]
fn should_write_resource_args() {
    let markdown = test::render("#[doc(args(a = \"b\"))]\
                                 resource r(a: bool) { }");
    assert str::contains(markdown, "Arguments:\n\n* `a` - b");
}

fn write_iface(ctxt: ctxt, doc: doc::ifacedoc) {
    write_header(ctxt, h2, doc::ifacetag(doc));
    write_brief(ctxt, doc.brief());
    write_desc(ctxt, doc.desc());
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
        doc.brief,
        doc.desc,
        doc.args,
        doc.return,
        doc.failure
    );
}

#[test]
fn should_write_iface_header() {
    let markdown = test::render("iface i { fn a(); }");
    assert str::contains(markdown, "## Interface `i`");
}

#[test]
fn should_write_iface_brief() {
    let markdown = test::render(
        "#[doc(brief = \"brief\")] iface i { fn a(); }");
    assert str::contains(markdown, "brief");
}

#[test]
fn should_write_iface_desc() {
    let markdown = test::render(
        "#[doc(desc = \"desc\")] iface i { fn a(); }");
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

#[test]
fn should_write_iface_method_argument_header() {
    let markdown = test::render(
        "iface a { fn a(b: int); }");
    assert str::contains(markdown, "\n\nArguments:\n\n");
}

#[test]
fn should_write_iface_method_arguments() {
    let markdown = test::render(
        "iface a { fn a(b: int); }");
    assert str::contains(markdown, "* `b`\n");
}

#[test]
fn should_not_write_iface_method_arguments_if_none() {
    let markdown = test::render(
        "iface a { fn a(); }");
    assert !str::contains(markdown, "Arguments");
}

#[test]
fn should_write_iface_method_return_info() {
    let markdown = test::render(
        "iface a { #[doc(return = \"test\")] fn a() -> int; }");
    assert str::contains(markdown, "Return value: test");
}

#[test]
fn should_write_iface_method_failure_conditions() {
    let markdown = test::render(
        "iface a { #[doc(failure = \"nuked\")] fn a(); }");
    assert str::contains(markdown, "Failure conditions: nuked");
}

fn write_impl(ctxt: ctxt, doc: doc::impldoc) {
    write_header(ctxt, h2, doc::impltag(doc));
    write_brief(ctxt, doc.brief());
    write_desc(ctxt, doc.desc());
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
fn should_write_impl_brief() {
    let markdown = test::render(
        "#[doc(brief = \"brief\")] impl i for int { fn a() { } }");
    assert str::contains(markdown, "brief");
}

#[test]
fn should_write_impl_desc() {
    let markdown = test::render(
        "#[doc(desc = \"desc\")] impl i for int { fn a() { } }");
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

#[test]
fn should_write_impl_method_argument_header() {
    let markdown = test::render(
        "impl a for int { fn a(b: int) { } }");
    assert str::contains(markdown, "\n\nArguments:\n\n");
}

#[test]
fn should_write_impl_method_arguments() {
    let markdown = test::render(
        "impl a for int { fn a(b: int) { } }");
    assert str::contains(markdown, "* `b`\n");
}

#[test]
fn should_not_write_impl_method_arguments_if_none() {
    let markdown = test::render(
        "impl a for int { fn a() { } }");
    assert !str::contains(markdown, "Arguments");
}

#[test]
fn should_write_impl_method_return_info() {
    let markdown = test::render(
        "impl a for int { #[doc(return = \"test\")] fn a() -> int { } }");
    assert str::contains(markdown, "Return value: test");
}

#[test]
fn should_write_impl_method_failure_conditions() {
    let markdown = test::render(
        "impl a for int { #[doc(failure = \"nuked\")] fn a() { } }");
    assert str::contains(markdown, "Failure conditions: nuked");
}

fn write_type(
    ctxt: ctxt,
    doc: doc::tydoc
) {
    write_header(ctxt, h2, doc::tytag(doc));
    write_sig(ctxt, doc.sig);
    write_brief(ctxt, doc.brief());
    write_desc(ctxt, doc.desc());
}

#[test]
fn should_write_type_header() {
    let markdown = test::render("type t = int;");
    assert str::contains(markdown, "## Type `t`");
}

#[test]
fn should_write_type_brief() {
    let markdown = test::render(
        "#[doc(brief = \"brief\")] type t = int;");
    assert str::contains(markdown, "\n\nbrief\n\n");
}

#[test]
fn should_write_type_desc() {
    let markdown = test::render(
        "#[doc(desc = \"desc\")] type t = int;");
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
    fn write_markdown_should_write_crate_header() {
        astsrv::from_str("") {|srv|
            let doc = extract::from_srv(srv, "belch");
            let doc = attr_pass::mk_pass().f(srv, doc);
            let markdown = write_markdown_str(doc);
            assert str::contains(markdown, "# Crate `belch`");
        }
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
