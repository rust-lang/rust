#[doc = "Generate markdown from a document tree"];

import std::io;
import std::io::writer_util;

export mk_pass;

fn mk_pass(
    writer: fn~() -> io::writer
) -> pass {
    ret fn~(
        _srv: astsrv::srv,
        doc: doc::cratedoc
    ) -> doc::cratedoc {
        write_markdown(doc, writer());
        doc
    };
}

type ctxt = {
    w: io::writer
};

fn write_markdown(
    doc: doc::cratedoc,
    writer: io::writer
) {
    let ctxt = {
        w: writer
    };

    write_crate(ctxt, doc);
}

enum hlvl {
    h1 = 1,
    h2 = 2,
    h3 = 3
}

fn write_header(ctxt: ctxt, lvl: hlvl, title: str) {
    let hashes = str::from_chars(vec::init_elt(lvl as uint, '#'));
    ctxt.w.write_line(#fmt("%s %s", hashes, title));
    ctxt.w.write_line("");
}

fn write_crate(
    ctxt: ctxt,
    doc: doc::cratedoc
) {
    write_header(ctxt, h1, #fmt("Crate %s", doc.topmod.name));
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
    let fullpath = str::connect(moddoc.path + [moddoc.name], "::");
    write_header(ctxt, h2, #fmt("Module `%s`", fullpath));
    write_mod_contents(ctxt, moddoc);
}

#[test]
fn should_write_full_path_to_mod() {
    let markdown = test::render("mod a { mod b { mod c { } } }");
    assert str::contains(markdown, "## Module `a::b::c`");
}

fn write_mod_contents(
    ctxt: ctxt,
    doc: doc::moddoc
) {
    write_brief(ctxt, doc.brief);
    write_desc(ctxt, doc.desc);

    for constdoc in *doc.consts {
        write_const(ctxt, constdoc);
    }

    for enumdoc in *doc.enums {
        write_enum(ctxt, enumdoc);
    }

    for fndoc in *doc.fns {
        write_fn(ctxt, fndoc);
    }

    for resdoc in *doc.resources {
        write_res(ctxt, resdoc);
    }

    for moddoc in *doc.mods {
        write_mod(ctxt, moddoc);
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

fn write_fn(
    ctxt: ctxt,
    doc: doc::fndoc
) {
    write_header(ctxt, h3, #fmt("Function `%s`", doc.name));
    write_sig(ctxt, doc.sig);
    write_brief(ctxt, doc.brief);
    write_desc(ctxt, doc.desc);
    write_args(ctxt, doc.args);
    write_return(ctxt, doc.return);
    write_failure(ctxt, doc.failure);
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
    let indented = vec::map(lines, { |line| #fmt("    %s", line) });
    str::connect(indented, "\n")
}

#[test]
fn write_markdown_should_write_function_header() {
    let markdown = test::render("fn func() { }");
    assert str::contains(markdown, "### Function `func`");
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
    let doc = ~{
        topmod: ~{
            fns: doc::fnlist([~{
                sig: some("line 1\nline 2")
                with *doc.topmod.fns[0]
            }])
            with *doc.topmod
        }
        with *doc
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
    assert option::is_some(arg.ty);
    ctxt.w.write_str(#fmt(
        "* `%s`: `%s`",
        arg.name,
        option::get(arg.ty)
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
         * `b`: `int`\n\
         * `c`: `int`\n\
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
    assert str::contains(markdown, "`a`: `bool` - milk");
}

fn write_return(
    ctxt: ctxt,
    doc: doc::retdoc
) {
    alt doc.ty {
      some(ty) {
        ctxt.w.write_str(#fmt("Returns `%s`", ty));
        alt doc.desc {
          some(d) {
            ctxt.w.write_line(#fmt(" - %s", d));
            ctxt.w.write_line("");
          }
          none {
            ctxt.w.write_line("");
            ctxt.w.write_line("");
          }
        }
      }
      none { }
    }
}

#[test]
fn should_write_return_type_on_new_line() {
    let markdown = test::render("fn a() -> int { }");
    assert str::contains(markdown, "\nReturns `int`");
}

#[test]
fn should_write_blank_line_between_return_type_and_next_header() {
    let markdown = test::render(
        "fn a() -> int { } \
         fn b() -> int { }"
    );
    assert str::contains(markdown, "Returns `int`\n\n##");
}

#[test]
fn should_not_write_return_type_when_there_is_none() {
    let markdown = test::render("fn a() { }");
    assert !str::contains(markdown, "Returns");
}

#[test]
fn should_write_blank_line_after_return_description() {
    let markdown = test::render(
        "#[doc(return = \"blorp\")] fn a() -> int { }"
    );
    assert str::contains(markdown, "blorp\n\n");
}

#[test]
fn should_write_return_description_on_same_line_as_type() {
    let markdown = test::render(
        "#[doc(return = \"blorp\")] fn a() -> int { }"
    );
    assert str::contains(markdown, "Returns `int` - blorp");
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
    write_header(ctxt, h3, #fmt("Const `%s`", doc.name));
    write_sig(ctxt, doc.ty);
    write_brief(ctxt, doc.brief);
    write_desc(ctxt, doc.desc);
}

#[test]
fn should_write_const_header() {
    let markdown = test::render("const a: bool = true;");
    assert str::contains(markdown, "### Const `a`\n\n");
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
    write_header(ctxt, h3, #fmt("Enum `%s`", doc.name));
    write_brief(ctxt, doc.brief);
    write_desc(ctxt, doc.desc);
    write_variants(ctxt, doc.variants);
}

#[test]
fn should_write_enum_header() {
    let markdown = test::render("enum a { b }");
    assert str::contains(markdown, "### Enum `a`\n\n");
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
    write_header(ctxt, h3, #fmt("Resource `%s`", doc.name));
    write_sig(ctxt, doc.sig);
    write_brief(ctxt, doc.brief);
    write_desc(ctxt, doc.desc);
    write_args(ctxt, doc.args);
}

#[test]
fn should_write_resource_header() {
    let markdown = test::render("resource r(a: bool) { }");
    assert str::contains(markdown, "### Resource `r`");
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
    assert str::contains(markdown, "Arguments:\n\n* `a`: `bool` - b");
}

#[cfg(test)]
mod test {
    fn render(source: str) -> str {
        let doc = create_doc(source);
        let markdown = write_markdown_str(doc);
        #debug("markdown: %s", markdown);
        markdown
    }

    fn create_doc(source: str) -> doc::cratedoc {
        let srv = astsrv::mk_srv_from_str(source);
        let doc = extract::from_srv(srv, "");
        #debug("doc (extract): %?", doc);
        let doc = tystr_pass::mk_pass()(srv, doc);
        #debug("doc (tystr): %?", doc);
        let doc = path_pass::mk_pass()(srv, doc);
        #debug("doc (path): %?", doc);
        let doc = attr_pass::mk_pass()(srv, doc);
        #debug("doc (attr): %?", doc);
        doc
    }

    fn write_markdown_str(
        doc: doc::cratedoc
    ) -> str {
        let buffer = io::mk_mem_buffer();
        let writer = io::mem_buffer_writer(buffer);
        write_markdown(doc, writer);
        ret io::mem_buffer_str(buffer);
    }

    #[test]
    fn write_markdown_should_write_crate_header() {
        let srv = astsrv::mk_srv_from_str("");
        let doc = extract::from_srv(srv, "belch");
        let doc = attr_pass::mk_pass()(srv, doc);
        let markdown = write_markdown_str(doc);
        assert str::contains(markdown, "# Crate belch");
    }

    #[test]
    fn write_markdown_should_write_mod_headers() {
        let markdown = render("mod moo { }");
        assert str::contains(markdown, "## Module `moo`");
    }

    #[test]
    fn should_leave_blank_line_after_header() {
        let markdown = render("mod morp { }");
        assert str::contains(markdown, "Module `morp`\n\n");
    }
}