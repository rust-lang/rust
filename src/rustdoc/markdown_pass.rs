#[doc = "Generate markdown from a document tree"];

export mk_pass;

fn mk_pass(config: config::config) -> pass {
    mk_pass_(config, markdown_writer(config))
}

enum writeinstr {
    write(str),
    done
}

type writer = fn~(+writeinstr);

impl writer for writer {
    fn write_str(str: str) {
        self(write(str));
    }

    fn write_line(str: str) {
        self.write_str(str + "\n");
    }

    fn write_done() {
        self(done)
    }
}

fn markdown_writer(config: config::config) -> writer {
    let filename = make_filename(config);
    let ch = task::spawn_listener {|po: comm::port<writeinstr>|
        let markdown = "";
        let keep_going = true;
        while keep_going {
            alt comm::recv(po) {
              write(s) { markdown += s; }
              done { keep_going = false; }
            }
        }
        write_file(filename, markdown);
    };

    fn~(+instr: writeinstr) {
        comm::send(ch, instr);
    }
}

fn make_filename(config: config::config) -> str {
    import std::fs;
    let cratefile = fs::basename(config.input_crate);
    let cratename = tuple::first(fs::splitext(cratefile));
    fs::connect(config.output_dir, cratename + ".md")
}

fn write_file(path: str, s: str) {
    import std::io;
    import std::io::writer_util;

    alt io::file_writer(path, [io::create, io::truncate]) {
      result::ok(writer) {
        writer.write_str(s);
      }
      result::err(e) { fail e }
    }
}

#[test]
fn should_use_markdown_file_name_based_off_crate() {
    let config = {
        output_dir: "output/dir"
        with config::default_config("input/test.rc")
    };
    assert make_filename(config) == "output/dir/test.md";
}

// FIXME: This is a really convoluted interface to work around trying
// to get a writer into a unique closure and then being able to test
// what was written afterward
fn mk_pass_(
    config: config::config,
    writer: writer
) -> pass {
    let f = fn~(srv: astsrv::srv, doc: doc::cratedoc) -> doc::cratedoc {
        run(srv, doc, config, writer)
    };

    {
        name: "markdown",
        f: f
    }
}

fn run(
    srv: astsrv::srv,
    doc: doc::cratedoc,
    _config: config::config,
    writer: writer
) -> doc::cratedoc {

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

    write_markdown(sorted_doc, writer);

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
    doc: doc::cratedoc,
    writer: writer
) {
    let ctxt = {
        w: writer
    };

    write_crate(ctxt, doc);
    ctxt.w.write_done();
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
    write_header(ctxt, h1, #fmt("Crate %s", doc.topmod.name()));
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
    let fullpath = str::connect(moddoc.path() + [moddoc.name()], "::");
    write_header(ctxt, h1, #fmt("Module `%s`", fullpath));
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

    for itemtag in doc.items {
        alt itemtag {
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

fn write_nmod(ctxt: ctxt, doc: doc::nmoddoc) {
    let fullpath = str::connect(doc.path() + [doc.name()], "::");
    write_header(ctxt, h1, #fmt("Native module `%s`", fullpath));

    write_brief(ctxt, doc.brief());
    write_desc(ctxt, doc.desc());

    for fndoc in doc.fns {
        write_fn(ctxt, fndoc);
    }
}

#[test]
fn should_write_native_mods() {
    let markdown = test::render("#[doc = \"test\"] native mod a { }");
    log(error, markdown);
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
    write_header(ctxt, h2, #fmt("Function `%s`", doc.name()));
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
        topmod: {
            items: [doc::fntag({
                sig: some("line 1\nline 2")
                with doc.topmod.fns()[0]
            })]
            with doc.topmod
        }
        with doc
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
    write_header(ctxt, h2, #fmt("Const `%s`", doc.name()));
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
    write_header(ctxt, h2, #fmt("Enum `%s`", doc.name()));
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
    write_header(ctxt, h2, #fmt("Resource `%s`", doc.name()));
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
    assert str::contains(markdown, "Arguments:\n\n* `a`: `bool` - b");
}

fn write_iface(ctxt: ctxt, doc: doc::ifacedoc) {
    write_header(ctxt, h2, #fmt("Interface `%s`", doc.name()));
    write_brief(ctxt, doc.brief());
    write_desc(ctxt, doc.desc());
    write_methods(ctxt, doc.methods);
}

fn write_methods(ctxt: ctxt, docs: [doc::methoddoc]) {
    vec::iter(docs) {|doc| write_method(ctxt, doc) }
}

fn write_method(ctxt: ctxt, doc: doc::methoddoc) {
    write_header(ctxt, h3, #fmt("Method `%s`", doc.name));
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
    assert str::contains(markdown, "* `b`: `int`\n");
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
        "iface a { fn a() -> int; }");
    assert str::contains(markdown, "Returns `int`");
}

#[test]
fn should_write_iface_method_failure_conditions() {
    let markdown = test::render(
        "iface a { #[doc(failure = \"nuked\")] fn a(); }");
    assert str::contains(markdown, "Failure conditions: nuked");
}

fn write_impl(ctxt: ctxt, doc: doc::impldoc) {
    assert option::is_some(doc.self_ty);
    let self_ty = option::get(doc.self_ty);
    alt doc.iface_ty {
      some(iface_ty) {
        write_header(ctxt, h2,
                     #fmt("Implementation `%s` of `%s` for `%s`",
                          doc.name(), iface_ty, self_ty));
      }
      none {
        write_header(ctxt, h2,
                     #fmt("Implementation `%s` for `%s`",
                          doc.name(), self_ty));
      }
    }
    write_brief(ctxt, doc.brief());
    write_desc(ctxt, doc.desc());
    write_methods(ctxt, doc.methods);
}

#[test]
fn should_write_impl_header() {
    let markdown = test::render("impl i for int { fn a() { } }");
    assert str::contains(markdown, "## Implementation `i` for `int`");
}

#[test]
fn should_write_impl_header_with_iface() {
    let markdown = test::render("impl i of j for int { fn a() { } }");
    assert str::contains(markdown, "## Implementation `i` of `j` for `int`");
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
    assert str::contains(markdown, "* `b`: `int`\n");
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
        "impl a for int { fn a() -> int { } }");
    assert str::contains(markdown, "Returns `int`");
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
    write_header(ctxt, h2, #fmt("Type `%s`", doc.name()));
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

    fn create_doc_srv(source: str) -> (astsrv::srv, doc::cratedoc) {
        astsrv::from_str(source) {|srv|
            let doc = extract::from_srv(srv, "");
            #debug("doc (extract): %?", doc);
            let doc = tystr_pass::mk_pass().f(srv, doc);
            #debug("doc (tystr): %?", doc);
            let doc = path_pass::mk_pass().f(srv, doc);
            #debug("doc (path): %?", doc);
            let doc = attr_pass::mk_pass().f(srv, doc);
            #debug("doc (attr): %?", doc);
            (srv, doc)
        }
    }

    fn create_doc(source: str) -> doc::cratedoc {
        let (_, doc) = create_doc_srv(source);
        doc
    }

    fn writer_future() -> (writer, future::future<str>) {
        let port = comm::port();
        let chan = comm::chan(port);
        let writer = fn~(+instr: writeinstr) {
            comm::send(chan, copy instr);
        };
        let future = future::from_fn {||
            let res = "";
            while true {
                alt comm::recv(port) {
                  write(s) { res += s }
                  done { break }
                }
            }
            res
        };
        (writer, future)
    }

    fn write_markdown_str(
        doc: doc::cratedoc
    ) -> str {
        let (writer, future) = writer_future();
        write_markdown(doc, writer);
        ret future::get(future);
    }

    fn write_markdown_str_srv(
        srv: astsrv::srv,
        doc: doc::cratedoc
    ) -> str {
        let config = {
            output_style: config::doc_per_crate
            with config::default_config("")
        };
        let (writer, future) = writer_future();
        let pass = mk_pass_(config, writer);
        pass.f(srv, doc);
        ret future::get(future);
    }

    #[test]
    fn write_markdown_should_write_crate_header() {
        astsrv::from_str("") {|srv|
            let doc = extract::from_srv(srv, "belch");
            let doc = attr_pass::mk_pass().f(srv, doc);
            let markdown = write_markdown_str(doc);
            assert str::contains(markdown, "# Crate belch");
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
