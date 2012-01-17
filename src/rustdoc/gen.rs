import std::io;
import std::io::writer_util;

export write_markdown;

type ctxt = {
    w: io::writer,
    mutable depth: uint
};

fn write_markdown(
    doc: doc::cratedoc,
    writer: io::writer
) {
    let ctxt = {
        w: writer,
        mutable depth: 1u
    };

    write_crate(ctxt, doc);
}

fn write_header(ctxt: ctxt, title: str) {
    let hashes = str::from_chars(vec::init_elt('#', ctxt.depth));
    ctxt.w.write_line(#fmt("%s %s", hashes, title));
}

fn subsection(ctxt: ctxt, f: fn&()) {
    ctxt.depth += 1u;
    f();
    ctxt.depth -= 1u;
}

fn write_crate(
    ctxt: ctxt,
    doc: doc::cratedoc
) {
    write_header(ctxt, #fmt("Crate %s", doc.topmod.name));
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

fn write_mod_contents(
    ctxt: ctxt,
    moddoc: doc::moddoc
) {
    for fndoc in *moddoc.fns {
        subsection(ctxt) {||
            write_fn(ctxt, fndoc);
        }
    }

    for moddoc in *moddoc.mods {
        subsection(ctxt) {||
            write_mod(ctxt, moddoc);
        }
    }
}

fn write_fn(
    ctxt: ctxt,
    doc: doc::fndoc
) {
    write_header(ctxt, #fmt("Function `%s`", doc.name));
    alt doc.brief {
      some(brief) {
        ctxt.w.write_line(brief);
      }
      none. { }
    }
    alt doc.desc {
        some(_d) {
            ctxt.w.write_line("");
            ctxt.w.write_line(_d);
            ctxt.w.write_line("");
        }
        none. { }
    }
    for (arg, desc) in doc.args {
        ctxt.w.write_str("### Argument `" + arg + "`: ");
        ctxt.w.write_str(desc)
    }
    alt doc.return {
      some(doc) {
        alt doc.ty {
          some(ty) {
            ctxt.w.write_line("### Returns `" + ty + "`");
            alt doc.desc {
              some(d) {
                ctxt.w.write_line(d);
              }
              none. { }
            }
          }
          none. { fail "unimplemented"; }
        }
      }
      none. { }
    }
}

#[cfg(test)]
mod tests {
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
        let source = "";
        let ast = parse::from_str(source);
        let doc = extract::extract(ast, "belch");
        let markdown = write_markdown_str(doc);
        assert str::contains(markdown, "# Crate belch\n");
    }

    #[test]
    fn write_markdown_should_write_function_header() {
        let source = "fn func() { }";
        let ast = parse::from_str(source);
        let doc = extract::extract(ast, "");
        let markdown = write_markdown_str(doc);
        assert str::contains(markdown, "## Function `func`");
    }
}