type ctxt = {
    ps: pprust::ps,
    w: io::writer
};

fn write_markdown(
    doc: doc::cratedoc,
    _crate: @ast::crate,
    writer: io::writer
) {
    let ctxt = {
        ps: pprust::rust_printer(writer),
        w: writer
    };

    write_header(ctxt, doc.topmod.name);
}

#[doc(
  brief = "Generate a crate document header.",
  args(rd = "Rustdoc context",
       name = "Crate name")
)]
fn write_header(ctxt: ctxt, name: str) {
    ctxt.w.write_line("# Crate " + name);
}

#[doc(
  brief = "Documents a single function.",
  args(rd = "Rustdoc context",
       ident = "Identifier for this function",
       doc = "Function docs extracted from attributes",
       _fn = "AST object representing this function")
)]
fn write_fndoc(ctxt: ctxt, ident: str, doc: doc::fndoc, decl: ast::fn_decl) {
    ctxt.w.write_line("## Function `" + ident + "`");
    ctxt.w.write_line(doc.brief);
    alt doc.desc {
        some(_d) {
            ctxt.w.write_line("");
            ctxt.w.write_line(_d);
            ctxt.w.write_line("");
        }
        none. { }
    }
    for arg: ast::arg in decl.inputs {
        ctxt.w.write_str("### Argument `" + arg.ident + "`: ");
        ctxt.w.write_line("`" + pprust::ty_to_str(arg.ty) + "`");
        alt doc.args.find(arg.ident) {
            some(_d) {
                ctxt.w.write_line(_d);
            }
            none. { }
        };
    }
    ctxt.w.write_line("### Returns `" + pprust::ty_to_str(decl.output) + "`");
    alt doc.return {
        some(_r) { ctxt.w.write_line(_r); }
        none. { }
    }
}

#[cfg(test)]
mod tests {
    fn write_markdown_str(
        doc: doc::cratedoc,
        crate: @ast::crate
    ) -> str {
        let buffer = io::mk_mem_buffer();
        let writer = io::mem_buffer_writer(buffer);
        write_markdown(doc, crate, writer);
        ret io::mem_buffer_str(buffer);
    }

    #[test]
    fn write_markdown_should_write_crate_header() {
        let source = "";
        let ast = parse::from_str(source);
        let doc = extract::extract(ast, "belch");
        let markdown = write_markdown_str(doc, ast);
        assert str::contains(markdown, "# Crate belch\n");
    }
}