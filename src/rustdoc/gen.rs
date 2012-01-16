type ctxt = {
    ps: pprust::ps,
    w: io::writer
};

fn write_markdown(
    doc: doc::cratedoc,
    crate: @ast::crate,
    writer: io::writer
) {
    let ctxt = {
        ps: pprust::rust_printer(writer),
        w: writer
    };

    write_header(ctxt, doc.topmod.name);
    write_top_module(ctxt, crate, doc.topmod);
}

fn write_top_module(
    ctxt: ctxt,
    crate: @ast::crate,
    moddoc: doc::moddoc
) {
    write_mod_contents(ctxt, crate, moddoc);
}

fn write_mod(
    ctxt: ctxt,
    crate: @ast::crate,
    moddoc: doc::moddoc
) {
    write_mod_contents(ctxt, crate, moddoc);
}

fn write_mod_contents(
    ctxt: ctxt,
    crate: @ast::crate,
    moddoc: doc::moddoc
) {
    for fndoc in *moddoc.fns {
        write_fn(ctxt, crate, fndoc);
    }

    for moddoc in *moddoc.mods {
        write_mod(ctxt, crate, moddoc);
    }
}

fn write_fn(
    ctxt: ctxt,
    crate: @ast::crate,
    fndoc: doc::fndoc
) {
    import rustc::middle::ast_map;

    let map = ast_map::map_crate(*crate);
    let decl = alt map.get(fndoc.id) {
      ast_map::node_item(@{
        node: ast::item_fn(decl, _, _), _
      }) { decl }
    };

    write_fndoc(ctxt, fndoc.name, fndoc, decl);
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

    #[test]
    fn write_markdown_should_write_function_header() {
        let source = "fn func() { }";
        let ast = parse::from_str(source);
        let doc = extract::extract(ast, "");
        let markdown = write_markdown_str(doc, ast);
        assert str::contains(markdown, "## Function `func`");
    }
}