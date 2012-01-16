type ctxt = {
    ps: pprust::ps,
    w: io::writer
};

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
