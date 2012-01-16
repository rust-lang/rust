type rustdoc = {
    ps: pprust::ps,
    w: io::writer
};

#[doc(
  brief = "Generate a crate document header.",
  args(rd = "Rustdoc context",
       name = "Crate name")
)]
fn write_header(rd: rustdoc, name: str) {
    rd.w.write_line("# Crate " + name);
}

#[doc(
  brief = "Documents a single function.",
  args(rd = "Rustdoc context",
       ident = "Identifier for this function",
       doc = "Function docs extracted from attributes",
       _fn = "AST object representing this function")
)]
fn write_fndoc(rd: rustdoc, ident: str, doc: doc::fndoc, decl: ast::fn_decl) {
    rd.w.write_line("## Function `" + ident + "`");
    rd.w.write_line(doc.brief);
    alt doc.desc {
        some(_d) {
            rd.w.write_line("");
            rd.w.write_line(_d);
            rd.w.write_line("");
        }
        none. { }
    }
    for arg: ast::arg in decl.inputs {
        rd.w.write_str("### Argument `" + arg.ident + "`: ");
        rd.w.write_line("`" + pprust::ty_to_str(arg.ty) + "`");
        alt doc.args.find(arg.ident) {
            some(_d) {
                rd.w.write_line(_d);
            }
            none. { }
        };
    }
    rd.w.write_line("### Returns `" + pprust::ty_to_str(decl.output) + "`");
    alt doc.return {
        some(_r) { rd.w.write_line(_r); }
        none. { }
    }
}
