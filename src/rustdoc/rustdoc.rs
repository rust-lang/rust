/* rustdoc: rust -> markdown translator
 * Copyright 2011 Google Inc.
 */

use std;
use rustc;

import option;
import option::{some, none};
import rustc::syntax::ast;
import rustc::syntax::codemap;
import rustc::syntax::parse::parser;
import rustc::syntax::print::pprust;
import rustc::syntax::visit;
import std::io;
import std::map;

type rustdoc = {
    ps: pprust::ps,
    w: io::writer
};

type fndoc = {
    brief: str,
    desc: option::t<str>,
    return: option::t<str>,
    args: map::hashmap<str, str>
};

#[doc(
  brief = "Documents a single function.",
  args(rd = "Rustdoc context",
       ident = "Identifier for this function",
       doc = "Function docs extracted from attributes",
       _fn = "AST object representing this function")
)]
fn doc_fn(rd: rustdoc, ident: str, doc: fndoc, decl: ast::fn_decl) {
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

#[doc(
  brief = "Parses function docs from a complex #[doc] attribute.",
  desc = "Supported attributes:

* `brief`: Brief description
* `desc`: Long description
* `return`: Description of return value
* `args`: List of argname = argdesc pairs
",
  args(items = "Doc attribute contents"),
  return = "Parsed function docs."
)]
fn parse_compound_fndoc(items: [@ast::meta_item]) -> fndoc {
    let brief = none;
    let desc = none;
    let return = none;
    let argdocs = map::new_str_hash::<str>();
    let argdocsfound = none;
    for item: @ast::meta_item in items {
        alt item.node {
            ast::meta_name_value("brief", {node: ast::lit_str(value),
                                           span: _}) {
                brief = some(value);
            }
            ast::meta_name_value("desc", {node: ast::lit_str(value),
                                              span: _}) {
                desc = some(value);
            }
            ast::meta_name_value("return", {node: ast::lit_str(value),
                                            span: _}) {
                return = some(value);
            }
            ast::meta_list("args", args) {
                argdocsfound = some(args);
            }
            _ { }
        }
    }

    alt argdocsfound {
        none. { }
        some(ds) {
            for d: @ast::meta_item in ds {
                alt d.node {
                    ast::meta_name_value(key, {node: ast::lit_str(value),
                                               span: _}) {
                        argdocs.insert(key, value);
                    }
                }
            }
        }
    }

    let _brief = alt brief {
        some(_b) { _b }
        none. { "_undocumented_" }
    };

    { brief: _brief, desc: desc, return: return, args: argdocs }
}

#[doc(
  brief = "Documents a single crate item.",
  args(rd = "Rustdoc context",
       item = "AST item to document")
)]
fn doc_item(rd: rustdoc, item: @ast::item) {
    let _fndoc = none;
    let noargdocs = map::new_str_hash::<str>();
    for attr: ast::attribute in item.attrs {
        alt attr.node.value.node {
            ast::meta_name_value("doc", {node: ast::lit_str(value), span: _}) {
                _fndoc = some({ brief: value,
                                desc: none,
                                return: none,
                                args: noargdocs });
            }
            ast::meta_list("doc", docs) {
                _fndoc = some(parse_compound_fndoc(docs));
            }
        }
    }

    let _fndoc0 = alt _fndoc {
        some(_d) { _d }
        none. { { brief: "_undocumented_", desc: none, return: none, args: noargdocs } }
    };

    alt item.node {
        ast::item_const(ty, expr) { }
        ast::item_fn(decl, _, _) {
            doc_fn(rd, item.ident, _fndoc0, decl);
        }
        ast::item_mod(_mod) { }
        ast::item_ty(ty, typarams) { }
        ast::item_tag(variant, typarams) { }
        ast::item_obj(_obj, typarams, node_id) { }
        ast::item_res(_, _, _, _, _) { }
    };
}

#[doc(
  brief = "Generate a crate document header.",
  args(rd = "Rustdoc context",
       name = "Crate name")
)]
fn doc_header(rd: rustdoc, name: str) {
    rd.w.write_line("# Crate " + name);
}

#[doc(
  brief = "Main function.",
  desc = "Command-line arguments:

*  argv[1]: crate file name",
  args(argv = "Command-line arguments.")
)]
fn main(argv: [str]) {
    let sess = @{cm: codemap::new_codemap(), mutable next_id: 0};
    let w = io::stdout();
    let rd = { ps: pprust::rust_printer(w), w: w };
    doc_header(rd, argv[1]);
    let p = parser::parse_crate_from_source_file(argv[1], [], sess);
    let v = visit::mk_simple_visitor(@{
        visit_item: bind doc_item(rd, _)
        with *visit::default_simple_visitor()});
    visit::visit_crate(*p, (), v);
}
