/* rustdoc: rust -> markdown translator
 * Copyright 2011 Google Inc.
 */

use std;
use rustc;

import option;
import option::{some, none};
import rustc::driver::diagnostic;
import rustc::syntax::ast;
import rustc::syntax::codemap;
import rustc::syntax::parse::parser;
import rustc::syntax::print::pprust;
import rustc::syntax::visit;
import std::io;
import io::writer_util;
import std::map;

type rustdoc = {
    ps: pprust::ps,
    w: io::writer
};

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
            ast::meta_name_value(
                "doc", {node: ast::lit_str(value), span: _}) {
                _fndoc = some(~{
                    name: "todo",
                    brief: value,
                    desc: none,
                    return: none,
                    args: noargdocs
                });
            }
            ast::meta_list("doc", docs) {
                _fndoc = some(
                    attr_parser::parse_compound_fndoc(docs));
            }
        }
    }

    let _fndoc0 = alt _fndoc {
        some(_d) { _d }
        none. {
          ~{
              name: "todo",
              brief: "_undocumented_",
              desc: none,
              return: none,
              args: noargdocs
          }
        }
    };

    alt item.node {
        ast::item_const(ty, expr) { }
        ast::item_fn(decl, _, _) {
            gen::write_fndoc(rd, item.ident, _fndoc0, decl);
        }
        ast::item_mod(_mod) { }
        ast::item_ty(ty, typarams) { }
        ast::item_tag(variant, typarams) { }
        ast::item_res(_, _, _, _, _) { }
    };
}

#[doc(
  brief = "Main function.",
  desc = "Command-line arguments:

*  argv[1]: crate file name",
  args(argv = "Command-line arguments.")
)]
fn main(argv: [str]) {

    if vec::len(argv) != 2u {
        io::println(#fmt("usage: %s <input>", argv[0]));
        ret;
    }

    let crate = parse::from_file(argv[1]);

    let w = io::stdout();
    let rd = { ps: pprust::rust_printer(w), w: w };
    gen::write_header(rd, argv[1]);

    let v = visit::mk_simple_visitor(@{
        visit_item: bind doc_item(rd, _)
        with *visit::default_simple_visitor()});
    visit::visit_crate(*crate, (), v);
}
