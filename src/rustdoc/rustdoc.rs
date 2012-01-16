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

    let source_file = argv[1];
    let default_name = source_file;
    let crate = parse::from_file(source_file);
    let doc = extract::extract(crate, default_name);
    gen::write_markdown(doc, crate, io::stdout());
}
