/* rustdoc: rust -> markdown translator
 * Copyright 2011 Google Inc.
 */

use std;
use rustc;

#[doc(
  brief = "Main function.",
  desc = "Command-line arguments:

*  argv[1]: crate file name",
  args(argv = "Command-line arguments.")
)]
fn main(argv: [str]) {

    if vec::len(argv) != 2u {
        std::io::println(#fmt("usage: %s <input>", argv[0]));
        ret;
    }

    let source_file = argv[1];
    let default_name = source_file;
    let crate = parse::from_file(source_file);
    let doc = extract::extract(crate, default_name);
    let doc = tystr_pass::run(doc, crate);
    gen::write_markdown(doc, std::io::stdout());
}
