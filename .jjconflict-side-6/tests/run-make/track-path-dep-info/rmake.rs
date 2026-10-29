//@ ignore-cross-compile
//@ needs-crate-type: proc-macro
//@ ignore-musl (FIXME: can't find `-lunwind`)

// This test checks the functionality of `tracked_path::path`, a procedural macro
// feature that adds a dependency to another file inside the procmacro. In this case,
// the text file is added through this method, and the test checks that the compilation
// output successfully added the file as a dependency.
// See https://github.com/rust-lang/rust/pull/84029

use run_make_support::{rfs, rustc};

fn main() {
    rustc().input("macro_def.rs").run();
    rustc().env("EXISTING_PROC_MACRO_ENV", "1").emit("dep-info").input("macro_use.rs").run();
    assert!(rfs::read_to_string("macro_use.d").contains("emojis.txt:"));
}
