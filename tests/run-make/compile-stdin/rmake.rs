// When provided standard input piped directly into rustc, this test checks that the compilation
// completes successfully and that the output can be executed.
//
// See <https://github.com/rust-lang/rust/pull/28805>.

//@ ignore-cross-compile

use run_make_support::{run, rustc};

fn main() {
    rustc().arg("-").stdin_buf("fn main() {}").run();
    run("rust_out");
}
