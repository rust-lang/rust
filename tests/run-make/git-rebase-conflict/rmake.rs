//@ needs-target-std
//
// An attempt to set the output `-o` into a directory or a file we cannot write into should indeed
// be an error; but not an ICE (Internal Compiler Error). This test attempts both and checks
// that the standard error matches what is expected.
// See https://github.com/rust-lang/rust/issues/66530

use run_make_support::{diff, rustc};

fn main() {
    // We use `.rust` purely to avoid `fmt` check on a file with conflict markers
    let file_out = rustc().input("main.rust").run_fail().stderr_utf8();
    diff().expected_file("file.stderr").actual_text("actual-file-stderr", file_out).run();
}
