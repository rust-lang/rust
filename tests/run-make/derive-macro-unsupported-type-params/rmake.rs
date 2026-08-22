//@ ignore-cross-compile
//@ needs-crate-type: proc-macro

use run_make_support::{diff, rustc, target};

fn main() {
    rustc().input("foo.rs").edition("2024").run();
    let out = rustc().input("bar.rs").edition("2024").run_fail().stderr_utf8();
    diff().expected_file("bar.stderr").actual_text("actual-bar-stderr", out).run();
}
