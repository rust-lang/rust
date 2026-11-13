//@ needs-target-std
//
// When rustc is looking for a crate but is given a staticlib instead,
// the error message should be helpful and indicate precisely the cause
// of the compilation failure.
// See https://github.com/rust-lang/rust/pull/21978

use run_make_support::rustc;

fn main() {
    rustc().input("foo.rs").crate_type("staticlib").run();
    rustc().input("bar.rs").run_fail().assert_stderr_contains("found staticlib");
}
