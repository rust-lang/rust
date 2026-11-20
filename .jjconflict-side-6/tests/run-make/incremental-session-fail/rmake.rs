// Failing to create the directory where output incremental
// files would be stored used to cause an ICE (Internal Compiler
// Error). This was patched in #85698, and this test checks that
// the ensuing compilation failure is not an ICE.
// See https://github.com/rust-lang/rust/pull/85698

use run_make_support::{rfs, rustc};

fn main() {
    rfs::create_file("session");
    // rustc should fail to create the session directory here.
    let out = rustc().input("foo.rs").crate_type("rlib").incremental("session").run_fail();
    out.assert_stderr_contains("could not create incremental compilation crate directory");
    out.assert_stderr_not_contains("internal compiler error");
}
