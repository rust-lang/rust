//@ needs-target-std
//
// An extended version of the ui/changing-crates.rs test, this test puts
// multiple mismatching crates into the search path of crateC (A2 and A3)
// and checks that the standard error contains helpful messages to indicate
// what should be done to fix the issue.
// See https://github.com/rust-lang/rust/issues/13266

use run_make_support::{rfs, rustc};

fn main() {
    rfs::create_dir("a1");
    rfs::create_dir("a2");
    rfs::create_dir("a3");
    rustc().crate_type("rlib").out_dir("a1").input("crateA1.rs").run();
    rustc().crate_type("rlib").library_search_path("a1").input("crateB.rs").run();
    rustc().crate_type("rlib").out_dir("a2").input("crateA2.rs").run();
    rustc().crate_type("rlib").out_dir("a3").input("crateA3.rs").run();
    // Ensure crateC fails to compile since A1 is "missing" and A2/A3 hashes do not match
    rustc()
        .crate_type("rlib")
        .library_search_path("a2")
        .library_search_path("a3")
        .input("crateC.rs")
        .run_fail()
        .assert_stderr_contains(
            "found possibly newer version of crate `crateA` which `crateB` depends on",
        )
        .assert_stderr_contains("note: perhaps that crate needs to be recompiled?")
        .assert_stderr_contains("crate `crateA`:")
        .assert_stderr_contains("crate `crateB`:");
    // the 'crate `crateA`' will match two entries.
}
