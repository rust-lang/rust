//@ needs-target-std
//
// A simple smoke test to check that rustc fails compilation
// and outputs a helpful message when a dependency is missing
// in a dependency chain.
// See https://github.com/rust-lang/rust/issues/12146

use run_make_support::{rfs, rust_lib_name, rustc};

fn main() {
    rustc().crate_type("rlib").input("crateA.rs").run();
    rustc().crate_type("rlib").input("crateB.rs").run();
    rfs::remove_file(rust_lib_name("crateA"));
    // Ensure that crateC fails to compile, as the crateA dependency is missing.
    rustc()
        .input("crateC.rs")
        .run_fail()
        .assert_stderr_contains("can't find crate for `crateA` which `crateB` depends on");
}
