// In this test, m4 depends on m3, which depends on m2, which depends on m1.
// Even though dependencies are chained like this and there is no direct mention
// of m1 or m2 in m4.rs, compilation and execution should still succeed. Unlike
// the dylib-chain test, rlibs do not contain upstream dependencies, and removing
// the libraries still allows m4 to successfully execute.
// See https://github.com/rust-lang/rust/issues/10434

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{rfs, run, rust_lib_name, rustc};

fn main() {
    rustc().input("m1.rs").run();
    rustc().input("m2.rs").run();
    rustc().input("m3.rs").run();
    rustc().input("m4.rs").run();
    run("m4");
    rfs::remove_file(rust_lib_name("m1"));
    rfs::remove_file(rust_lib_name("m2"));
    rfs::remove_file(rust_lib_name("m3"));
    run("m4");
}
