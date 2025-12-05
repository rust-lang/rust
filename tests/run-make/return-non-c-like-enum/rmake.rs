// Check that we treat enum variants like union members in call ABIs.
// Added in #68443.
// Original issue: #68190.

//@ ignore-cross-compile

use run_make_support::{cc, extra_c_flags, extra_cxx_flags, run, rustc, static_lib_name};

fn main() {
    rustc().crate_type("staticlib").input("nonclike.rs").run();
    cc().input("test.c")
        .arg(&static_lib_name("nonclike"))
        .out_exe("test")
        .args(extra_c_flags())
        .args(extra_cxx_flags())
        .run();
    run("test");
}
