//@ only-linux
//@ ignore-cross-compile

use run_make_support::{bin_name, llvm_nm, run, rust_lib_name, rustc};

fn main() {
    rustc().edition("2021").input("dep.rs").crate_type("rlib").run();

    rustc()
        .edition("2021")
        .input("middle.rs")
        .crate_type("rlib")
        .extern_("dep", rust_lib_name("dep"))
        .run();

    rustc().edition("2021").input("app.rs").extern_("middle", rust_lib_name("middle")).run();

    run("app").assert_stdout_equals("7").assert_stderr_not_contains("constructor ran");

    llvm_nm()
        .input(bin_name("app"))
        .run()
        .assert_stdout_not_contains_regex("[Tt] _*mark_ctor");
}
