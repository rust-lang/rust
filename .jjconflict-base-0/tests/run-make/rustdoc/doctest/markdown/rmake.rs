// Doctests need std and executables that can run on the host
//@ needs-target-std
//@ ignore-wasm
//@ ignore-sgx
//@ ignore-pauthtest
//@ ignore-remote (e.g. armhf-gnu)

use run_make_support::{rust_lib_name, rustc, rustdoc};

fn main() {
    rustdoc().arg("--test").input("good.md").run().assert_exit_code(0).assert_stdout_contains(
        "test result: ok. 3 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out;",
    );

    rustdoc()
        .arg("--test")
        .input("bad.md")
        .run_fail()
        .assert_exit_code(101)
        .assert_stdout_contains(
            "test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out;",
        );

    rustc().input("extadd.rs").crate_type("rlib").run();
    rustdoc()
        .arg("--test")
        .extern_("aux", rust_lib_name("extadd"))
        .input("extern.md")
        .run()
        .assert_exit_code(0)
        .assert_stdout_contains(
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;",
        );
}
