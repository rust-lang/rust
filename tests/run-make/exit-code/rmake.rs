//@ ignore-cross-compile

// Test that we exit with the correct exit code for successful / unsuccessful / ICE compilations

use run_make_support::{rustc, rustdoc};

fn main() {
    rustc().arg("success.rs").run();

    rustc().arg("--invalid-arg-foo").run_fail().assert_exit_code(1);

    rustc().arg("compile-error.rs").run_fail().assert_exit_code(1);

    rustc()
        .env("RUSTC_ICE", "0")
        .arg("-Ztreat-err-as-bug")
        .arg("compile-error.rs")
        .run_fail()
        .assert_exit_code(101);

    rustdoc().arg("success.rs").out_dir("exit-code").run();

    rustdoc().arg("--invalid-arg-foo").run_fail().assert_exit_code(1);

    rustdoc().arg("compile-error.rs").run_fail().assert_exit_code(1);

    rustdoc().arg("lint-failure.rs").run_fail().assert_exit_code(1);
}
