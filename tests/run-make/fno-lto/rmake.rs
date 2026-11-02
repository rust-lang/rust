//@ needs-target-std
//@ ignore-msvc
//@ ignore-wasm
//@ needs-backends: gcc

use run_make_support::rustc;

fn main() {
    rustc().input("main.rs").print("link-args").run_unchecked().assert_stdout_contains("-fno-lto");

    rustc()
        .input("main.rs")
        .arg("-Clinker-plugin-lto")
        .print("link-args")
        .run_unchecked()
        .assert_stdout_not_contains("-fno-lto");
}
