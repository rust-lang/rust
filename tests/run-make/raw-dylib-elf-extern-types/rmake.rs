//@ only-elf
//@ ignore-cross-compile: Runs a binary.
//@ needs-dynamic-linking
// FIXME(raw_dylib_elf): Debug the failures on other targets.
//@ only-gnu
//@ only-x86_64

//@ ignore-rustc-debug-assertions

use run_make_support::{build_native_dynamic_lib, diff, run, rustc};

fn main() {
    rustc().crate_type("bin").crate_name("raw_dylib_test").input("bin.rs").run();
    build_native_dynamic_lib("extern");

    let out_raw = run("raw_dylib_test").stdout_utf8();
    diff().expected_file("output.txt").actual_text("actual", out_raw).normalize(r#"\r"#, "").run();
}
