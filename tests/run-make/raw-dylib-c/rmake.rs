// `raw-dylib` is a Windows-specific attribute which emits idata sections for the items in the
// attached extern block,
// so they may be linked against without linking against an import library.
// To learn more, read https://github.com/rust-lang/rfcs/blob/master/text/2627-raw-dylib-kind.md
// This test is the simplest of the raw-dylib tests, simply smoke-testing that the feature
// can be used to build an executable binary with an expected output with native C files
// compiling into dynamic libraries.
// See https://github.com/rust-lang/rust/pull/86419

//@ only-windows

use run_make_support::{build_native_dynamic_lib, diff, run, rustc};

fn main() {
    rustc().crate_type("lib").crate_name("raw_dylib_test").input("lib.rs").run();
    rustc().crate_type("bin").input("driver.rs").run();
    rustc().crate_type("bin").crate_name("raw_dylib_test_bin").input("lib.rs").run();
    build_native_dynamic_lib("extern_1");
    build_native_dynamic_lib("extern_2");
    let out_driver = run("driver").stdout_utf8();
    let out_raw = run("raw_dylib_test_bin").stdout_utf8();

    diff()
        .expected_file("output.txt")
        .actual_text("actual", out_driver)
        .normalize(r#"\r"#, "")
        .run();
    diff().expected_file("output.txt").actual_text("actual", out_raw).normalize(r#"\r"#, "").run();
}
