// `raw-dylib` is a Windows-specific attribute which emits idata sections for the items in the
// attached extern block,
// so they may be linked against without linking against an import library.
// To learn more, read https://github.com/rust-lang/rfcs/blob/master/text/2627-raw-dylib-kind.md
// This test uses this feature alongside alternative calling conventions, checking that both
// features are compatible and result in the expected output upon execution of the binary.
// See https://github.com/rust-lang/rust/pull/84171

//@ only-x86
//@ only-windows

use run_make_support::{
    build_native_dynamic_lib, diff, is_windows_msvc, run, run_with_args, rustc,
};

fn main() {
    rustc()
        .crate_type("lib")
        .crate_name("raw_dylib_alt_calling_convention_test")
        .input("lib.rs")
        .run();
    rustc().crate_type("bin").input("driver.rs").run();
    build_native_dynamic_lib("extern");
    let out = run("driver").stdout_utf8();
    diff().expected_file("output.txt").actual_text("actual", out).normalize(r#"\r"#, "").run();
    if is_windows_msvc() {
        let out_msvc = run_with_args("driver", &["true"]).stdout_utf8();
        diff()
            .expected_file("output.msvc.txt")
            .actual_text("actual", out_msvc)
            .normalize(r#"\r"#, "")
            .run();
    }
}
