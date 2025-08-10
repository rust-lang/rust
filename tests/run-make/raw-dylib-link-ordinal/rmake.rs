// `raw-dylib` is a Windows-specific attribute which emits idata sections for the items in the
// attached extern block,
// so they may be linked against without linking against an import library.
// To learn more, read https://github.com/rust-lang/rfcs/blob/master/text/2627-raw-dylib-kind.md
// `#[link_ordinal(n)]` allows Rust to link against DLLs that export symbols by ordinal rather
// than by name. As long as the ordinal matches, the name of the function in Rust is not
// required to match the name of the corresponding function in the exporting DLL.
// This test is a sanity check for this feature, done by comparing its output against expected
// output.
// See https://github.com/rust-lang/rust/pull/89025

//@ only-windows

use run_make_support::{cc, diff, extra_c_flags, is_windows_msvc, run, rustc};

// NOTE: build_native_dynamic lib is not used, as the special `def` files
// must be passed to the CC compiler.

fn main() {
    rustc().crate_type("lib").crate_name("raw_dylib_test").input("lib.rs").run();
    rustc().crate_type("bin").input("driver.rs").run();
    if is_windows_msvc() {
        cc().arg("-c").out_exe("exporter").input("exporter.c").run();
        cc().input("exporter.obj")
            .arg("exporter.def")
            .args(&["-link", "-dll", "-noimplib", "-out:exporter.dll"])
            .args(extra_c_flags())
            .run();
    } else {
        cc().arg("-v").arg("-c").out_exe("exporter.obj").input("exporter.c").run();
        cc().input("exporter.obj").arg("exporter.def").arg("-shared").output("exporter.dll").run();
    };
    let out = run("driver").stdout_utf8();
    diff().expected_file("output.txt").actual_text("actual", out).normalize(r#"\r"#, "").run();
}
