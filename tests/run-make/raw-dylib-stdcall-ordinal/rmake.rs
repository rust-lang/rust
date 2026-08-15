// `raw-dylib` is a Windows-specific attribute which emits idata sections for the items in the
// attached extern block,
// so they may be linked against without linking against an import library.
// To learn more, read https://github.com/rust-lang/rfcs/blob/master/text/2627-raw-dylib-kind.md
// Almost identical to `raw-dylib-link-ordinal`, but with the addition of calling conventions,
// such as stdcall.
// See https://github.com/rust-lang/rust/pull/90782

//@ only-x86
//@ only-windows
// Reason: this test specifically exercises a 32bit Windows calling convention.

use run_make_support::{cc, diff, is_windows_msvc, run, rustc};

// NOTE: build_native_dynamic lib is not used, as the special `def` files
// must be passed to the CC compiler.

fn main() {
    rustc().crate_type("lib").crate_name("raw_dylib_test").input("lib.rs").run();
    rustc().crate_type("bin").input("driver.rs").run();
    if is_windows_msvc() {
        cc().arg("-c").out_exe("exporter").input("exporter.c").run();
        cc().input("exporter.obj")
            .arg("exporter-msvc.def")
            .args(&["-link", "-dll", "-noimplib", "-out:exporter.dll"])
            .run();
    } else {
        cc().arg("-v").arg("-c").out_exe("exporter.obj").input("exporter.c").run();
        cc().input("exporter.obj")
            .arg("exporter-gnu.def")
            .arg("-shared")
            .output("exporter.dll")
            .run();
    };
    let out = run("driver").stdout_utf8();
    diff()
        .expected_file("expected_output.txt")
        .actual_text("actual", out)
        .normalize(r#"\r"#, "")
        .run();
}
