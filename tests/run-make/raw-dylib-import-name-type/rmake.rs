// `raw-dylib` is a Windows-specific attribute which emits idata sections for the items in the
// attached extern block,
// so they may be linked against without linking against an import library.
// To learn more, read https://github.com/rust-lang/rfcs/blob/master/text/2627-raw-dylib-kind.md
// This test uses this feature alongside `import_name_type`, which allows for customization
// of how Windows symbols will be named. A sanity check of this feature is done by comparison
// with expected output.
// See https://github.com/rust-lang/rust/pull/100732

//@ only-x86
//@ only-windows
// Reason: this test specifically exercises a 32bit Windows calling convention.

use run_make_support::{cc, diff, is_windows_msvc, run, rustc};

// NOTE: build_native_dynamic lib is not used, as the special `def` files
// must be passed to the CC compiler.

fn main() {
    rustc().crate_type("bin").input("driver.rs").run();
    if is_windows_msvc() {
        cc().arg("-c").out_exe("extern").input("extern.c").run();
        cc().input("extern.obj")
            .arg("extern.msvc.def")
            .args(&["-link", "-dll", "-noimplib", "-out:extern.dll"])
            .run();
    } else {
        cc().arg("-v").arg("-c").out_exe("extern.obj").input("extern.c").run();
        cc().input("extern.obj")
            .arg("extern.gnu.def")
            .args(&["--no-leading-underscore", "-shared"])
            .output("extern.dll")
            .run();
    };
    let out = run("driver").stdout_utf8();
    diff().expected_file("output.txt").actual_text("actual", out).normalize(r#"\r"#, "").run();
}
