//@ ignore-cross-compile (the doctest binary is executed)

// See <https://github.com/rust-lang/rust/issues/153318>.

use run_make_support::{build_native_static_lib, cwd, rustdoc};

fn main() {
    rustdoc()
        .input("lib.rs")
        .out_dir("doc")
        .arg("-lcfoo")
        .arg("-O")
        .arg("-g")
        .arg("--explain")
        .arg("E0001")
        .run();

    build_native_static_lib("cfoo");
    rustdoc().input("lib.rs").arg("--test").arg("-L").arg(cwd()).arg("-lstatic=cfoo").run();

    // Merged doctests link the native library through the bundle rlib, not the runner.
    rustdoc()
        .input("lib.rs")
        .edition("2024")
        .arg("--test")
        .arg("-Zunstable-options")
        .arg("--merge-doctests=yes")
        .arg("-L")
        .arg(cwd())
        .arg("-lstatic:+whole-archive=cfoo")
        .run();
}
