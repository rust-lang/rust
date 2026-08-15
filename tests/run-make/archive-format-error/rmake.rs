// Regression test for https://github.com/rust-lang/rust/issues/148217
// BSD format archive on a Linux target should emit a format mismatch warning.

//@ ignore-cross-compile
//@ only-linux

use run_make_support::{cc, llvm_ar, path, rfs, rustc, static_lib_name};

fn main() {
    rfs::create_dir("archive");

    cc().arg("-c").input("native.c").output("archive/native.o").run();
    let bsd_archive = path("archive").join(static_lib_name("native_bsd"));
    llvm_ar().arg("rcus").arg("--format=bsd").output_input(&bsd_archive, "archive/native.o").run();
    rustc()
        .input("lib.rs")
        .crate_type("rlib")
        .library_search_path("archive")
        .arg("-lstatic=native_bsd")
        .run()
        .assert_stderr_contains("was built as BSD format, but the target expects GNU");
}
