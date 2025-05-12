//@ needs-symlink
//@ ignore-cross-compile

// Tests that the multiple candidate dependencies diagnostic prints relative
// paths if a relative library path was passed in.

use run_make_support::{bare_rustc, diff, rfs, rustc};

fn main() {
    // Check that relative paths are preserved in the diagnostic
    rfs::create_dir("mylibs");
    rustc().input("crateresolve1-1.rs").out_dir("mylibs").extra_filename("-1").run();
    rustc().input("crateresolve1-2.rs").out_dir("mylibs").extra_filename("-2").run();
    check("./mylibs");

    // Check that symlinks aren't followed when printing the diagnostic
    rfs::rename("mylibs", "original");
    rfs::symlink_dir("original", "mylibs");
    check("./mylibs");
}

fn check(library_path: &str) {
    let out = rustc()
        .input("multiple-candidates.rs")
        .library_search_path(library_path)
        .ui_testing()
        .run_fail()
        .stderr_utf8();
    diff()
        .expected_file("multiple-candidates.stderr")
        .normalize(r"\\", "/")
        .actual_text("(rustc)", &out)
        .run();
}
