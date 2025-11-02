//@ needs-target-std

use run_make_support::{Rustc, cwd, diff, rust_lib_name, rustc};

fn rustc_with_common_args() -> Rustc {
    let mut rustc = rustc();
    rustc.remap_path_prefix(cwd(), "$DIR");
    rustc.edition("2018"); // Don't require `extern crate`
    rustc
}

fn main() {
    rustc_with_common_args()
        .input("foo-v1.rs")
        .crate_type("rlib")
        .crate_name("foo")
        .extra_filename("-v1")
        .metadata("-v1")
        .run();

    rustc_with_common_args()
        .input("foo-v2.rs")
        .crate_type("rlib")
        .crate_name("foo")
        .extra_filename("-v2")
        .metadata("-v2")
        .run();

    rustc_with_common_args()
        .input("re-export-foo.rs")
        .crate_type("rlib")
        .extern_("foo", rust_lib_name("foo-v2"))
        .run();

    let stderr = rustc_with_common_args()
        .input("main.rs")
        .extern_("foo", rust_lib_name("foo-v1"))
        .extern_("re_export_foo", rust_lib_name("re_export_foo"))
        .library_search_path(cwd())
        .ui_testing()
        .run_fail()
        .stderr_utf8();

    diff().expected_file("main.stderr").normalize(r"\\", "/").actual_text("(rustc)", &stderr).run();
}
