//@ needs-target-std
//@ needs-crate-type: dylib
//@ needs-dynamic-linking

// When a fake library was given to the compiler, it would
// result in an obscure and unhelpful error message. This test
// creates a false "foo" dylib, and checks that the standard error
// explains that the file exists, but that its metadata is incorrect.
// See https://github.com/rust-lang/rust/pull/88368

use run_make_support::{dynamic_lib_name, rfs, rustc};

fn main() {
    rfs::create_file(dynamic_lib_name("foo"));
    rustc()
        .crate_type("lib")
        .extern_("foo", dynamic_lib_name("foo"))
        .input("bar.rs")
        .run_fail()
        .assert_stderr_contains("invalid metadata files for crate `foo`");
}
