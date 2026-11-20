//@ needs-target-std
//
// Trying to access mid-level internal representation (MIR) in statics
// used to cause an internal compiler error (ICE), now handled as a proper
// error since #100211. This test checks that the correct error is printed
// during the linking process, and not the ICE.
// See https://github.com/rust-lang/rust/issues/85401

use run_make_support::{bin_name, rust_lib_name, rustc};

fn main() {
    rustc()
        .crate_type("rlib")
        .crate_name("foo")
        .arg("-Crelocation-model=pic")
        .edition("2018")
        .input("foo.rs")
        .arg("-Zalways-encode-mir=yes")
        .emit("metadata")
        .output("libfoo.rmeta")
        .run();
    rustc()
        .crate_type("rlib")
        .crate_name("bar")
        .arg("-Crelocation-model=pic")
        .edition("2018")
        .input("bar.rs")
        .output(rust_lib_name("bar"))
        .extern_("foo", "libfoo.rmeta")
        .run();
    rustc()
        .crate_type("bin")
        .crate_name("baz")
        .arg("-Crelocation-model=pic")
        .edition("2018")
        .input("baz.rs")
        .output(bin_name("baz"))
        .extern_("bar", rust_lib_name("bar"))
        .run_fail()
        .assert_stderr_contains(
            "crate `foo` required to be available in rlib format, but was not found in this form",
        )
        .assert_stdout_not_contains("internal compiler error");
}
