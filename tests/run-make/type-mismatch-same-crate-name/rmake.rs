//@ needs-target-std
//
// When a compilation failure deals with seemingly identical types, some helpful
// errors should be printed.
// The main use case of this error is when there are two crates
// (generally different versions of the same crate) with the same name
// causing a type mismatch. In this test, one of the crates
// is only introduced as an indirect dependency and the type is accessed via a reexport.
// See https://github.com/rust-lang/rust/pull/42826

use run_make_support::{rust_lib_name, rustc};

fn main() {
    rustc().crate_type("rlib").input("crateA.rs").metadata("-1").extra_filename("-1").run();
    rustc().crate_type("rlib").input("crateA.rs").metadata("-2").extra_filename("-2").run();
    rustc()
        .crate_type("rlib")
        .input("crateB.rs")
        .extern_("crateA", rust_lib_name("crateA-1"))
        .run();
    rustc()
        .input("crateC.rs")
        .extern_("crateA", rust_lib_name("crateA-2"))
        .run_fail()
        .assert_stderr_contains("mismatched types")
        .assert_stderr_contains("crateB::try_foo(foo2);")
        .assert_stderr_contains("different versions of crate `crateA`")
        .assert_stderr_contains("crateB::try_bar(bar2);")
        .assert_stderr_contains("expected trait `crateA::bar::Bar`, found trait `Bar`")
        .assert_stderr_contains("different versions of crate `crateA`");
}
