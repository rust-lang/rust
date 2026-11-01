//@ needs-target-std
//
// The compilation error caused by calling on an unimported crate
// should have a suggestion to write, say, crate::bar::Foo instead
// of just bar::Foo. However, this suggestion used to only appear for
// extern crate statements, not crate struct. After this was fixed in #51456,
// this test checks that the correct suggestion is printed no matter what.
// See https://github.com/rust-lang/rust/issues/51212

use run_make_support::{rust_lib_name, rustc};

fn main() {
    rustc().input("ep-nested-lib.rs").run();
    rustc()
        .input("use-suggestions.rs")
        .edition("2018")
        .extern_("ep_nested_lib", rust_lib_name("ep_nested_lib"))
        .run_fail()
        .assert_stderr_contains("use ep_nested_lib::foo::bar::Baz");
}
