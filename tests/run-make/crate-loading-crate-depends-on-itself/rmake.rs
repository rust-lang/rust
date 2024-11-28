//@ only-linux
//@ ignore-wasm32
//@ ignore-wasm64
// ignore-tidy-linelength

// Verify that if the current crate depends on a different version of the same crate, *and* types
// and traits of the different versions are mixed, we produce diagnostic output and not an ICE.
// #133563

use run_make_support::{rust_lib_name, rustc};

fn main() {
    rustc().input("foo-prev.rs").run();

    rustc()
        .extra_filename("current")
        .metadata("current")
        .input("foo-current.rs")
        .extern_("foo", rust_lib_name("foo"))
        .run_fail()
        .assert_stderr_contains(r#"
note: there are multiple different versions of crate `foo` in the dependency graph
  --> foo-current.rs:7:1
   |
4  | extern crate foo;
   | ----------------- one version of crate `foo` is used here, as a direct dependency of the current crate
5  |
6  | pub struct Struct;
   | ----------------- this type implements the required trait
7  | pub trait Trait {}
   | ^^^^^^^^^^^^^^^ this is the required trait"#);
}
