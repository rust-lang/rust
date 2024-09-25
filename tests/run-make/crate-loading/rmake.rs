//@ only-linux
//@ ignore-wasm32
//@ ignore-wasm64
// ignore-tidy-linelength

use run_make_support::{rust_lib_name, rustc};

fn main() {
    rustc().input("multiple-dep-versions-1.rs").run();
    rustc().input("multiple-dep-versions-2.rs").extra_filename("2").metadata("2").run();
    rustc()
        .input("multiple-dep-versions-3.rs")
        .extern_("dependency", rust_lib_name("dependency2"))
        .run();

    rustc()
        .input("multiple-dep-versions.rs")
        .extern_("dependency", rust_lib_name("dependency"))
        .extern_("dep_2_reexport", rust_lib_name("foo"))
        .run_fail()
        .assert_stderr_contains(
            r#"error[E0277]: the trait bound `dep_2_reexport::Type: Trait` is not satisfied
  --> multiple-dep-versions.rs:7:18
   |
7  |     do_something(Type);
   |     ------------ ^^^^ the trait `Trait` is not implemented for `dep_2_reexport::Type`
   |     |
   |     required by a bound introduced by this call
   |
help: there are multiple different versions of crate `dependency` in the dependency graph
  --> multiple-dep-versions.rs:1:1
   |
1  | extern crate dep_2_reexport;
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ one version of crate `dependency` is used here, as a dependency of crate `foo`
2  | extern crate dependency;
   | ^^^^^^^^^^^^^^^^^^^^^^^^ one version of crate `dependency` is used here, as a direct dependency of the current crate"#,
        )
        .assert_stderr_contains(
            r#"
3  | pub struct Type(pub i32);
   | ^^^^^^^^^^^^^^^ this type implements the required trait
4  | pub trait Trait {
   | --------------- this is the required trait"#,
        )
        .assert_stderr_contains(
            r#"
3  | pub struct Type;
   | ^^^^^^^^^^^^^^^ this type doesn't implement the required trait"#,
        )
        .assert_stderr_contains(
            r#"
error[E0599]: no method named `foo` found for struct `dep_2_reexport::Type` in the current scope
 --> multiple-dep-versions.rs:8:10
  |
8 |     Type.foo();
  |          ^^^ method not found in `Type`
  |
note: there are multiple different versions of crate `dependency` in the dependency graph"#,
        )
        .assert_stderr_contains(
            r#"
4 | pub trait Trait {
  | ^^^^^^^^^^^^^^^ this is the trait that is needed
5 |     fn foo(&self);
  |     -------------- the method is available for `dep_2_reexport::Type` here
  |
 ::: multiple-dep-versions.rs:4:18
  |
4 | use dependency::{Trait, do_something};
  |                  ----- `Trait` imported here doesn't correspond to the right version of crate `dependency`"#,
        )
        .assert_stderr_contains(
            r#"
4 | pub trait Trait {
  | --------------- this is the trait that was imported"#,
        )
        .assert_stderr_contains(
            r#"
error[E0599]: no function or associated item named `bar` found for struct `dep_2_reexport::Type` in the current scope
 --> multiple-dep-versions.rs:9:11
  |
9 |     Type::bar();
  |           ^^^ function or associated item not found in `Type`
  |
note: there are multiple different versions of crate `dependency` in the dependency graph"#,
        )
        .assert_stderr_contains(
            r#"
4 | pub trait Trait {
  | ^^^^^^^^^^^^^^^ this is the trait that is needed
5 |     fn foo(&self);
6 |     fn bar();
  |     --------- the associated function is available for `dep_2_reexport::Type` here
  |
 ::: multiple-dep-versions.rs:4:18
  |
4 | use dependency::{Trait, do_something};
  |                  ----- `Trait` imported here doesn't correspond to the right version of crate `dependency`"#,
        );
}
