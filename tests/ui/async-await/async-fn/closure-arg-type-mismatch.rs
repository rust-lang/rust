//! Regression test for <https://github.com/rust-lang/rust/issues/155636>
//@ edition:2021

fn foo(_: impl AsyncFn(&mut i32)) {}

fn main() {
    foo(|_: i32| async {});
    //~^ ERROR type mismatch in closure arguments
}
