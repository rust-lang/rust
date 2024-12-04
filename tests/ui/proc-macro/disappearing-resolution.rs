// Regression test for issue #64803 (initial attribute resolution can disappear later).

//@ proc-macro: test-macros.rs

#[macro_use]
extern crate test_macros;

mod m {
    use test_macros::Empty;
}
use m::Empty; //~ ERROR derive macro import `Empty` is private

// To resolve `empty_helper` we need to resolve `Empty`.
// During initial resolution `use m::Empty` introduces no entries, so we proceed to `macro_use`,
// successfully resolve `Empty` from there, and then resolve `empty_helper` as its helper.
// During validation `use m::Empty` introduces a `Res::Err` stub, so `Empty` resolves to it,
// and `empty_helper` can no longer be resolved.
#[empty_helper] //~ ERROR cannot find attribute `empty_helper` in this scope
#[derive(Empty)]
struct S;

fn main() {}
