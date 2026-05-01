//@ edition: 2021
// Regression test for <https://github.com/rust-lang/rust/issues/40066>.

mod foo {
    pub(in crate::bar) struct Foo;
    //~^ ERROR visibilities can only be restricted to ancestor modules
}

mod bar {
    pub(in crate::foo) struct Bar;
    //~^ ERROR visibilities can only be restricted to ancestor modules
}

fn main() {}
