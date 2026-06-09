//! Regression test for https://github.com/rust-lang/rust/issues/13775

//@ edition: 2015
//@ check-pass

trait Foo {
    #[allow(anonymous_parameters)]
    fn bar(&self, isize) {}
}

fn main() {}
