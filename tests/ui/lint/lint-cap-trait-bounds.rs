// Regression test for https://github.com/rust-lang/rust/issues/43134

//@ check-pass
//@ compile-flags: --cap-lints allow

type Foo<T: Clone> = Option<T>;

fn main() {}
