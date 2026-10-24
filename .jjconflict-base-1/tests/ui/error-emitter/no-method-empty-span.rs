//! Regression test for <https://github.com/rust-lang/rust/issues/41652>.
//! This used to ICE on empty string indexing internally, which came from
//! empty code span in error message, when method cannot be found.
//@ aux-build:no-method-empty-span.rs

extern crate no_method_empty_span;

struct S;

impl no_method_empty_span::Tr for S {
    fn f() {
        3.f()
        //~^ ERROR can't call method `f` on ambiguous numeric type `{integer}`
    }
}

fn main() {}
