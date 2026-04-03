// Regression test for #132055
// This used to ICE with "assertion failed: !t.has_escaping_bound_vars()"
// when using return_type_notation with a trait that has a lifetime parameter
// but omitting the lifetime in the bound. Now correctly reports E0106.

#![feature(return_type_notation)]

trait A<'a> {
    fn method() -> impl Sized;
}

fn foo<T: A<method(..): Send>>() {}
//~^ ERROR missing lifetime specifier

fn main() {}
