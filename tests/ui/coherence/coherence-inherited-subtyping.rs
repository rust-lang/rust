// Test that two distinct impls which match subtypes of one another
// yield coherence errors (or not) depending on the variance.
//
// Note: This scenario is currently accepted, but as part of the
// universe transition (#56105) may eventually become an error.

struct Foo<T> {
    t: T,
}

impl Foo<for<'a, 'b> fn(&'a u8, &'b u8) -> &'a u8> {
    fn method1(&self) {} //~ ERROR duplicate definitions with name `method1`
}

impl Foo<for<'a> fn(&'a u8, &'a u8) -> &'a u8> {
    fn method1(&self) {}
}

fn main() {}
