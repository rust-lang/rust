// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// #11612
// We weren't updating the auto adjustments with all the resolved
// type information after type check.

// pretty-expanded FIXME #23616

trait A { fn dummy(&self) { } }

struct B<'a, T:'a> {
    f: &'a T
}

impl<'a, T> A for B<'a, T> {}

fn foo(_: &dyn A) {}

fn bar<G>(b: &B<G>) {
    foo(b);       // Coercion should work
    foo(b as &dyn A); // Explicit cast should work as well
}

fn main() {}
