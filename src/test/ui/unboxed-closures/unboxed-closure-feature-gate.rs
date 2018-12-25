// Check that parenthetical notation is feature-gated except with the
// `Fn` traits.

use std::marker;

trait Foo<A> {
    type Output;

    fn dummy(&self, a: A) { }
}

fn main() {
    let x: Box<Foo(isize)>;
    //~^ ERROR parenthetical notation is only stable when used with `Fn`-family

    // No errors with these:
    let x: Box<Fn(isize)>;
    let x: Box<FnMut(isize)>;
    let x: Box<FnOnce(isize)>;
}
