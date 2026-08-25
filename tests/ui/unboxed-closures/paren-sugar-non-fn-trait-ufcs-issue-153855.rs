#![feature(rustc_attrs, unboxed_closures)]
#![allow(internal_features)]

// Regression test for #153855: diagnostics should not ICE when a
// `#[rustc_paren_sugar]` trait is used with parenthesized syntax but does not
// have an `Fn`-family generic layout.
#[rustc_paren_sugar]
trait Tr<'a, 'b, T> {
    fn method() {}
}

fn main() {
    <u8 as Tr(&u8)>::method;
    //~^ ERROR associated item constraints are not allowed here
    //~| ERROR the trait bound `u8: Tr<'_, '_, (&u8,)>` is not satisfied
}
