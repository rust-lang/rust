//! Regression test for <https://github.com/rust-lang/rust/issues/141265>.
//! Make sure expressions in top-level guard patterns are only resolved once.

fn main() {
    for
    else if b 0 {}
    //~^ ERROR expected identifier, found keyword `else`
    //~| ERROR missing `in` in `for` loop
    //~| ERROR cannot find value `b` in this scope
    //~| ERROR guard patterns are experimental
    //~| ERROR `{integer}` is not an iterator
}
