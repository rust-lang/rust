//@ compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/rust/issues/161527>

trait Z<'a, T: ?Sized>
where
    T: Z<'a, ()>, //~ ERROR: the trait bound `(): Z<'a, ()>` is not satisfied
    for<'b> <T as Z<'b, ()>>::W: 'a,
{
    type W;
}

fn main() {}
