// Regression test for <https://github.com/rust-lang/rust/issues/151461>.
//
// The next solver normalizes `TypeOutlives` goals before registering them as region obligations.
// This should not ICE when we already emitted an error for an undeclared lifetime.
//@ compile-flags: -Znext-solver=globally

trait X<'a> {
    type U: ?Sized;
}

impl X<'_> for u32
where
    for<'b> <Self as X<'b>>::U: 'a,
    //~^ ERROR use of undeclared lifetime name `'a`
{
    type U = str;
}

fn main() {}
