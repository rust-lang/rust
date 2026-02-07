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
