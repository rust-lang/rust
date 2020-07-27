trait X<'a>
where
    for<'b> <Self as X<'b>>::U: Clone,
{
    type U: ?Sized;
    fn f(&self, x: &Self::U) {
        <Self::U>::clone(x);
    }
}

impl X<'_> for u32
where
    for<'b> <Self as X<'b>>::U: Clone,
{
    type U = str;
}

fn main() {
    1u32.f("abc");
    //~^ ERROR no method named `f` found for type `u32` in the current scope
}
