trait X<'a>
where
    for<'b> <Self as X<'b>>::U: Clone,
{
    type U: ?Sized;
    fn f(&self, x: &Self::U) {
        <Self::U>::clone(x);
    }
}

impl X<'_> for u32 //~ ERROR overflow evaluating the requirement `for<'b> u32: X<'b>`
where
    for<'b> <Self as X<'b>>::U: Clone,
{
    type U = str;
}

fn main() {
    1u32.f("abc");
}
