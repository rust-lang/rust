trait X<'a>
where
    for<'b> <Self as X<'b>>::U: Clone,
{
    type U: ?Sized;
    fn f(&self, x: &Self::U) {
        <Self::U>::clone(x);
    }
}

impl X<'_> for i32 {
    type U = str;
    //~^ ERROR the trait bound `for<'b> <i32 as X<'b>>::U: Clone`
}

fn main() {
    1i32.f("abc");
}
