trait X<'a>
where
    for<'b> <Self as X<'b>>::U: Clone,
{
    type U: ?Sized;
}
fn f<'a, T: X<'a> + ?Sized>(x: &<T as X<'a>>::U) {
    //~^ ERROR the trait bound `for<'b> <T as X<'b>>::U: Clone` is not satisfied
    <<T as X<'_>>::U>::clone(x);
    //~^ ERROR the trait bound `for<'b> <T as X<'b>>::U: Clone` is not satisfied
    //~| ERROR the trait bound `for<'b> <T as X<'b>>::U: Clone` is not satisfied
    //~| ERROR the trait bound `for<'b> <T as X<'b>>::U: Clone` is not satisfied
    //~| ERROR the trait bound `<T as X<'_>>::U: Clone` is not satisfied
}

pub fn main() {
    f::<dyn X<'_, U = str>>("abc");
}
