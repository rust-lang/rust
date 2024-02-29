trait X<'a>
where
    for<'b> <Self as X<'b>>::U: Clone,
{
    type U: ?Sized;
}
fn f<'a, T: X<'a> + ?Sized>(x: &<T as X<'a>>::U) {
    //~^ ERROR trait `Clone` is not implemented for `<T as X<'_>>::U`
    <<T as X<'_>>::U>::clone(x);
    //~^ ERROR trait `Clone` is not implemented for `<T as X<'_>>::U`
    //~| ERROR trait `Clone` is not implemented for `<T as X<'_>>::U`
    //~| ERROR trait `Clone` is not implemented for `<T as X<'_>>::U`
}

pub fn main() {
    f::<dyn X<'_, U = str>>("abc");
}
