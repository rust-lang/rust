trait X<'a, T>
where
    for<'b> (T,): X<'b, T>,
    for<'b> <(T,) as X<'b, T>>::U: Clone,
{
    type U: ?Sized;
    fn f(x: &<(T,) as X<'_, T>>::U) {
        <<(T,) as X<'_, T>>::U>::clone(x);
    }
}

impl<S, T> X<'_, T> for (S,) {
    type U = str;
    //~^ ERROR the trait bound `str: Clone` is not satisfied
}

pub fn main() {
    <(i32,) as X<i32>>::f("abc");
    //~^ ERROR the trait bound `str: Clone` is not satisfied
}
