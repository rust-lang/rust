trait Cycle: Sized {
    type Next: Cycle<Next = Self>;
}

impl<T> Cycle for Box<T> {
    type Next = Vec<T>;
}

impl<T> Cycle for Vec<T> {
    type Next = Box<T>;
}

trait X<'a, T: Cycle + for<'b> X<'b, T>>
where
    for<'b> <T as X<'b, T>>::U: Clone,
    for<'b> T::Next: X<'b, T::Next>,
    for<'b> <T::Next as X<'b, T::Next>>::U: Clone,
{
    type U: ?Sized;
    fn f(x: &<T as X<'_, T>>::U) {
        <<T as X<'_, T>>::U>::clone(x);
    }
}

impl<S, T> X<'_, Vec<T>> for S {
    type U = str;
    //~^ ERROR the trait bound `str: Clone` is not satisfied
}

impl<S, T> X<'_, Box<T>> for S {
    type U = str;
    //~^ ERROR the trait bound `str: Clone` is not satisfied
}

pub fn main() {
    <i32 as X<Box<i32>>>::f("abc");
    //~^ ERROR the trait bound `str: Clone` is not satisfied
}
