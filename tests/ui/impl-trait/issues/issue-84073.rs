use std::marker::PhantomData;

pub trait StatefulFuture<S> {}
pub struct Never<T>(PhantomData<T>);
impl<T> StatefulFuture<T> for Never<T> {}

pub struct RaceBuilder<F, S> {
    future: F,
    _phantom: PhantomData<S>,
}

impl<T, F> RaceBuilder<T, F>
where
    F: StatefulFuture<Option<T>>,
{
    pub fn when(self) {}
}

pub struct Race<T, R> {
    race: R,
    _phantom: PhantomData<T>,
}

impl<T, R> Race<T, R>
where
    R: Fn(RaceBuilder<T, Never<T>>),
{
    pub fn new(race: R) {}
}

fn main() {
    Race::new(|race| race.when());
    //~^ ERROR overflow assigning `_` to `Option<_>`
}
