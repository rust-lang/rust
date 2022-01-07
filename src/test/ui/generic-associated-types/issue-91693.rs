// check-fail
// FIXME(generic_associated_types): This *should* pass, but fails likely due to
// leak check/universe related things

#![feature(generic_associated_types)]

pub trait LendingIterator {
    type Item<'a>
    where
        Self: 'a;
    fn next(&mut self) -> Option<Self::Item<'_>>;

    fn for_each<F>(mut self, mut f: F)
    where
        Self: Sized,
        F: FnMut(Self::Item<'_>),
    {
        while let Some(item) = self.next() {
            f(item)
        }
    }
}

pub struct Mutator<T>(T);

impl<T> LendingIterator for Mutator<T> {
    type Item<'a> = &'a mut T
    where
        Self: 'a;
    fn next(&mut self) -> Option<Self::Item<'_>> {
        Some(&mut self.0)
    }
}

pub fn bar<T>(m: Mutator<T>) {
    m.for_each(|_: &mut T| {});
    //~^ ERROR the parameter type
    //~| ERROR the parameter type
}

fn main() {}
