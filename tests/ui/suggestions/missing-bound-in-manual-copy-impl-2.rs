//@ run-rustfix
#![allow(dead_code)]

#[derive(Clone)]
struct Wrapper<T>(T);

struct OnlyCopyIfDisplay<T>(std::marker::PhantomData<T>);

impl<T: std::fmt::Display> Clone for OnlyCopyIfDisplay<T> {
    fn clone(&self) -> Self {
        OnlyCopyIfDisplay(std::marker::PhantomData)
    }
}

impl<T: std::fmt::Display> Copy for OnlyCopyIfDisplay<T> {}

impl<S> Copy for Wrapper<OnlyCopyIfDisplay<S>> {}
//~^ ERROR the trait `Copy` cannot be implemented for this type

fn main() {}
