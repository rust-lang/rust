// run-pass

#![allow(incomplete_features)]
#![feature(const_trait_impl)]
#![feature(const_fn)]

use std::marker::PhantomData;

struct S<T>(PhantomData<T>);

impl<T> Copy for S<T> {}
impl<T> Clone for S<T> {
    fn clone(&self) -> Self {
        S(PhantomData)
    }
}

impl<T> const std::ops::Add for S<T> {
    type Output = Self;

    fn add(self, _: Self) -> Self {
        S(std::marker::PhantomData)
    }
}

const fn twice<T: std::ops::Add>(arg: S<T>) -> S<T> {
    arg + arg
}

fn main() {
    let _ = twice(S(PhantomData::<i32>));
}
