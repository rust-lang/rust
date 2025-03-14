//@ check-pass

#![feature(const_trait_impl, const_ops)]

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
    const _: S<i32> = twice(S(PhantomData));
    let _ = twice(S(PhantomData::<i32>));
}
