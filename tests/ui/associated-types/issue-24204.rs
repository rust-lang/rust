//@ check-pass

#![allow(dead_code)]

trait MultiDispatch<T> {
    type O;
}

trait Trait: Sized {
    type A: MultiDispatch<Self::B, O = Self>;
    type B;

    fn new<U>(u: U) -> <Self::A as MultiDispatch<U>>::O
    where
        Self::A: MultiDispatch<U>;
}

fn test<T: Trait<B = i32>>(b: i32) -> T
where
    T::A: MultiDispatch<i32>,
{
    T::new(b)
}

fn main() {}
