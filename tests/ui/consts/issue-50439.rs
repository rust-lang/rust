#![feature(specialization)]
#![allow(incomplete_features)]

pub trait ReflectDrop {
    const REFLECT_DROP: bool = false;
}

impl<T> ReflectDrop for T where T: Clone {}

pub trait PinDropInternal {
    fn is_valid()
    where
        Self: ReflectDrop;
}

struct Bears<T>(T);

default impl<T> ReflectDrop for Bears<T> {}

impl<T: Sized> PinDropInternal for Bears<T> {
    fn is_valid()
    where
        Self: ReflectDrop,
    {
        let _ = [(); 0 - !!(<Bears<T> as ReflectDrop>::REFLECT_DROP) as usize];
        //~^ ERROR constant expression depends on a generic parameter
        //~| ERROR constant expression depends on a generic parameter
    }
}

fn main() {}
