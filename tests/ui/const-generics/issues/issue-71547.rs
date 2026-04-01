//@ check-pass

#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

pub trait GetType<const N: &'static str> {
    type Ty;
    fn get(&self) -> &Self::Ty;
}

pub fn get_val<T>(value: &T) -> &T::Ty
where
    T: GetType<"hello">,
{
    value.get()
}

fn main() {}
