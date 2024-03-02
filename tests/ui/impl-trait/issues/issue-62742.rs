use std::marker::PhantomData;

fn _alias_check() {
    WrongImpl::foo(0i32);
    //~^ ERROR trait `Raw<_>` is not implemented for `RawImpl<_>`
    //~| ERROR trait `Raw<_>` is not implemented for `RawImpl<_>`
    WrongImpl::<()>::foo(0i32);
    //~^ ERROR trait `Raw<()>` is not implemented for `RawImpl<()>`
    //~| ERROR trait bounds were not satisfied
    CorrectImpl::foo(0i32);
}

pub trait Raw<T: ?Sized> {
    type Value;
}

pub type WrongImpl<T> = SafeImpl<T, RawImpl<T>>;

pub type CorrectImpl<T> = SafeImpl<[T], RawImpl<T>>;

pub struct RawImpl<T>(PhantomData<T>);

impl<T> Raw<[T]> for RawImpl<T> {
    type Value = T;
}

pub struct SafeImpl<T: ?Sized, A: Raw<T>>(PhantomData<(A, T)>);

impl<T: ?Sized, A: Raw<T>> SafeImpl<T, A> {
    pub fn foo(value: A::Value) {}
}

fn main() {}
