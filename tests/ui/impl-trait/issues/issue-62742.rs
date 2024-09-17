use std::marker::PhantomData;

fn a() {
    WrongImpl::foo(0i32);
    //~^ ERROR the function or associated item `foo` exists for struct `SafeImpl<_, RawImpl<_>>`, but its trait bounds were not satisfied
    //~| ERROR the trait bound `RawImpl<_>: Raw<_>` is not satisfied
}

fn b() {
    WrongImpl::<()>::foo(0i32);
    //~^ ERROR the trait bound `RawImpl<()>: Raw<()>` is not satisfied
    //~| ERROR trait bounds were not satisfied
}

fn c() {
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
