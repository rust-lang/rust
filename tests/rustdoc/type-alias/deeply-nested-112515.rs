// Regression test for <https://github.com/rust-lang/rust/issues/112515>.
// It's to ensure that this code doesn't have infinite loop in rustdoc when
// trying to retrive type alias implementations.

// ignore-tidy-linelength

pub type Boom = S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<S<u64, u8>, ()>, ()>, ()>, u8>, ()>, u8>, ()>, u8>, u8>, ()>, ()>, ()>, u8>, u8>, u8>, ()>, ()>, u8>, ()>, ()>, ()>, u8>, u8>, ()>, ()>, ()>, ()>, ()>, u8>, ()>, ()>, u8>, ()>, ()>, ()>, u8>, ()>, ()>, u8>, u8>, u8>, u8>, ()>, u8>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>;
pub struct S<T, U>(T, U);

pub trait A {}

pub trait B<T> {
    type P;
}

impl A for u64 {}

impl<T, U> A for S<T, U> {}

impl<T> B<u8> for S<T, ()>
where
    T: B<u8>,
    <T as B<u8>>::P: A,
{
    type P = ();
}

impl<T: A, U, V> B<T> for S<U, V> {
    type P = ();
}
