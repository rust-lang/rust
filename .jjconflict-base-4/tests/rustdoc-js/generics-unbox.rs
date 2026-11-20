#![feature(rustdoc_internals)]

#[doc(search_unbox)]
pub struct Out<A, B = ()> {
    a: A,
    b: B,
}

#[doc(search_unbox)]
pub struct Out1<A, const N: usize> {
    a: [A; N],
}

#[doc(search_unbox)]
pub struct Out2<A, const N: usize> {
    a: [A; N],
}

#[doc(search_unbox)]
pub struct Out3<A, B> {
    a: A,
    b: B,
}

#[doc(search_unbox)]
pub struct Out4<A, B> {
    a: A,
    b: B,
}

#[doc(search_unbox)]
pub struct Inside<T>(T);

pub fn alpha<const N: usize, T>(_: Inside<T>) -> Out<Out1<T, N>, Out2<T, N>> {
    loop {}
}

pub fn beta<T, U>(_: Inside<T>) -> Out<Out3<T, U>, Out4<U, T>> {
    loop {}
}

pub fn gamma<T, U>(_: Inside<T>) -> Out<Out3<U, T>, Out4<T, U>> {
    loop {}
}

pub fn delta(_: i32) -> Epsilon<Sigma> {
    loop {}
}

#[doc(search_unbox)]
pub struct Theta<T>(T);

#[doc(search_unbox)]
pub type Epsilon<T> = Theta<T>;

pub struct Sigma;
