#![feature(rustdoc_internals)]
#![allow(unused_unconstructable_pub_structs)]

#[doc(search_unbox)]
pub struct Inside<T>(T);

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
