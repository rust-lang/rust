#![crate_type = "lib"]
#![feature(negative_impls)]
#![feature(rustc_attrs)]

pub trait Future {}

#[rustc_with_negative_coherence]
impl<E> !Future for Option<E> where E: Sized {}
