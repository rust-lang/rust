#![crate_type = "lib"]
#![feature(negative_impls)]
#![feature(rustc_attrs)]
#![feature(with_negative_coherence)]

pub trait Future {}

impl<E> !Future for Option<E> where E: Sized {}
