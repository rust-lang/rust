// We used to ICE here while trying to synthesize auto trait impls.
//@ check-pass

#![feature(const_default)]
#![feature(const_trait_impl)]

pub struct Inner<T>(T);

pub struct Outer<T>(Inner<T>);

impl<T> Unpin for Inner<T>
where
    T: const std::default::Default,
{}
