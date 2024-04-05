// We used to ICE here while trying to synthesize auto trait impls.
// issue: 123370
//@ check-pass

pub struct Inner<'a, Q>(&'a (), Q);

pub struct Outer<'a, Q>(Inner<'a, Q>);

impl<'a, Q: Trait<'a>> std::marker::Unpin for Inner<'static, Q> {}

pub trait Trait<'a> {}
