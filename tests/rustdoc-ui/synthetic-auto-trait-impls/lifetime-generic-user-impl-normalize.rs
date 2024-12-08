// We used to ICE here while trying to synthesize auto trait impls.
// issue: 112242
//@ check-pass
//@ compile-flags: -Znormalize-docs

pub trait MyTrait<'a> {
    type MyItem;
}
pub struct Inner<Q>(Q);
pub struct Outer<Q>(Inner<Q>);

impl<'a, Q> std::marker::Unpin for Inner<Q>
where
    Q: MyTrait<'a>,
    <Q as MyTrait<'a>>::MyItem: Copy,
{
}
