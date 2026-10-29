//@ check-pass
// Regression test for issue #63677 - ensure that
// coherence checking can properly handle 'impl trait'
// in type aliases
#![feature(type_alias_impl_trait)]

pub trait Trait {}
pub struct S1<T>(T);
pub struct S2<T>(T);

pub type T1 = impl Trait;
pub type T2 = S1<T1>;
pub type T3 = S2<T2>;

impl<T> Trait for S1<T> {}
impl<T: Trait> S2<T> {}
impl T3 {}

#[define_opaque(T1)]
pub fn use_t1() -> T1 {
    S1(())
}

fn main() {}
