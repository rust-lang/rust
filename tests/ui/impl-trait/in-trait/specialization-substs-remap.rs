//@ check-pass

#![feature(specialization)]
#![allow(incomplete_features)]

pub trait Foo {
    fn bar(&self) -> impl Sized;
}

impl<U> Foo for U
where
    U: Copy,
{
    #[expect(refining_impl_trait)]
    fn bar(&self) -> U {
        *self
    }
}

impl Foo for i32 {}

fn main() {
    let _: i32 = 1i32.bar();
}
