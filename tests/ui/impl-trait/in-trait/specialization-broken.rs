// check-pass

#![feature(specialization)]
#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

trait Foo {
    fn bar(&self) -> impl Sized;
}

default impl<U> Foo for U
where
    U: Copy,
{
    fn bar(&self) -> U {
        *self
    }
}

impl Foo for i32 {}

fn main() {
    1i32.bar();
}
