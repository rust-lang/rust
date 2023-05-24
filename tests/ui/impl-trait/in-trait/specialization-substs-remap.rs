// check-pass

#![feature(specialization)]
#![feature(return_position_impl_trait_in_trait)]
#![feature(refine)]
#![allow(incomplete_features)]

trait Foo {
    fn bar(&self) -> impl Sized;
}

impl<U> Foo for U
where
    U: Copy,
{
    #[refine]
    fn bar(&self) -> U {
        *self
    }
}

impl Foo for i32 {}

fn main() {
    let _: i32 = 1i32.bar();
}
