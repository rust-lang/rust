// Regression test for issue #119924.
#![feature(const_trait_impl, effects)]
#![allow(incomplete_features)]

pub struct Type;

#[const_trait]
trait Trait0 {
    fn provided() {
        impl Type {
            fn perform<T: ~const Trait0>() {}
            //~^ ERROR `~const` is not allowed here
        }
    }
}

struct Expr<const N: u32>;

#[const_trait]
trait Trait1 {
    fn required(_: Expr<{
        impl Type {
            fn compute<T: ~const Trait1>() {}
            //~^ ERROR `~const` is not allowed here
        }
        0
    }>);
}

#[const_trait]
trait Trait2<const N: u32> {}

const fn operate<T: Trait2<{
    struct I<U: ~const Trait2<0>>(U);
    //~^ ERROR `~const` is not allowed here
    0
}>>() {}

fn main() {}
