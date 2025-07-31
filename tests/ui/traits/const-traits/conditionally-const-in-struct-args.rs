#![feature(const_trait_impl)]

struct S;
#[const_trait]
trait Trait<const N: u32> {}

const fn f<
    T: Trait<
        {
            const fn g<U: [const] Trait<0>>() {}

            struct I<U: [const] Trait<0>>(U);
            //~^ ERROR `[const]` is not allowed here

            0
        },
    >,
>() {
}

pub fn main() {}
