#![feature(const_trait_impl, impl_trait_in_bindings)]

struct S;
#[const_trait]
trait Trait<const N: u32> {}

impl const Trait<0> for () {}

const fn f<
    T: Trait<
        {
            const fn g<U: [const] Trait<0>>() {}

            struct I<U: [const] Trait<0>>(U);
            //~^ ERROR `[const]` is not allowed here

            let x: &impl [const] Trait<0> = &();
            //~^ ERROR `[const]` is not allowed here

            0
        },
    >,
>(x: &T) {
    // Should be allowed here
    let y: &impl [const] Trait<0> = x;
}

pub fn main() {}
