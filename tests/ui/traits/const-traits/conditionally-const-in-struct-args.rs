//@ compile-flags: -Znext-solver
//@ known-bug: #132067
//@ check-pass

#![feature(const_trait_impl)]

struct S;
#[const_trait]
trait Trait<const N: u32> {}

const fn f<
    T: Trait<
        {
            struct I<U: [const] Trait<0>>(U);
            0
        },
    >,
>() {
}

pub fn main() {}
