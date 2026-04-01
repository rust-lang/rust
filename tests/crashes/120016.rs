//@ known-bug: #120016
//@ compile-flags: -Zvalidate-mir
//@ edition: 2021

#![feature(type_alias_impl_trait)]

struct Bug {
    V1: [(); {
        type F = impl Sized;
        #[define_opaque(F)]
        fn concrete_use() -> F {
            //~^ ERROR
            1i32
        }
        let f: F = 0u32;

        1
    }],
}

fn main() {}
