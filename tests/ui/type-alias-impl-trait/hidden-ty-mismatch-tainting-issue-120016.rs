//@ compile-flags: --edition=2021

#![feature(type_alias_impl_trait, const_async_blocks)]

struct Bug {
    V1: [(); {
        type F = impl std::future::Future<Output = impl Sized>;
        fn concrete_use() -> F {
            async {}
        }
        let f: F = async { 1 };
        //~^ ERROR concrete type differs from previous defining opaque type use
        //~| ERROR concrete type differs from previous defining opaque type use
        1
    }],
}

fn main() {}
