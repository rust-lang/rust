//@ known-bug: #120016
//@ compile-flags: -Zcrate-attr=feature(const_async_blocks)
//@ edition: 2021

#![feature(type_alias_impl_trait, const_async_blocks)]

struct Bug {
    V1: [(); {
        type F = impl std::future::Future<Output = impl Sized>;
        #[define_opaque(F)]
        fn concrete_use() -> F {
            //~^ ERROR to be a future that resolves to `u8`, but it resolves to `()`
            async {}
        }
        let f: F = async { 1 };
        //~^ ERROR `async` blocks are not allowed in constants
        1
    }],
}

fn main() {}
