//@ edition:2018

#![feature(type_alias_impl_trait)]

struct Bug {
    V1: [(); {
        type F = impl core::future::Future<Output = u8>;
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
