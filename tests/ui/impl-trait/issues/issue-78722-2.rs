//! test that we cannot register hidden types for opaque types
//! declared outside an anonymous constant.
// edition:2018

#![feature(type_alias_impl_trait)]

type F = impl core::future::Future<Output = u8>;

struct Bug {
    V1: [(); {
        fn concrete_use() -> F {
            //~^ ERROR future that resolves to `u8`, but it resolves to `()`
            async {}
        }
        let f: F = async { 1 };
        //~^ ERROR item constrains opaque type that is not in its signature
        //~| ERROR `async` blocks are not allowed in constants
        1
    }],
}

fn main() {}
