//! test that we cannot register hidden types for opaque types
//! declared outside an anonymous constant.
//@ edition:2018

#![feature(type_alias_impl_trait)]

type F = impl core::future::Future<Output = u8>;

struct Bug {
    V1: [(); {
        #[define_opaque(F)]
        fn concrete_use() -> F {
            //~^ ERROR future that resolves to `u8`, but it resolves to `()`
            async {}
        }
        // FIXME(type_alias_impl_trait): inform the user about why `F` is not available here.
        let f: F = async { 1 };
        //~^ ERROR mismatched types
        1
    }],
}

fn main() {}
