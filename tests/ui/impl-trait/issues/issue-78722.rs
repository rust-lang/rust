// edition:2018

// revisions: global in_const

#![feature(type_alias_impl_trait)]

#[cfg(global)]
type F = impl core::future::Future<Output = u8>;

struct Bug {
    V1: [(); {
        #[cfg(in_const)]
        type F = impl core::future::Future<Output = u8>;
        fn concrete_use() -> F {
            //[in_const]~^ ERROR to be a future that resolves to `u8`, but it resolves to `()`
            async {}
        }
        let f: F = async { 1 };
        //[in_const]~^ ERROR `async` blocks are not allowed in constants
        //[in_const]~| ERROR destructor of
        //[global]~^^^ ERROR mismatched types
        1
    }],
}

fn main() {}
