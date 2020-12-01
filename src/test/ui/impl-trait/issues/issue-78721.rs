// edition:2018

#![feature(impl_trait_in_bindings)]
//~^ WARN the feature `impl_trait_in_bindings` is incomplete

struct Bug {
    V1: [(); {
        let f: impl core::future::Future<Output = u8> = async { 1 };
        //~^ ERROR `async` blocks are not allowed in constants
        //~| ERROR destructors cannot be evaluated at compile-time
        1
    }],
}

fn main() {}
