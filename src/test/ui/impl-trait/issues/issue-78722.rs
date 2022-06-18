// edition:2018

#![feature(type_alias_impl_trait)]

type F = impl core::future::Future<Output = u8>;

struct Bug {
    V1: [(); {
        fn concrete_use() -> F {
            async {} //~ ERROR type mismatch
        }
        let f: F = async { 1 };
        //~^ ERROR `async` blocks are not allowed in constants
        //~| ERROR destructors cannot be evaluated at compile-time
        1
    }],
}

fn main() {}
