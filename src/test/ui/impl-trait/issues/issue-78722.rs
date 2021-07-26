// edition:2018

#![feature(type_alias_impl_trait)]

type F = impl core::future::Future<Output = u8>;

struct Bug {
    V1: [(); {
        fn concrete_use() -> F {
            async {}
        }
        let f: F = async { 1 };
        //~^ ERROR mismatched types [E0308]
        1
    }],
}

fn main() {}
