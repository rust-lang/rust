// edition:2018

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

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
