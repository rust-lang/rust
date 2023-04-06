// edition:2018

#![feature(type_alias_impl_trait)]

type F = impl core::future::Future<Output = u8>;

// FIXME(type_alias_impl_trait): this attribute can be added, but is useless
#[defines(F)]
struct Bug {
    V1: [(); {
        #[defines(F)]
        fn concrete_use() -> F {
            async {}
        }
        let f: F = async { 1 };
        //~^ ERROR: mismatched types
        1
    }],
}

fn main() {}
