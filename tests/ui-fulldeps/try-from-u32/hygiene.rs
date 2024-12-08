#![feature(rustc_private)]
//@ edition: 2021
//@ check-pass

// Checks that the derive macro still works even if the surrounding code has
// shadowed the relevant library types.

extern crate rustc_macros;

mod submod {
    use rustc_macros::TryFromU32;

    struct Result;
    trait TryFrom {}
    #[allow(non_camel_case_types)]
    struct u32;
    struct Ok;
    struct Err;
    mod core {}
    mod std {}

    #[derive(TryFromU32)]
    pub(crate) enum MyEnum {
        Zero,
        One,
    }
}

fn main() {
    use submod::MyEnum;
    let _: Result<MyEnum, u32> = MyEnum::try_from(1u32);
}
