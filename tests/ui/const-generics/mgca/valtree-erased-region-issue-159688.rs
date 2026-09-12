//@ check-fail

#![feature(min_generic_const_args)]
#![feature(min_adt_const_params)]
#![expect(incomplete_features)]

pub struct Foo {
    slice: &'static [u8],
}

#[derive(PartialEq, Eq)]
pub struct Foo_<const F: Foo>;
//~^ ERROR `Foo` must implement `ConstParamTy`
//~| ERROR `Foo` must implement `ConstParamTy`
//~| ERROR `Foo` must implement `ConstParamTy`
//~| ERROR `Foo` must implement `ConstParamTy`

impl
    Foo_<
        {
            Foo {
                slice: &[1, 2, 3],
            }
        },
    >
{
}

fn main() {}
