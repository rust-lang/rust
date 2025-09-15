#![allow(incomplete_features)]
#![feature(adt_const_params, unsized_const_params)]

union Union {
    a: u8,
}

impl PartialEq for Union {
    fn eq(&self, other: &Union) -> bool {
        true
    }
}
impl Eq for Union {}

impl std::marker::ConstParamTy_ for Union {}
//~^ ERROR the trait `ConstParamTy` may not be implemented for this type

#[derive(std::marker::ConstParamTy)]
//~^ ERROR this trait cannot be derived for unions
union UnionDerive {
    a: u8,
}

impl PartialEq for UnionDerive {
    fn eq(&self, other: &UnionDerive) -> bool {
        true
    }
}
impl Eq for UnionDerive {}

fn main() {}
