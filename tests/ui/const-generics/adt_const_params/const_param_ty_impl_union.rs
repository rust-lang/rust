#![allow(incomplete_features)]
#![feature(adt_const_params, structural_match)]

union Union {
    a: u8,
}

impl PartialEq for Union {
    fn eq(&self, other: &Union) -> bool {
        true
    }
}
impl Eq for Union {}
impl std::marker::StructuralEq for Union {}

impl std::marker::ConstParamTy for Union {}
//~^ ERROR the type `Union` does not `#[derive(PartialEq)]`

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
impl std::marker::StructuralEq for UnionDerive {}


fn main() {}
