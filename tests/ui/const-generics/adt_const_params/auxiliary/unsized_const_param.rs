#![feature(adt_const_params, unsized_const_params)]
#![allow(unconstructable_pub_struct)]

#[derive(std::marker::ConstParamTy, Eq, PartialEq)]
pub struct Foo([u8]);

#[derive(std::marker::ConstParamTy, Eq, PartialEq)]
pub struct GenericNotUnsizedParam<T>(T);
