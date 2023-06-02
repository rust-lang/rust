// [full] check-pass
// revisions: full min
#![cfg_attr(full, feature(adt_const_params))]
#![cfg_attr(full, allow(incomplete_features))]

#[cfg(full)]
use std::marker::ConstParamTy;

#[derive(PartialEq, Eq)]
#[cfg_attr(full, derive(ConstParamTy))]
enum IceEnum {
    Variant
}

struct IceStruct;

impl IceStruct {
    fn ice_struct_fn<const I: IceEnum>() {}
    //[min]~^ ERROR `IceEnum` is forbidden as the type of a const generic parameter
}

fn main() {
    IceStruct::ice_struct_fn::<{IceEnum::Variant}>();
}
