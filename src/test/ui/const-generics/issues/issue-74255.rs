// [full] check-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

#[derive(PartialEq, Eq)]
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
