// Ban Self in const generics when min_adt_const_params and adt_const_params are not enabled
// #149203
trait MyTrait {
    fn foo<const N: i32>();
}

impl MyTrait for i32 {
    fn foo<const N: Self>() {}
    //~^ HELP add `#![feature(min_adt_const_params)]` to the crate attributes to enable `Self` as a const parameter type
    //~| ERROR cannot use `Self` in const parameter type
}

fn main(){}
