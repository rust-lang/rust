#![feature(adt_const_params)]
#![allow(incomplete_features)]

use std::sync::Arc;

#[derive(PartialEq, Eq)]
enum Bar {
    Bar(Arc<i32>)
}

fn test<const BAR: Bar>() {}
//~^ ERROR  `Bar` must implement `ConstParamTy` to be used as the type of a const generic parameter

fn main() {}
