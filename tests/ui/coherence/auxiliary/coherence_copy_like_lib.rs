#![crate_type = "rlib"]
#![feature(fundamental)]

pub trait MyCopy { }
impl MyCopy for i32 { }

pub struct MyStruct<T>(T);

#[fundamental]
pub struct MyFundamentalStruct<T>(T);
