#![crate_type = "rlib"]
#![feature(fundamental)]
#![allow(unused_unconstructable_pub_structs)]

pub trait MyCopy { }
impl MyCopy for i32 { }

pub struct MyStruct<T>(T);

#[fundamental]
pub struct MyFundamentalStruct<T>(T);
