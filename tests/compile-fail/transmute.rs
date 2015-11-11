#![feature(core)]
#![feature(plugin)]
#![plugin(clippy)]
#![deny(useless_transmute)]

extern crate core;

use std::mem::transmute as my_transmute;
use std::vec::Vec as MyVec;

fn my_vec() -> MyVec<i32> {
    vec![]
}

#[allow(needless_lifetimes)]
unsafe fn _generic<'a, T, U: 'a>(t: &'a T) {
    let _: &'a T = core::intrinsics::transmute(t);
    //~^ ERROR transmute from a type (`&'a T`) to itself

    let _: &'a U = core::intrinsics::transmute(t);
}

fn main() {
    unsafe {
        let _: Vec<i32> = core::intrinsics::transmute(my_vec());
        //~^ ERROR transmute from a type (`collections::vec::Vec<i32>`) to itself

        let _: Vec<i32> = core::mem::transmute(my_vec());
        //~^ ERROR transmute from a type (`collections::vec::Vec<i32>`) to itself

        let _: Vec<i32> = std::intrinsics::transmute(my_vec());
        //~^ ERROR transmute from a type (`collections::vec::Vec<i32>`) to itself

        let _: Vec<i32> = std::mem::transmute(my_vec());
        //~^ ERROR transmute from a type (`collections::vec::Vec<i32>`) to itself

        let _: Vec<i32> = my_transmute(my_vec());
        //~^ ERROR transmute from a type (`collections::vec::Vec<i32>`) to itself

        let _: Vec<u32> = core::intrinsics::transmute(my_vec());
        let _: Vec<u32> = core::mem::transmute(my_vec());
        let _: Vec<u32> = std::intrinsics::transmute(my_vec());
        let _: Vec<u32> = std::mem::transmute(my_vec());
        let _: Vec<u32> = my_transmute(my_vec());
    }
}
