#![feature(plugin)]
#![plugin(clippy)]

extern crate core;

use std::mem::transmute as my_transmute;
use std::vec::Vec as MyVec;

fn my_vec() -> MyVec<i32> {
    vec![]
}

#[allow(needless_lifetimes)]
#[deny(useless_transmute)]
unsafe fn _generic<'a, T, U: 'a>(t: &'a T) {
    let _: &'a T = core::intrinsics::transmute(t);
    //~^ ERROR transmute from a type (`&'a T`) to itself

    let _: &'a U = core::intrinsics::transmute(t);
}

#[deny(useless_transmute)]
fn useless() {
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

#[deny(crosspointer_transmute)]
fn crosspointer() {
    let mut vec: Vec<i32> = vec![];
    let vec_const_ptr: *const Vec<i32> = &vec as *const Vec<i32>;
    let vec_mut_ptr: *mut Vec<i32> = &mut vec as *mut Vec<i32>;

    unsafe {
        let _: Vec<i32> = core::intrinsics::transmute(vec_const_ptr);
        //~^ ERROR transmute from a type (`*const collections::vec::Vec<i32>`) to the type that it points to (`collections::vec::Vec<i32>`)

        let _: Vec<i32> = core::intrinsics::transmute(vec_mut_ptr);
        //~^ ERROR transmute from a type (`*mut collections::vec::Vec<i32>`) to the type that it points to (`collections::vec::Vec<i32>`)

        let _: *const Vec<i32> = core::intrinsics::transmute(my_vec());
        //~^ ERROR transmute from a type (`collections::vec::Vec<i32>`) to a pointer to that type (`*const collections::vec::Vec<i32>`)

        let _: *mut Vec<i32> = core::intrinsics::transmute(my_vec());
        //~^ ERROR transmute from a type (`collections::vec::Vec<i32>`) to a pointer to that type (`*mut collections::vec::Vec<i32>`)
    }
}

fn main() {
    useless();
    crosspointer();
}
