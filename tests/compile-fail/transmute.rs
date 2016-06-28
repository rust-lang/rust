#![feature(plugin)]
#![plugin(clippy)]

extern crate core;

use std::mem::transmute as my_transmute;
use std::vec::Vec as MyVec;

fn my_int() -> Usize {
    Usize(42)
}

fn my_vec() -> MyVec<i32> {
    vec![]
}

#[allow(needless_lifetimes)]
#[deny(useless_transmute)]
unsafe fn _generic<'a, T, U: 'a>(t: &'a T) {
    let _: &'a T = core::intrinsics::transmute(t);
    //~^ ERROR transmute from a type (`&'a T`) to itself

    let _: &'a U = core::intrinsics::transmute(t);

    let _: *const T = core::intrinsics::transmute(t);
    //~^ ERROR transmute from a reference to a pointer
    //~| HELP try
    //~| SUGGESTION = t as *const T

    let _: *mut T = core::intrinsics::transmute(t);
    //~^ ERROR transmute from a reference to a pointer
    //~| HELP try
    //~| SUGGESTION = t as *const T as *mut T

    let _: *const U = core::intrinsics::transmute(t);
    //~^ ERROR transmute from a reference to a pointer
    //~| HELP try
    //~| SUGGESTION = t as *const T as *const U
}

#[deny(transmute_ptr_to_ref)]
unsafe fn _ptr_to_ref<T, U>(p: *const T, m: *mut T, o: *const U, om: *mut U) {
    let _: &T = std::mem::transmute(p);
    //~^ ERROR transmute from a pointer type (`*const T`) to a reference type (`&T`)
    //~| HELP try
    //~| SUGGESTION = &*p;
    let _: &T = &*p;

    let _: &mut T = std::mem::transmute(m);
    //~^ ERROR transmute from a pointer type (`*mut T`) to a reference type (`&mut T`)
    //~| HELP try
    //~| SUGGESTION = &mut *m;
    let _: &mut T = &mut *m;

    let _: &T = std::mem::transmute(m);
    //~^ ERROR transmute from a pointer type (`*mut T`) to a reference type (`&T`)
    //~| HELP try
    //~| SUGGESTION = &*m;
    let _: &T = &*m;

    let _: &T = std::mem::transmute(o);
    //~^ ERROR transmute from a pointer type (`*const U`) to a reference type (`&T`)
    //~| HELP try
    //~| SUGGESTION = &*(o as *const T);
    let _: &T = &*(o as *const T);

    let _: &mut T = std::mem::transmute(om);
    //~^ ERROR transmute from a pointer type (`*mut U`) to a reference type (`&mut T`)
    //~| HELP try
    //~| SUGGESTION = &mut *(om as *mut T);
    let _: &mut T = &mut *(om as *mut T);

    let _: &T = std::mem::transmute(om);
    //~^ ERROR transmute from a pointer type (`*mut U`) to a reference type (`&T`)
    //~| HELP try
    //~| SUGGESTION = &*(om as *const T);
    let _: &T = &*(om as *const T);
}

#[deny(useless_transmute)]
fn useless() {
    unsafe {
        let _: Vec<i32> = core::intrinsics::transmute(my_vec());
        //~^ ERROR transmute from a type (`std::vec::Vec<i32>`) to itself

        let _: Vec<i32> = core::mem::transmute(my_vec());
        //~^ ERROR transmute from a type (`std::vec::Vec<i32>`) to itself

        let _: Vec<i32> = std::intrinsics::transmute(my_vec());
        //~^ ERROR transmute from a type (`std::vec::Vec<i32>`) to itself

        let _: Vec<i32> = std::mem::transmute(my_vec());
        //~^ ERROR transmute from a type (`std::vec::Vec<i32>`) to itself

        let _: Vec<i32> = my_transmute(my_vec());
        //~^ ERROR transmute from a type (`std::vec::Vec<i32>`) to itself

        let _: Vec<u32> = core::intrinsics::transmute(my_vec());
        let _: Vec<u32> = core::mem::transmute(my_vec());
        let _: Vec<u32> = std::intrinsics::transmute(my_vec());
        let _: Vec<u32> = std::mem::transmute(my_vec());
        let _: Vec<u32> = my_transmute(my_vec());

        let _: *const usize = std::mem::transmute(5_isize);
        //~^ ERROR transmute from an integer to a pointer
        //~| HELP try
        //~| SUGGESTION 5_isize as *const usize
    }
}

struct Usize(usize);

#[deny(crosspointer_transmute)]
fn crosspointer() {
    let mut int: Usize = Usize(0);
    let int_const_ptr: *const Usize = &int as *const Usize;
    let int_mut_ptr: *mut Usize = &mut int as *mut Usize;

    unsafe {
        let _: Usize = core::intrinsics::transmute(int_const_ptr);
        //~^ ERROR transmute from a type (`*const Usize`) to the type that it points to (`Usize`)

        let _: Usize = core::intrinsics::transmute(int_mut_ptr);
        //~^ ERROR transmute from a type (`*mut Usize`) to the type that it points to (`Usize`)

        let _: *const Usize = core::intrinsics::transmute(my_int());
        //~^ ERROR transmute from a type (`Usize`) to a pointer to that type (`*const Usize`)

        let _: *mut Usize = core::intrinsics::transmute(my_int());
        //~^ ERROR transmute from a type (`Usize`) to a pointer to that type (`*mut Usize`)
    }
}

fn main() {
    useless();
    crosspointer();
}
