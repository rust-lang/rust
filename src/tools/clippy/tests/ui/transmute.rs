#![feature(f128)]
#![feature(f16)]
#![allow(
    dead_code,
    clippy::borrow_as_ptr,
    unnecessary_transmutes,
    integer_to_ptr_transmutes,
    clippy::needless_lifetimes,
    clippy::missing_transmute_annotations
)]
//@no-rustfix
extern crate core;

use std::mem::transmute as my_transmute;
use std::vec::Vec as MyVec;

fn my_int() -> Usize {
    Usize(42)
}

fn my_vec() -> MyVec<i32> {
    vec![]
}

#[allow(clippy::needless_lifetimes, clippy::transmute_ptr_to_ptr)]
#[warn(clippy::useless_transmute)]
unsafe fn _generic<'a, T, U: 'a>(t: &'a T) {
    unsafe {
        // FIXME: should lint
        // let _: &'a T = core::mem::transmute(t);

        let _: &'a U = core::mem::transmute(t);

        let _: *const T = core::mem::transmute(t);
        //~^ useless_transmute

        let _: *mut T = core::mem::transmute(t);
        //~^ useless_transmute

        let _: *const U = core::mem::transmute(t);
        //~^ useless_transmute
    }
}

#[warn(clippy::useless_transmute)]
fn useless() {
    unsafe {
        let _: Vec<i32> = core::mem::transmute(my_vec());
        //~^ useless_transmute

        let _: Vec<i32> = core::mem::transmute(my_vec());
        //~^ useless_transmute

        let _: Vec<i32> = std::mem::transmute(my_vec());
        //~^ useless_transmute

        let _: Vec<i32> = std::mem::transmute(my_vec());
        //~^ useless_transmute

        let _: Vec<i32> = my_transmute(my_vec());
        //~^ useless_transmute

        let _: *const usize = std::mem::transmute(5_isize);

        let _ = std::ptr::dangling::<usize>();

        let _: *const usize = std::mem::transmute(1 + 1usize);

        let _ = (1 + 1_usize) as *const usize;
    }

    unsafe fn _f<'a, 'b>(x: &'a u32) -> &'b u32 {
        unsafe { std::mem::transmute(x) }
    }

    unsafe fn _f2<'a, 'b>(x: *const (dyn Iterator<Item = u32> + 'a)) -> *const (dyn Iterator<Item = u32> + 'b) {
        unsafe { std::mem::transmute(x) }
    }

    unsafe fn _f3<'a, 'b>(x: fn(&'a u32)) -> fn(&'b u32) {
        unsafe { std::mem::transmute(x) }
    }

    unsafe fn _f4<'a, 'b>(x: std::borrow::Cow<'a, str>) -> std::borrow::Cow<'b, str> {
        unsafe { std::mem::transmute(x) }
    }
}

struct Usize(usize);

#[warn(clippy::crosspointer_transmute)]
fn crosspointer() {
    let mut int: Usize = Usize(0);
    let int_const_ptr: *const Usize = &int as *const Usize;
    let int_mut_ptr: *mut Usize = &mut int as *mut Usize;

    unsafe {
        let _: Usize = core::mem::transmute(int_const_ptr);
        //~^ crosspointer_transmute

        let _: Usize = core::mem::transmute(int_mut_ptr);
        //~^ crosspointer_transmute

        let _: *const Usize = core::mem::transmute(my_int());
        //~^ crosspointer_transmute

        let _: *mut Usize = core::mem::transmute(my_int());
        //~^ crosspointer_transmute
    }
}

#[warn(clippy::transmute_int_to_bool)]
fn int_to_bool() {
    let _: bool = unsafe { std::mem::transmute(0_u8) };
    //~^ transmute_int_to_bool
}

fn bytes_to_str(mb: &mut [u8]) {
    const B: &[u8] = b"";

    let _: &str = unsafe { std::mem::transmute(B) };
    //~^ transmute_bytes_to_str

    let _: &mut str = unsafe { std::mem::transmute(mb) };
    //~^ transmute_bytes_to_str

    const _: &str = unsafe { std::mem::transmute(B) };
    //~^ transmute_bytes_to_str
}

fn main() {}
