#![feature(plugin)]
#![plugin(clippy)]

#![allow(dead_code)]

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
#[warn(useless_transmute)]
unsafe fn _generic<'a, T, U: 'a>(t: &'a T) {
    let _: &'a T = core::intrinsics::transmute(t);

    let _: &'a U = core::intrinsics::transmute(t);

    let _: *const T = core::intrinsics::transmute(t);

    let _: *mut T = core::intrinsics::transmute(t);

    let _: *const U = core::intrinsics::transmute(t);
}

#[warn(transmute_ptr_to_ref)]
unsafe fn _ptr_to_ref<T, U>(p: *const T, m: *mut T, o: *const U, om: *mut U) {
    let _: &T = std::mem::transmute(p);
    let _: &T = &*p;

    let _: &mut T = std::mem::transmute(m);
    let _: &mut T = &mut *m;

    let _: &T = std::mem::transmute(m);
    let _: &T = &*m;

    let _: &mut T = std::mem::transmute(p as *mut T);
    let _ = &mut *(p as *mut T);

    let _: &T = std::mem::transmute(o);
    let _: &T = &*(o as *const T);

    let _: &mut T = std::mem::transmute(om);
    let _: &mut T = &mut *(om as *mut T);

    let _: &T = std::mem::transmute(om);
    let _: &T = &*(om as *const T);
}

#[warn(transmute_ptr_to_ref)]
fn issue1231() {
    struct Foo<'a, T: 'a> {
        bar: &'a T,
    }

    let raw = 42 as *const i32;
    let _: &Foo<u8> = unsafe { std::mem::transmute::<_, &Foo<_>>(raw) };

    let _: &Foo<&u8> = unsafe { std::mem::transmute::<_, &Foo<&_>>(raw) };

    type Bar<'a> = &'a u8;
    let raw = 42 as *const i32;
    unsafe { std::mem::transmute::<_, Bar>(raw) };
}

#[warn(useless_transmute)]
fn useless() {
    unsafe {
        let _: Vec<i32> = core::intrinsics::transmute(my_vec());

        let _: Vec<i32> = core::mem::transmute(my_vec());

        let _: Vec<i32> = std::intrinsics::transmute(my_vec());

        let _: Vec<i32> = std::mem::transmute(my_vec());

        let _: Vec<i32> = my_transmute(my_vec());

        let _: Vec<u32> = core::intrinsics::transmute(my_vec());
        let _: Vec<u32> = core::mem::transmute(my_vec());
        let _: Vec<u32> = std::intrinsics::transmute(my_vec());
        let _: Vec<u32> = std::mem::transmute(my_vec());
        let _: Vec<u32> = my_transmute(my_vec());

        let _: *const usize = std::mem::transmute(5_isize);

        let _  = 5_isize as *const usize;

        let _: *const usize = std::mem::transmute(1+1usize);

        let _  = (1+1_usize) as *const usize;
    }
}

struct Usize(usize);

#[warn(crosspointer_transmute)]
fn crosspointer() {
    let mut int: Usize = Usize(0);
    let int_const_ptr: *const Usize = &int as *const Usize;
    let int_mut_ptr: *mut Usize = &mut int as *mut Usize;

    unsafe {
        let _: Usize = core::intrinsics::transmute(int_const_ptr);

        let _: Usize = core::intrinsics::transmute(int_mut_ptr);

        let _: *const Usize = core::intrinsics::transmute(my_int());

        let _: *mut Usize = core::intrinsics::transmute(my_int());
    }
}

fn main() { }
