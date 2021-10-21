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

#[allow(clippy::needless_lifetimes, clippy::transmute_ptr_to_ptr)]
#[warn(clippy::useless_transmute)]
unsafe fn _generic<'a, T, U: 'a>(t: &'a T) {
    let _: &'a T = core::intrinsics::transmute(t);

    let _: &'a U = core::intrinsics::transmute(t);

    let _: *const T = core::intrinsics::transmute(t);

    let _: *mut T = core::intrinsics::transmute(t);

    let _: *const U = core::intrinsics::transmute(t);
}

#[warn(clippy::useless_transmute)]
fn useless() {
    unsafe {
        let _: Vec<i32> = core::intrinsics::transmute(my_vec());

        let _: Vec<i32> = core::mem::transmute(my_vec());

        let _: Vec<i32> = std::intrinsics::transmute(my_vec());

        let _: Vec<i32> = std::mem::transmute(my_vec());

        let _: Vec<i32> = my_transmute(my_vec());

        let _: *const usize = std::mem::transmute(5_isize);

        let _ = 5_isize as *const usize;

        let _: *const usize = std::mem::transmute(1 + 1usize);

        let _ = (1 + 1_usize) as *const usize;
    }
}

struct Usize(usize);

#[warn(clippy::crosspointer_transmute)]
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

#[warn(clippy::transmute_int_to_char)]
fn int_to_char() {
    let _: char = unsafe { std::mem::transmute(0_u32) };
    let _: char = unsafe { std::mem::transmute(0_i32) };
}

#[warn(clippy::transmute_int_to_bool)]
fn int_to_bool() {
    let _: bool = unsafe { std::mem::transmute(0_u8) };
}

#[warn(clippy::transmute_int_to_float)]
mod int_to_float {
    fn test() {
        let _: f32 = unsafe { std::mem::transmute(0_u32) };
        let _: f32 = unsafe { std::mem::transmute(0_i32) };
        let _: f64 = unsafe { std::mem::transmute(0_u64) };
        let _: f64 = unsafe { std::mem::transmute(0_i64) };
    }

    mod issue_5747 {
        const VALUE32: f32 = unsafe { std::mem::transmute(0_u32) };
        const VALUE64: f64 = unsafe { std::mem::transmute(0_i64) };

        const fn from_bits_32(v: i32) -> f32 {
            unsafe { std::mem::transmute(v) }
        }

        const fn from_bits_64(v: u64) -> f64 {
            unsafe { std::mem::transmute(v) }
        }
    }
}

mod num_to_bytes {
    fn test() {
        unsafe {
            let _: [u8; 1] = std::mem::transmute(0u8);
            let _: [u8; 4] = std::mem::transmute(0u32);
            let _: [u8; 16] = std::mem::transmute(0u128);
            let _: [u8; 1] = std::mem::transmute(0i8);
            let _: [u8; 4] = std::mem::transmute(0i32);
            let _: [u8; 16] = std::mem::transmute(0i128);
            let _: [u8; 4] = std::mem::transmute(0.0f32);
            let _: [u8; 8] = std::mem::transmute(0.0f64);
        }
    }
    const fn test_const() {
        unsafe {
            let _: [u8; 1] = std::mem::transmute(0u8);
            let _: [u8; 4] = std::mem::transmute(0u32);
            let _: [u8; 16] = std::mem::transmute(0u128);
            let _: [u8; 1] = std::mem::transmute(0i8);
            let _: [u8; 4] = std::mem::transmute(0i32);
            let _: [u8; 16] = std::mem::transmute(0i128);
            let _: [u8; 4] = std::mem::transmute(0.0f32);
            let _: [u8; 8] = std::mem::transmute(0.0f64);
        }
    }
}

fn bytes_to_str(b: &[u8], mb: &mut [u8]) {
    let _: &str = unsafe { std::mem::transmute(b) };
    let _: &mut str = unsafe { std::mem::transmute(mb) };
}

fn main() {}
