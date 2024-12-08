#![feature(f128)]
#![feature(f16)]
#![allow(
    dead_code,
    clippy::borrow_as_ptr,
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
    // FIXME: should lint
    // let _: &'a T = core::intrinsics::transmute(t);

    let _: &'a U = core::intrinsics::transmute(t);

    let _: *const T = core::intrinsics::transmute(t);
    //~^ ERROR: transmute from a reference to a pointer
    //~| NOTE: `-D clippy::useless-transmute` implied by `-D warnings`

    let _: *mut T = core::intrinsics::transmute(t);
    //~^ ERROR: transmute from a reference to a pointer

    let _: *const U = core::intrinsics::transmute(t);
    //~^ ERROR: transmute from a reference to a pointer
}

#[warn(clippy::useless_transmute)]
fn useless() {
    unsafe {
        let _: Vec<i32> = core::intrinsics::transmute(my_vec());
        //~^ ERROR: transmute from a type (`std::vec::Vec<i32>`) to itself

        let _: Vec<i32> = core::mem::transmute(my_vec());
        //~^ ERROR: transmute from a type (`std::vec::Vec<i32>`) to itself

        let _: Vec<i32> = std::intrinsics::transmute(my_vec());
        //~^ ERROR: transmute from a type (`std::vec::Vec<i32>`) to itself

        let _: Vec<i32> = std::mem::transmute(my_vec());
        //~^ ERROR: transmute from a type (`std::vec::Vec<i32>`) to itself

        let _: Vec<i32> = my_transmute(my_vec());
        //~^ ERROR: transmute from a type (`std::vec::Vec<i32>`) to itself

        let _: *const usize = std::mem::transmute(5_isize);
        //~^ ERROR: transmute from an integer to a pointer

        let _ = 5_isize as *const usize;

        let _: *const usize = std::mem::transmute(1 + 1usize);
        //~^ ERROR: transmute from an integer to a pointer

        let _ = (1 + 1_usize) as *const usize;
    }

    unsafe fn _f<'a, 'b>(x: &'a u32) -> &'b u32 {
        std::mem::transmute(x)
    }

    unsafe fn _f2<'a, 'b>(x: *const (dyn Iterator<Item = u32> + 'a)) -> *const (dyn Iterator<Item = u32> + 'b) {
        std::mem::transmute(x)
    }

    unsafe fn _f3<'a, 'b>(x: fn(&'a u32)) -> fn(&'b u32) {
        std::mem::transmute(x)
    }

    unsafe fn _f4<'a, 'b>(x: std::borrow::Cow<'a, str>) -> std::borrow::Cow<'b, str> {
        std::mem::transmute(x)
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
        //~^ ERROR: transmute from a type (`*const Usize`) to the type that it points to (
        //~| NOTE: `-D clippy::crosspointer-transmute` implied by `-D warnings`

        let _: Usize = core::intrinsics::transmute(int_mut_ptr);
        //~^ ERROR: transmute from a type (`*mut Usize`) to the type that it points to (`U

        let _: *const Usize = core::intrinsics::transmute(my_int());
        //~^ ERROR: transmute from a type (`Usize`) to a pointer to that type (`*const Usi

        let _: *mut Usize = core::intrinsics::transmute(my_int());
        //~^ ERROR: transmute from a type (`Usize`) to a pointer to that type (`*mut Usize
    }
}

#[warn(clippy::transmute_int_to_bool)]
fn int_to_bool() {
    let _: bool = unsafe { std::mem::transmute(0_u8) };
    //~^ ERROR: transmute from a `u8` to a `bool`
    //~| NOTE: `-D clippy::transmute-int-to-bool` implied by `-D warnings`
}

#[warn(clippy::transmute_int_to_float)]
mod int_to_float {
    fn test() {
        let _: f16 = unsafe { std::mem::transmute(0_u16) };
        //~^ ERROR: transmute from a `u16` to a `f16`
        //~| NOTE: `-D clippy::transmute-int-to-float` implied by `-D warnings`
        let _: f16 = unsafe { std::mem::transmute(0_i16) };
        //~^ ERROR: transmute from a `i16` to a `f16`
        let _: f32 = unsafe { std::mem::transmute(0_u32) };
        //~^ ERROR: transmute from a `u32` to a `f32`
        let _: f32 = unsafe { std::mem::transmute(0_i32) };
        //~^ ERROR: transmute from a `i32` to a `f32`
        let _: f64 = unsafe { std::mem::transmute(0_u64) };
        //~^ ERROR: transmute from a `u64` to a `f64`
        let _: f64 = unsafe { std::mem::transmute(0_i64) };
        //~^ ERROR: transmute from a `i64` to a `f64`
        let _: f128 = unsafe { std::mem::transmute(0_u128) };
        //~^ ERROR: transmute from a `u128` to a `f128`
        let _: f128 = unsafe { std::mem::transmute(0_i128) };
        //~^ ERROR: transmute from a `i128` to a `f128`
    }

    mod issue_5747 {
        const VALUE16: f16 = unsafe { std::mem::transmute(0_u16) };
        //~^ ERROR: transmute from a `u16` to a `f16`
        const VALUE32: f32 = unsafe { std::mem::transmute(0_u32) };
        //~^ ERROR: transmute from a `u32` to a `f32`
        const VALUE64: f64 = unsafe { std::mem::transmute(0_i64) };
        //~^ ERROR: transmute from a `i64` to a `f64`
        const VALUE128: f128 = unsafe { std::mem::transmute(0_i128) };
        //~^ ERROR: transmute from a `i128` to a `f128`

        const fn from_bits_16(v: i16) -> f16 {
            unsafe { std::mem::transmute(v) }
            //~^ ERROR: transmute from a `i16` to a `f16`
        }

        const fn from_bits_32(v: i32) -> f32 {
            unsafe { std::mem::transmute(v) }
            //~^ ERROR: transmute from a `i32` to a `f32`
        }

        const fn from_bits_64(v: u64) -> f64 {
            unsafe { std::mem::transmute(v) }
            //~^ ERROR: transmute from a `u64` to a `f64`
        }

        const fn from_bits_128(v: u128) -> f128 {
            unsafe { std::mem::transmute(v) }
            //~^ ERROR: transmute from a `u128` to a `f128`
        }
    }
}

mod num_to_bytes {
    fn test() {
        unsafe {
            let _: [u8; 1] = std::mem::transmute(0u8);
            //~^ ERROR: transmute from a `u8` to a `[u8; 1]`
            //~| NOTE: `-D clippy::transmute-num-to-bytes` implied by `-D warnings`
            let _: [u8; 4] = std::mem::transmute(0u32);
            //~^ ERROR: transmute from a `u32` to a `[u8; 4]`
            let _: [u8; 16] = std::mem::transmute(0u128);
            //~^ ERROR: transmute from a `u128` to a `[u8; 16]`
            let _: [u8; 1] = std::mem::transmute(0i8);
            //~^ ERROR: transmute from a `i8` to a `[u8; 1]`
            let _: [u8; 4] = std::mem::transmute(0i32);
            //~^ ERROR: transmute from a `i32` to a `[u8; 4]`
            let _: [u8; 16] = std::mem::transmute(0i128);
            //~^ ERROR: transmute from a `i128` to a `[u8; 16]`

            let _: [u8; 2] = std::mem::transmute(0.0f16);
            //~^ ERROR: transmute from a `f16` to a `[u8; 2]`
            let _: [u8; 4] = std::mem::transmute(0.0f32);
            //~^ ERROR: transmute from a `f32` to a `[u8; 4]`
            let _: [u8; 8] = std::mem::transmute(0.0f64);
            //~^ ERROR: transmute from a `f64` to a `[u8; 8]`
            let _: [u8; 16] = std::mem::transmute(0.0f128);
            //~^ ERROR: transmute from a `f128` to a `[u8; 16]`
        }
    }
    const fn test_const() {
        unsafe {
            let _: [u8; 1] = std::mem::transmute(0u8);
            //~^ ERROR: transmute from a `u8` to a `[u8; 1]`
            let _: [u8; 4] = std::mem::transmute(0u32);
            //~^ ERROR: transmute from a `u32` to a `[u8; 4]`
            let _: [u8; 16] = std::mem::transmute(0u128);
            //~^ ERROR: transmute from a `u128` to a `[u8; 16]`
            let _: [u8; 1] = std::mem::transmute(0i8);
            //~^ ERROR: transmute from a `i8` to a `[u8; 1]`
            let _: [u8; 4] = std::mem::transmute(0i32);
            //~^ ERROR: transmute from a `i32` to a `[u8; 4]`
            let _: [u8; 16] = std::mem::transmute(0i128);
            //~^ ERROR: transmute from a `i128` to a `[u8; 16]`

            let _: [u8; 2] = std::mem::transmute(0.0f16);
            //~^ ERROR: transmute from a `f16` to a `[u8; 2]`
            let _: [u8; 4] = std::mem::transmute(0.0f32);
            //~^ ERROR: transmute from a `f32` to a `[u8; 4]`
            let _: [u8; 8] = std::mem::transmute(0.0f64);
            //~^ ERROR: transmute from a `f64` to a `[u8; 8]`
            let _: [u8; 16] = std::mem::transmute(0.0f128);
            //~^ ERROR: transmute from a `f128` to a `[u8; 16]`
        }
    }
}

fn bytes_to_str(mb: &mut [u8]) {
    const B: &[u8] = b"";

    let _: &str = unsafe { std::mem::transmute(B) };
    //~^ ERROR: transmute from a `&[u8]` to a `&str`
    //~| NOTE: `-D clippy::transmute-bytes-to-str` implied by `-D warnings`
    let _: &mut str = unsafe { std::mem::transmute(mb) };
    //~^ ERROR: transmute from a `&mut [u8]` to a `&mut str`
    const _: &str = unsafe { std::mem::transmute(B) };
    //~^ ERROR: transmute from a `&[u8]` to a `&str`
}

fn main() {}
