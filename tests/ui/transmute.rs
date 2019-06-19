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

#[warn(clippy::transmute_ptr_to_ref)]
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

#[warn(clippy::transmute_ptr_to_ref)]
fn issue1231() {
    struct Foo<'a, T> {
        bar: &'a T,
    }

    let raw = 42 as *const i32;
    let _: &Foo<u8> = unsafe { std::mem::transmute::<_, &Foo<_>>(raw) };

    let _: &Foo<&u8> = unsafe { std::mem::transmute::<_, &Foo<&_>>(raw) };

    type Bar<'a> = &'a u8;
    let raw = 42 as *const i32;
    unsafe { std::mem::transmute::<_, Bar>(raw) };
}

#[warn(clippy::useless_transmute)]
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
fn int_to_float() {
    let _: f32 = unsafe { std::mem::transmute(0_u32) };
    let _: f32 = unsafe { std::mem::transmute(0_i32) };
}

fn bytes_to_str(b: &[u8], mb: &mut [u8]) {
    let _: &str = unsafe { std::mem::transmute(b) };
    let _: &mut str = unsafe { std::mem::transmute(mb) };
}

// Make sure we can modify lifetimes, which is one of the recommended uses
// of transmute

// Make sure we can do static lifetime transmutes
#[warn(clippy::transmute_ptr_to_ptr)]
unsafe fn transmute_lifetime_to_static<'a, T>(t: &'a T) -> &'static T {
    std::mem::transmute::<&'a T, &'static T>(t)
}

// Make sure we can do non-static lifetime transmutes
#[warn(clippy::transmute_ptr_to_ptr)]
unsafe fn transmute_lifetime<'a, 'b, T>(t: &'a T, u: &'b T) -> &'b T {
    std::mem::transmute::<&'a T, &'b T>(t)
}

struct LifetimeParam<'a> {
    s: &'a str,
}

struct GenericParam<T> {
    t: T,
}

#[warn(clippy::transmute_ptr_to_ptr)]
fn transmute_ptr_to_ptr() {
    let ptr = &1u32 as *const u32;
    let mut_ptr = &mut 1u32 as *mut u32;
    unsafe {
        // pointer-to-pointer transmutes; bad
        let _: *const f32 = std::mem::transmute(ptr);
        let _: *mut f32 = std::mem::transmute(mut_ptr);
        // ref-ref transmutes; bad
        let _: &f32 = std::mem::transmute(&1u32);
        let _: &f64 = std::mem::transmute(&1f32);
        // ^ this test is here because both f32 and f64 are the same TypeVariant, but they are not
        // the same type
        let _: &mut f32 = std::mem::transmute(&mut 1u32);
        let _: &GenericParam<f32> = std::mem::transmute(&GenericParam { t: 1u32 });
    }

    // these are recommendations for solving the above; if these lint we need to update
    // those suggestions
    let _ = ptr as *const f32;
    let _ = mut_ptr as *mut f32;
    let _ = unsafe { &*(&1u32 as *const u32 as *const f32) };
    let _ = unsafe { &mut *(&mut 1u32 as *mut u32 as *mut f32) };

    // transmute internal lifetimes, should not lint
    let s = "hello world".to_owned();
    let lp = LifetimeParam { s: &s };
    let _: &LifetimeParam<'static> = unsafe { std::mem::transmute(&lp) };
    let _: &GenericParam<&LifetimeParam<'static>> = unsafe { std::mem::transmute(&GenericParam { t: &lp }) };
}

fn main() {}
