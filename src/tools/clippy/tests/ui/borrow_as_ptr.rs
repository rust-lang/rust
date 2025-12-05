//@aux-build:proc_macros.rs
#![warn(clippy::borrow_as_ptr)]
#![allow(clippy::useless_vec)]

extern crate proc_macros;

fn a() -> i32 {
    0
}

#[clippy::msrv = "1.75"]
fn main() {
    let val = 1;
    let _p = &val as *const i32;
    //~^ borrow_as_ptr
    let _p = &0 as *const i32;
    let _p = &a() as *const i32;
    let vec = vec![1];
    let _p = &vec.len() as *const usize;

    let mut val_mut = 1;
    let _p_mut = &mut val_mut as *mut i32;
    //~^ borrow_as_ptr

    let mut x: [i32; 2] = [42, 43];
    let _raw = (&mut x[1] as *mut i32).wrapping_offset(-1);
    //~^ borrow_as_ptr
}

fn issue_13882() {
    let mut x: [i32; 2] = [42, 43];
    let _raw = (&mut x[1] as *mut i32).wrapping_offset(-1);
    //~^ borrow_as_ptr
}

fn implicit_cast() {
    let val = 1;
    let p: *const i32 = &val;
    //~^ borrow_as_ptr

    let mut val = 1;
    let p: *mut i32 = &mut val;
    //~^ borrow_as_ptr

    let mut val = 1;
    // Only lint the leftmost argument, the rightmost is ref to a temporary
    core::ptr::eq(&val, &1);
    //~^ borrow_as_ptr

    // Do not lint references to temporaries
    core::ptr::eq(&0i32, &1i32);
}

fn issue_15141() {
    let a = String::new();
    // Don't lint cast to dyn trait pointers
    let b = &a as *const dyn std::any::Any;
}

fn issue15389() {
    proc_macros::with_span! {
        span
        let var = 0u32;
        // Don't lint in proc-macros
        let _ = &var as *const u32;
    };
}
