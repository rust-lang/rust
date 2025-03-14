#![warn(clippy::borrow_as_ptr)]
#![allow(clippy::useless_vec)]

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
