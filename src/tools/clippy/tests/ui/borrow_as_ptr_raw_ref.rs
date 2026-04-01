#![warn(clippy::borrow_as_ptr)]
#![allow(clippy::useless_vec)]

fn a() -> i32 {
    0
}

#[clippy::msrv = "1.82"]
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
}
