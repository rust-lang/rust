// run-rustfix
#![warn(clippy::borrow_as_ptr)]

fn main() {
    let val = 1;
    let _p = &val as *const i32;

    let mut val_mut = 1;
    let _p_mut = &mut val_mut as *mut i32;
}
