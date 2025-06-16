//@ check-pass

use std::mem::MaybeUninit;

pub static X: (i32, MaybeUninit<i32>) = (1, foo(&X.0));

const fn foo(x: &i32) -> MaybeUninit<i32> {
    let mut temp = MaybeUninit::<i32>::uninit();
    unsafe {
        std::ptr::copy(x, temp.as_mut_ptr(), 1);
    }
    temp
}

fn main() {}
