use std::mem::{ManuallyDrop, transmute};
const DATA: ManuallyDrop<&i32> = unsafe { transmute(4_usize) };

fn main() {
    if let DATA = DATA {} //~ERROR: cannot be used as pattern

    {
        struct W<T, U>(T, U);
        const DATA: ManuallyDrop<&W<i32, W<i32, i32>>> = unsafe { transmute(4_usize) };

        if let DATA = DATA {} //~ERROR: cannot be used as pattern
    }
}
