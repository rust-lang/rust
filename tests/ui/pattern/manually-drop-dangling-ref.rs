use std::mem::{ManuallyDrop, transmute};
const DATA: ManuallyDrop<&i32> = unsafe { transmute(4_usize) };

fn main() {
    if let DATA = DATA {} //~ERROR: cannot be used as pattern
}
