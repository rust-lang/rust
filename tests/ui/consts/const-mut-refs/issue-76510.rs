use std::mem::{transmute, ManuallyDrop};

const S: &'static mut str = &mut " hello ";
//~^ ERROR: mutable references are not allowed in the final value of constants

const fn trigger() -> [(); unsafe {
        let s = transmute::<(*const u8, usize), &ManuallyDrop<str>>((S.as_ptr(), 3));
        0
    }] {
    [(); 0]
}

fn main() {}
