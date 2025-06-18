use std::mem::{transmute, ManuallyDrop};

const S: &'static mut str = &mut " hello ";
//~^ ERROR: mutable borrows of lifetime-extended temporaries

const fn trigger() -> [(); unsafe {
        let s = transmute::<(*const u8, usize), &ManuallyDrop<str>>((S.as_ptr(), 3));
        0
    }] {
    [(); 0]
}

fn main() {}
