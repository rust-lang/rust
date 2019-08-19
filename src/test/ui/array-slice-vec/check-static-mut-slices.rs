// run-pass
#![allow(dead_code)]

// Checks that mutable static items can have mutable slices


static mut TEST: &'static mut [isize] = &mut [1];
static mut EMPTY: &'static mut [isize] = &mut [];

pub fn main() {
    unsafe {
        TEST[0] += 1;
        assert_eq!(TEST[0], 2);
    }
}
