//@ run-pass
#![allow(dead_code)]

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

// Checks that mutable static items can have mutable slices and other references


static mut TEST: &'static mut [isize] = &mut [1];
static mut EMPTY: &'static mut [isize] = &mut [];
static mut INT: &'static mut isize = &mut 1;

// And the same for raw pointers.

static mut TEST_RAW: *mut [isize] = &mut [1isize] as *mut _;
static mut EMPTY_RAW: *mut [isize] = &mut [] as *mut _;
static mut INT_RAW: *mut isize = &mut 1isize as *mut _;

pub fn main() {
    unsafe {
        TEST[0] += 1;
        assert_eq!(TEST[0], 2);
        *INT_RAW += 1;
        assert_eq!(*INT_RAW, 2);
    }
}
