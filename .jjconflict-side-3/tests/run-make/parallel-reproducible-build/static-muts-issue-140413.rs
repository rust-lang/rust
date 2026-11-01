// Checks that mutable static items can have mutable slices and other references

pub static mut TEST: &'static mut [isize] = &mut [1];
pub static mut EMPTY: &'static mut [isize] = &mut [];
pub static mut INT: &'static mut isize = &mut 1;

// And the same for raw pointers.

pub static mut TEST_RAW: *mut [isize] = &mut [1isize] as *mut _;
pub static mut EMPTY_RAW: *mut [isize] = &mut [] as *mut _;
pub static mut INT_RAW: *mut isize = &mut 1isize as *mut _;

pub fn main() {
    unsafe {
        TEST[0] += 1;
        assert_eq!(TEST[0], 2);
        *INT_RAW += 1;
        assert_eq!(*INT_RAW, 2);
    }
}
