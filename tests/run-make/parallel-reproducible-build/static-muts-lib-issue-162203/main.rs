#![crate_type = "lib"]
static mut TEST: &'static mut [isize] = &mut [1];
static mut TEST_RAW: *mut [isize] = &mut [1isize] as *mut _;
pub fn main() {}
