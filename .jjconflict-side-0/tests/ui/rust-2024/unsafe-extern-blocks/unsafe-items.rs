//@ revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024

unsafe extern "C" {
    unsafe static TEST1: i32;
    unsafe fn test1(i: i32);
}

fn test2() {
    unsafe {
        test1(TEST1);
    }
}

fn test3() {
    test1(TEST1);
    //~^ ERROR: call to unsafe function `test1` is unsafe
    //~| ERROR: use of extern static is unsafe
}

fn main() {}
