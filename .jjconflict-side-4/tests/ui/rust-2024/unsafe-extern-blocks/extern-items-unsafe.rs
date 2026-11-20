//@ revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024

unsafe extern "C" {
    static TEST1: i32;
    fn test1(i: i32);
}

fn test2() {
    test1(TEST1);
    //~^ ERROR: call to unsafe function `test1` is unsafe
    //~| ERROR: use of extern static is unsafe
}

fn test3() {
    unsafe {
        test1(TEST1);
    }
}

fn main() {}
