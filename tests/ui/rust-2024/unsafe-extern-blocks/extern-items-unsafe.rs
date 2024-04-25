//@ revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024
//@[edition2024] compile-flags: -Zunstable-options

unsafe extern "C" {
    fn test1(i: i32);
}

fn test2(i: i32) {
    test1(i);
    //~^ ERROR: call to unsafe function `test1` is unsafe
}

fn test3(i: i32) {
    unsafe {
        test1(i);
    }
}

fn main() {}
