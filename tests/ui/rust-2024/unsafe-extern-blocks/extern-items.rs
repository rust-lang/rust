//@ revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2021] check-pass
//@[edition2024] edition:2024

extern "C" {
    //[edition2024]~^ ERROR extern blocks must be unsafe
    static TEST1: i32;
    fn test1(i: i32);
}

unsafe extern "C" {
    static TEST2: i32;
    fn test2(i: i32);
}

fn main() {}
