//@ revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024
//@[edition2024] compile-flags: -Zunstable-options
//@ check-pass

extern "C" {
    //[edition2024]~^ WARN extern blocks should be unsafe [missing_unsafe_on_extern]
    static TEST1: i32;
    fn test1(i: i32);
}

unsafe extern "C" {
    static TEST2: i32;
    fn test2(i: i32);
}

fn main() {}
