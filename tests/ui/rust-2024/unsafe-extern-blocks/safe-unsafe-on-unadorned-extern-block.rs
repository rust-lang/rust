//@ revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024

extern "C" {
    //[edition2024]~^ ERROR extern blocks must be unsafe
    safe static TEST1: i32;
    //~^ ERROR items in `extern` blocks without an `unsafe` qualifier cannot have safety qualifiers
    safe fn test1(i: i32);
    //~^ ERROR items in `extern` blocks without an `unsafe` qualifier cannot have safety qualifiers
}

fn test2() {
    test1(TEST1);
}

fn main() {}
