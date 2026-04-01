// https://github.com/rust-lang/rust/issues/8761
enum Foo {
    A = 1i64,
    //~^ ERROR: mismatched types
    //~| NOTE: expected `isize`, found `i64`
    //~| NOTE: enum variant discriminant
    B = 2u8
    //~^ ERROR: mismatched types
    //~| NOTE: expected `isize`, found `u8`
    //~| NOTE: enum variant discriminant
}

fn main() {}
