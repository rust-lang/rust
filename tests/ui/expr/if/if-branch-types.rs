fn main() {
    let x = if true { 10i32 } else { 10u32 };
    //~^ ERROR `if` and `else` have incompatible types
    //~| NOTE expected `i32`, found `u32`
    //~| NOTE expected because of this
}
