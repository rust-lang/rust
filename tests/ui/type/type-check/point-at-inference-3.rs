//@ run-rustfix
fn main() {
    let mut v = Vec::new();
    v.push(0i32);
    //~^ NOTE this argument has type `i32`...
    //~| NOTE ... which causes `v` to have type `Vec<i32>`
    v.push(0);
    v.push(1u32); //~ ERROR mismatched types
    //~^ NOTE expected `i32`, found `u32`
    //~| NOTE arguments to this method are incorrect
    //~| NOTE method defined here
    //~| HELP change the type of the numeric literal from `u32` to `i32`
}
