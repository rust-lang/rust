use std::ops::Add;

fn main() {
    let i: i32 = 0;
    let j: &impl Add = &i;
    //~^ `impl Trait` not allowed outside of function and method return types
}
