use std::ops::Add;

fn main() {
    let i: i32 = 0;
    let j: &impl Add = &i;
    //~^ `impl Trait` only allowed in function and inherent method return types
}
