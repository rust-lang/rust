use std::ops::Add;

fn main() {
    let i: i32 = 0;
    let j: &impl Add = &i;
    //~^ ERROR `impl Trait` is not allowed in the type of variable bindings
}
