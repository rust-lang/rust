use std::ops::Add;

fn main() {
    let i: i32 = 0;
    let j: &impl Add = &i;
    //~^ `impl Trait` not allowed within variable binding [E0562]
}
