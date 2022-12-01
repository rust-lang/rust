use std::ops::Add;

fn main() {
    let i: i32 = 0;
    let j: &impl Add = &i;
    //~^ `impl Trait` isn't allowed within variable binding [E0562]
}
