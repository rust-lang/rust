use std::fmt::Debug;

fn main() {
    let x: Option<impl Debug> = Some(44_u32);
    //~^ ERROR `impl Trait` is not allowed in the type of variable bindings
    println!("{:?}", x);
}
