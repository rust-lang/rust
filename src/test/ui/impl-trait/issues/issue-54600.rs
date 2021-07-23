use std::fmt::Debug;

fn main() {
    let x: Option<impl Debug> = Some(44_u32);
    //~^ `impl Trait` not allowed outside of function and method return types
    println!("{:?}", x);
}
