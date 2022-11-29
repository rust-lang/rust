use std::fmt::Debug;

fn main() {
    let x: Option<impl Debug> = Some(44_u32);
    //~^ `impl Trait` not allowed within variable binding [E0562]
    println!("{:?}", x);
}
