#![feature(box_syntax)]
use std::any::Any;

fn main()
{
    fn h(x:i32) -> i32 {3*x}
    let mut vfnfer:Vec<Box<dyn Any>> = vec![];
    vfnfer.push(box h);
    println!("{:?}",(vfnfer[0] as dyn Fn)(3));
    //~^ ERROR the precise format of `Fn`-family traits'
    //~| ERROR wrong number of type arguments: expected 1, found 0 [E0107]
    //~| ERROR the value of the associated type `Output` (from the trait `std::ops::FnOnce`)
}
