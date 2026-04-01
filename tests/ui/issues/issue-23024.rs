use std::any::Any;

fn main()
{
    fn h(x:i32) -> i32 {3*x}
    let mut vfnfer:Vec<Box<dyn Any>> = vec![];
    vfnfer.push(Box::new(h));
    println!("{:?}",(vfnfer[0] as dyn Fn)(3));
    //~^ ERROR the precise format of `Fn`-family traits'
    //~| ERROR missing generics for trait `Fn`
}
