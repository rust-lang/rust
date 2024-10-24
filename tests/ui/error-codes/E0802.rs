#![feature(return_type_notation)]

trait Trait {}

fn test<T: Trait>()
where
    <T as Trait>::method(..): Send,
    //~^ ERROR associated function `method` not found for `T`
{
}

fn main() {}
