#![feature(return_type_notation)]

trait Trait {
    fn method() -> impl Sized;
}

fn test()
where
    Trait::method(..): Send,
    //~^ ERROR ambiguous associated type
{
}

fn main() {}
