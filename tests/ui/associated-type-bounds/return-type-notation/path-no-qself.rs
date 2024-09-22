#![feature(return_type_notation)]
//~^ WARN the feature `return_type_notation` is incomplete

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
