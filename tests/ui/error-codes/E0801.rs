#![feature(return_type_notation)]

fn test()
where
    test(..): Send,
    //~^ ERROR expected associated function, found function `test`
    //~| ERROR return type notation not allowed in this position yet
{
}

fn main() {}
