#![feature(return_type_notation)]

fn test()
where
    test(..): Send,
    //~^ ERROR expected associated function, found function `test`
{
}

fn main() {}
