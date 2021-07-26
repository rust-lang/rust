#![feature(type_equality_constraints)]

pub fn foo<A,B>(a: A) -> B
    where A = B
{
    a
    //~^ ERROR mismatched types
}

 fn main() {}
