// run-pass
#![allow(dead_code)]
// Test that we recognize that if you have
//
//     'a : 'static
//
// then
//
//     'a : 'b

fn test<'a,'b>(x: &'a i32) -> &'b i32
    where 'a: 'static
{
    x
}

fn main() { }
