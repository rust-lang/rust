#![feature(trivial_bounds)]
#![allow(unused)]

struct B;

trait A {
    type X;
    fn get_x() -> Self::X;
}

impl A for B {
    type X = u8;
    fn get_x() -> u8 { 0 }
}

fn global_bound_is_hidden() -> u8
where
    B: A<X = i32>
{
    B::get_x() //~ ERROR
}

fn main () {}
