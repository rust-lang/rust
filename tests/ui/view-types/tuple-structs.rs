#![feature(view_types)]
#![allow(unused)]

struct Pair(usize, u32);

impl Pair {
    fn foo(&mut self.{ 0, 1 }) {}
    //~^ ERROR expected parameter name
    //~| ERROR expected one of
    fn bar(_pair: &mut Pair.{ 0, 1 }) {}
    //~^ ERROR expected identifier, found `0`
}

fn main() {}
