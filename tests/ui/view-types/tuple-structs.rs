//@ run-pass

#![feature(view_types)]
#![allow(unused)]

struct Pair(usize, u32);

impl Pair {
    fn foo(&mut self.{ 0, 1 }) {}
    fn bar(_pair: &mut Pair.{ 0, 1 }) {}
}

fn main() {}
