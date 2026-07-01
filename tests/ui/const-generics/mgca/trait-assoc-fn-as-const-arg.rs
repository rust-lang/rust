#![feature(min_generic_const_args)]

// Regression test for #138088. assoc functions cannot be lowered as const
// args, and this should emit a regular diagnostic instead of ICEing.
trait Bar {
    fn x(&self) -> [i32; Bar::x];
    //~^ ERROR function items cannot be used as const arguments
}

fn main() {}
