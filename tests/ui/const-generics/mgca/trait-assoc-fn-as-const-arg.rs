#![feature(min_generic_const_args)]

trait Bar {
    fn x(&self) -> [i32; Bar::x];
    //~^ ERROR function items cannot be used as const arguments
}

fn main() {}
