#![feature(min_generic_const_args)]

// Regression test for #138088.
trait Bar {
    fn x(&self) -> [i32; Bar::x];
    //~^ ERROR cycle detected when evaluating type-level constant
}

fn main() {}
