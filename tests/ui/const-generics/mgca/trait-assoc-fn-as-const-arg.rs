#![feature(min_generic_const_args)]

// Regression test for #138088. assoc functions cannot be lowered as const
// args, and this should emit a regular diagnostic instead of ICEing.
trait Bar {
    fn x(&self) -> [i32; Bar::x];
    //~^ ERROR the constant `<Self as Bar>::x` is not of type `usize`
}

fn main() {}
