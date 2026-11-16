#![feature(function_arg_const_generics, gca_min_const_items)]

trait Foo {
    fn a(const N: usize, self: [isize; N]) -> isize; //~ ERROR
    fn b(&self) -> isize {
        self.a() + 2 //~ ERROR E0599
    }
}

impl Foo for isize {
    fn a(&self) -> isize { //~ ERROR E0185
        3
    }
}

fn main() {}
