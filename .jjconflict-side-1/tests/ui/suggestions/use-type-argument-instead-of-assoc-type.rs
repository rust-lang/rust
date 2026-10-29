pub trait T<X, Y> {
    type A;
    type B;
    type C;
}
pub struct Foo {
    i: Box<dyn T<usize, usize, usize, usize, B = usize>>,
    //~^ ERROR trait takes 2 generic arguments but 4 generic arguments were supplied
}

fn take(_: impl Iterator<0>) {}
//~^ ERROR trait takes 0 generic arguments but 1 generic argument was supplied

fn main() {}
