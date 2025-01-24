pub trait T<X, Y> {
    type A;
    type B;
    type C;
}
pub struct Foo {
    i: Box<dyn T<usize, usize, usize, usize, B=usize>>,
    //~^ ERROR trait takes 2 generic arguments but 4 generic arguments were supplied
}


fn main() {}
