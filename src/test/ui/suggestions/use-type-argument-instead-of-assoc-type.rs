pub trait T<X, Y> {
    type A;
    type B;
    type C;
}
pub struct Foo {
    i: Box<dyn T<usize, usize, usize, usize, B=usize>>,
    //~^ ERROR must be specified
    //~| ERROR this trait takes 2 type arguments but 4 type arguments were supplied
}


fn main() {}
