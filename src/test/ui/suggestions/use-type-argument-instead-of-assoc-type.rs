pub trait T<X, Y> {
    type A;
    type B;
    type C;
}
pub struct Foo {
    i: Box<dyn T<usize, usize, usize, usize, B=usize>>,
    //~^ ERROR must be specified
    //~| ERROR wrong number of type arguments
}


fn main() {}
