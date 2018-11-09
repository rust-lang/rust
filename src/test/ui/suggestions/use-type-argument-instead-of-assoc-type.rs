pub trait T<X, Y> {
    type A;
    type B;
    type C;
}
 pub struct Foo { i: Box<T<usize, usize, usize, usize, B=usize>> }

 fn main() {}
