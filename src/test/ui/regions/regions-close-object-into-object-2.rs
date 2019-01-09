#![feature(box_syntax)]

trait A<T> { }
struct B<'a, T:'a>(&'a (A<T>+'a));

trait X { }
impl<'a, T> X for B<'a, T> {}

fn g<'a, T: 'static>(v: Box<A<T>+'a>) -> Box<X+'static> {
    box B(&*v) as Box<X> //~ ERROR cannot infer
}

fn main() { }
