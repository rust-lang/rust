#![feature(box_syntax)]

trait A<T> { }
struct B<'a, T:'a>(&'a (A<T>+'a));

trait X { }
impl<'a, T> X for B<'a, T> {}

fn i<'a, T, U>(v: Box<A<U>+'a>) -> Box<X+'static> {
    box B(&*v) as Box<X> //~ ERROR cannot infer
}

fn main() {}
