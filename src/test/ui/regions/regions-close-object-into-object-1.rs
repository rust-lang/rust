#![feature(box_syntax)]
#![allow(warnings)]

trait A<T> { }
struct B<'a, T:'a>(&'a (A<T>+'a));

trait X { }

impl<'a, T> X for B<'a, T> {}

fn f<'a, T:'static, U>(v: Box<A<T>+'static>) -> Box<X+'static> {
    box B(&*v) as Box<X> //~ ERROR `*v` does not live long enough
}

fn main() {}
