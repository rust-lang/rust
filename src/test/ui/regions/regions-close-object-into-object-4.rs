#![feature(box_syntax)]

trait A<T> { }
struct B<'a, T:'a>(&'a (dyn A<T> + 'a));

trait X { }
impl<'a, T> X for B<'a, T> {}

fn i<'a, T, U>(v: Box<dyn A<U>+'a>) -> Box<dyn X + 'static> {
    box B(&*v) as Box<dyn X> //~ ERROR cannot infer
}

fn main() {}
