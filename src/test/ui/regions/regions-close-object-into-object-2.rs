#![feature(box_syntax)]

trait A<T> { }
struct B<'a, T:'a>(&'a (dyn A<T> + 'a));

trait X { }
impl<'a, T> X for B<'a, T> {}

fn g<'a, T: 'static>(v: Box<dyn A<T> + 'a>) -> Box<dyn X + 'static> {
    box B(&*v) as Box<dyn X> //~ ERROR cannot infer
}

fn main() { }
