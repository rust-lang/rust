trait A<T> { }

struct B<'a, T:'a>(&'a (dyn A<T> + 'a));

trait X { }
impl<'a, T> X for B<'a, T> {}

fn g<'a, T: 'static>(v: Box<dyn A<T> + 'a>) -> Box<dyn X + 'static> {
    Box::new(B(&*v)) as Box<dyn X>
    //~^ ERROR lifetime may not live long enough
    //~| ERROR cannot return value referencing local data `*v` [E0515]
}

fn main() { }
