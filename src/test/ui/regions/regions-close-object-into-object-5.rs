#![allow(warnings)]


trait A<T>
{
    fn get(&self) -> T { panic!() }
}

struct B<'a, T: 'a>(&'a (A<T> + 'a));

trait X { fn foo(&self) {} }

impl<'a, T> X for B<'a, T> {}

fn f<'a, T, U>(v: Box<A<T> + 'static>) -> Box<X + 'static> {
    // oh dear!
    Box::new(B(&*v)) as Box<dyn X>
    //~^ ERROR the parameter type `T` may not live long enough
    //~| ERROR the parameter type `T` may not live long enough
    //~| ERROR the parameter type `T` may not live long enough
    //~| ERROR the parameter type `T` may not live long enough
    //~| ERROR the parameter type `T` may not live long enough
    //~| ERROR the parameter type `T` may not live long enough
    //~| ERROR the parameter type `T` may not live long enough
}

fn main() {}
