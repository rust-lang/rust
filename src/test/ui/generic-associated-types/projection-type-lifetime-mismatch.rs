#![feature(generic_associated_types)]

pub trait X {
    type Y<'a> where Self: 'a;
    fn m(&self) -> Self::Y<'_>;
}

impl X for () {
    type Y<'a> = &'a ();

    fn m(&self) -> Self::Y<'_> {
        self
    }
}

fn f(x: &impl for<'a> X<Y<'a> = &'a ()>) -> &'static () {
    x.m()
    //~^ ERROR `x` has an anonymous lifetime `'_` but it needs to satisfy a `'static` lifetime requirement [E0759]
}

fn g<T: for<'a> X<Y<'a> = &'a ()>>(x: &T) -> &'static () {
    x.m()
    //~^ ERROR `x` has an anonymous lifetime `'_` but it needs to satisfy a `'static` lifetime requirement [E0759]
}

fn h(x: &()) -> &'static () {
    x.m()
    //~^ ERROR `x` has an anonymous lifetime `'_` but it needs to satisfy a `'static` lifetime requirement [E0759]
}

fn main() {
    f(&());
    g(&());
    h(&());
}
