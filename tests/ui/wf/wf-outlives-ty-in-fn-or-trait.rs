#![feature(rustc_attrs)]
#![allow(dead_code)]

trait Trait<'a, T> {
    type Out;
}

impl<'a, T> Trait<'a, T> for usize {
    type Out = &'a fn(T); //~ ERROR `T` may not live long enough
}

struct Foo<'a,T> {
    f: &'a fn(T),
}

trait Baz<T> { }

impl<'a, T> Trait<'a, T> for u32 {
    type Out = &'a dyn Baz<T>; //~ ERROR `T` may not live long enough
}

fn main() { }
