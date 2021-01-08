// check-pass

#![allow(unused)]

use std::borrow::Borrow;
use std::ops::Deref;

struct Foo<T>(T);

#[derive(Clone)]
struct Bar<T>(T);

struct DerefExample<T>(T);

impl<T> Deref for DerefExample<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn main() {
    let foo = &Foo(1u32);
    let foo_clone: &Foo<u32> = foo.clone();
    //~^ WARNING call to method that does nothing [noop_method_call]

    let bar = &Bar(1u32);
    let bar_clone: Bar<u32> = bar.clone();

    let deref = &&DerefExample(12u32);
    let derefed: &DerefExample<u32> = deref.deref();
    //~^ WARNING call to method that does nothing [noop_method_call]

    let deref = &DerefExample(12u32);
    let derefed: &u32 = deref.deref();

    let a = &&Foo(1u32);
    let borrowed: &Foo<u32> = a.borrow();
    //~^ WARNING call to method that does nothing [noop_method_call]
}

fn generic<T>(foo: &Foo<T>) {
    foo.clone();
}

fn non_generic(foo: &Foo<u32>) {
    foo.clone(); //~ WARNING call to method that does nothing [noop_method_call]
}
