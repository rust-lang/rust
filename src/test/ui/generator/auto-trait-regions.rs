#![feature(generators)]
#![feature(auto_traits)]
#![feature(negative_impls)]

use std::cell::Cell;

auto trait Foo {}

struct No;

impl !Foo for No {}

struct A<'a, 'b>(Cell<&'a bool>, Cell<&'b mut bool>, No);

impl<'a, 'b> Drop for A<'a, 'b> {
    fn drop(&mut self) {}
}

impl<'a, 'b: 'a> Foo for A<'a, 'b> {}

struct OnlyFooIfStaticRef(No);
impl Foo for &'static OnlyFooIfStaticRef {}

struct OnlyFooIfRef(No);
impl<'a> Foo for &'a OnlyFooIfRef {}

fn assert_foo<T: Foo>(_f: T) {}

fn main() {
    let z = OnlyFooIfStaticRef(No);
    let x = &z;
    let gen = || {
        let x = x;
        yield;
        assert_foo(x);
    };
    assert_foo(gen); // bad
    //~^ ERROR implementation of `Foo` is not general enough
    drop(z);

    // Allow impls which matches any lifetime
    let z = OnlyFooIfRef(No);
    let x = &z;
    let gen = || {
        let x = x;
        yield;
        assert_foo(x);
    };
    assert_foo(gen); // ok
    drop(z);

    let gen = static || {
        let mut y = true;
        let a = A::<'static, '_>(Cell::new(&true), Cell::new(&mut y), No);
        yield;
        drop(a);
    };
    assert_foo(gen);
    //~^ ERROR implementation of `Foo` is not general enough
}
