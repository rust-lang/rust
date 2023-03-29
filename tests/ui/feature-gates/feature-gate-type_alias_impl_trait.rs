// ignore-compare-mode-chalk

use std::fmt::Debug;

type Foo = impl Debug;
//~^ ERROR: `impl Trait` in type aliases is unstable

struct Bar(Foo);

#[defines(Foo)]
//~^ ERROR: is an experimental feature
fn define() -> Bar {
    Bar(42)
}

fn define2() {
    type Foo2 = impl Debug;
    //~^ ERROR: `impl Trait` in type aliases is unstable
    let x = || -> Foo2 { 42 };
}

type Foo3 = impl Debug;
//~^ ERROR: `impl Trait` in type aliases is unstable

#[defines(Foo3)]
//~^ ERROR: is an experimental feature
fn define3(x: Foo3) {
    let y: i32 = x;
}

fn define4() {
    type Foo4 = impl Debug;
    //~^ ERROR: `impl Trait` in type aliases is unstable
    let y: Foo4 = 42;
}

fn main() {}
