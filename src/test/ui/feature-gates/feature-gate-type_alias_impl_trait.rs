// ignore-compare-mode-chalk
#![feature(min_type_alias_impl_trait)]
use std::fmt::Debug;

type Foo = impl Debug;
//~^ ERROR could not find defining uses

struct Bar(Foo);
fn define() -> Bar {
    Bar(42) //~ ERROR mismatched types
}

type Foo2 = impl Debug;

fn define2() {
    let x = || -> Foo2 { 42 }; //~ ERROR not permitted here
}

type Foo3 = impl Debug;
//~^ ERROR could not find defining uses

fn define3(x: Foo3) {
    let y: i32 = x; //~ ERROR mismatched types
}
fn define3_1() {
    define3(42) //~ ERROR mismatched types
}

type Foo4 = impl Debug;
//~^ ERROR could not find defining uses

fn define4() {
    let y: Foo4 = 42;
    //~^ ERROR not permitted here
}

fn main() {}
