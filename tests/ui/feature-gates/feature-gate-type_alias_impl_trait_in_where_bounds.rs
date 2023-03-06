// ignore-compare-mode-chalk
// check-pass
#![feature(type_alias_impl_trait, type_alias_impl_trait_in_where_bounds)]
use std::fmt::Debug;

type Foo = impl Debug;

struct Bar(Foo);
fn define() -> Bar {
    Bar(42)
}

type Foo2 = impl Debug;

fn define2()
where
    Foo2: Debug,
{
    let x = || -> Foo2 { 42 };
}

type Foo3 = impl Debug;

fn define3(x: Foo3) {
    let y: i32 = x;
}
fn define3_1()
where
    Foo3: Debug,
{
    define3(42)
}

type Foo4 = impl Debug;

fn define4()
where
    Foo4: Debug,
{
    let y: Foo4 = 42;
}

fn main() {}
