#![feature(sized_hierarchy)]

use std::marker::PointeeSized;

type Foo = dyn PointeeSized;
//~^ ERROR `PointeeSized` cannot be used with trait objects

fn foo(f: &Foo) {}

fn main() {
    foo(&());

    let x = main;
    let y: Box<dyn PointeeSized> = x;
//~^ ERROR `PointeeSized` cannot be used with trait objects
}
