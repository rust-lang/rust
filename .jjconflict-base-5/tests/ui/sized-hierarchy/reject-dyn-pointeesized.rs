#![feature(sized_hierarchy)]

use std::marker::PointeeSized;

type Foo = dyn PointeeSized;
//~^ ERROR at least one trait is required for an object type

fn foo(f: &Foo) {}

fn main() {
    foo(&());

    let x = main;
    let y: Box<dyn PointeeSized> = x;
//~^ ERROR at least one trait is required for an object type
}
