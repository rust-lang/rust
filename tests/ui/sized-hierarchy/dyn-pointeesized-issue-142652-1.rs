//@ check-pass
#![feature(sized_hierarchy)]

use std::marker::PointeeSized;

type Foo = dyn PointeeSized;

fn foo(f: &Foo) {}

fn main() {
    foo(&());
}
