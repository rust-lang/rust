//@ run-pass
#![allow(dead_code)]
#![allow(unconditional_recursion)]


trait Foo {
}

fn b(_x: Box<dyn Foo+Send>) {
}

fn c(x: Box<dyn Foo+Sync+Send>) {
    e(x);
}

fn d(x: Box<dyn Foo+Send>) {
    e(x);
}

fn e(x: Box<dyn Foo>) {
    e(x);
}

pub fn main() { }
