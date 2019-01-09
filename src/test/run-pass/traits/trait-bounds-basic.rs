// run-pass
#![allow(dead_code)]
#![allow(unconditional_recursion)]

// pretty-expanded FIXME #23616

trait Foo {
}

fn b(_x: Box<Foo+Send>) {
}

fn c(x: Box<Foo+Sync+Send>) {
    e(x);
}

fn d(x: Box<Foo+Send>) {
    e(x);
}

fn e(x: Box<Foo>) {
    e(x);
}

pub fn main() { }
