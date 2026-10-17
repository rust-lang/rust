//@ run-pass
#![allow(dead_code)]



struct Foo<T> {
    a: T
}

type Bar<T> = Foo<T>;

fn takebar<T>(_b: Bar<T>) { }

pub fn main() { }
