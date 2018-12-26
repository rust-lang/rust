// run-pass
#![allow(dead_code)]


// pretty-expanded FIXME #23616

struct Foo<T> {
    a: T
}

type Bar<T> = Foo<T>;

fn takebar<T>(_b: Bar<T>) { }

pub fn main() { }
