//@ run-pass
#![allow(unused_variables)]
use std::marker::PhantomData;

fn main() {
    struct Symbol<'a, F: Fn(Vec<&'a str>) -> &'a str> { function: F, marker: PhantomData<&'a ()> }
    let f = |x: Vec<&str>| -> &str { "foobar" };
    let sym = Symbol { function: f, marker: PhantomData };
    (sym.function)(vec![]);
}
