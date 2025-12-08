//@ run-pass
#![allow(dead_code)]
#![allow(dead_code)]

#![feature(coroutines)]

fn bar<'a>() {
    let a: &'static str = "hi";
    let b: &'a str = a;

    #[coroutine] || { //~ WARN unused coroutine that must be used
        a.yield;
        b.yield;
    };
}

fn main() {}
