//@ run-pass
#![allow(dead_code)]
#![allow(dead_code)]

#![feature(coroutines)]

fn bar<'a>() {
    let a: &'static str = "hi";
    let b: &'a str = a;

    #[coroutine] || { //~ WARN unused coroutine that must be used
        yield a;
        yield b;
    };
}

fn main() {}
