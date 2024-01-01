// build-pass

#![feature(coroutines)]

fn bar<'a>() {
    let a: &'static str = "hi";
    let b: &'a str = a;

    || {
        yield a;
        yield b;
    };
}

fn main() {}
