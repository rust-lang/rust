// edition: 2021

#![deny(dead_code)]

pub fn foo() {
    let closure = || {
        fn a() {}   //~ ERROR function `a` is never used
    };
    closure()
}

pub async fn async_foo() {
    const A: usize = 1; //~ ERROR constant `A` is never used
}

fn main() {}
