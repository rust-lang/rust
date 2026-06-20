//@ check-pass
#![allow(dead_code)]

#![feature(coroutines)]

enum Enum {
    A(String),
    B
}

fn main() {
    #[coroutine] || {
        loop {
            if let true = true {
                match Enum::A(String::new()) {
                    Enum::A(_var) => {}
                    Enum::B => {}
                }
            }
            yield;
        }
    };
}
