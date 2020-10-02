// run-pass
#![allow(dead_code)]

#![feature(generators)]

enum Enum {
    A(String),
    B
}

fn main() {
    || { //~ WARN unused generator that must be used
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
