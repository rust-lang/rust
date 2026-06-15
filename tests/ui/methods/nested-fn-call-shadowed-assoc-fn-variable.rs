//@ check-pass
#![allow(dead_code)]
#![allow(unused_variables)]

struct Index;

impl Index {
    fn new() -> Self { Index }
}

fn user() {
    let new = Index::new;

    fn inner() {
        let index = Index::new();
    }

    let index2 = new();
}

fn main() {}
