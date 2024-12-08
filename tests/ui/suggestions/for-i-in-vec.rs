//@ run-rustfix
#![allow(dead_code)]

struct Foo {
    v: Vec<u32>,
    h: std::collections::HashMap<i32, i32>,
}

impl Foo {
    fn bar(&self) {
        for _ in self.v { //~ ERROR cannot move out of `self.v` which is behind a shared reference
        }
        for _ in self.h { //~ ERROR cannot move out of `self.h` which is behind a shared reference
        }
    }
}

const LOADERS: &Vec<&'static u8> = &Vec::new();

pub fn break_code() -> Option<&'static u8> {
    for loader in *LOADERS { //~ ERROR cannot move out of a shared reference
        return Some(loader);
    }
    None
}

fn main() {}
