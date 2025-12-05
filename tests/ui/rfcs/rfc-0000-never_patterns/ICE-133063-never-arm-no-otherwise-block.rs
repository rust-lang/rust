#![feature(never_type)]
#![feature(never_patterns)]
#![allow(incomplete_features)]

enum Void {}

fn foo(x: Void) {
    loop {
        match x {
            (!|!) if false => {} //~ ERROR a never pattern is always unreachable
            _ => {}
        }
    }
}

fn main() {}
