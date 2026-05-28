#![feature(never_type)]
#![feature(never_patterns)]
#![allow(incomplete_features)]

enum Void {}

fn foo(x: Void) {
    match x {
        (!|!) if true => {} //~ ERROR a never pattern is always unreachable
        (!|!) if true => {} //~ ERROR a never pattern is always unreachable
    }
}

fn main() {}
