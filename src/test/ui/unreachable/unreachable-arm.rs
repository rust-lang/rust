#![feature(box_patterns)]
#![feature(box_syntax)]
#![allow(dead_code)]
#![deny(unreachable_patterns)]

enum Foo { A(Box<Foo>, isize), B(usize), }

fn main() {
    match Foo::B(1) {
        Foo::B(_) | Foo::A(box _, 1) => { }
        Foo::A(_, 1) => { } //~ ERROR unreachable pattern
        _ => { }
    }
}
