//@ run-pass
#![deny(warnings)]

#[derive(Hash, Ord, PartialOrd, Eq, PartialEq, Debug, Clone, Copy)]
struct Foo;

fn main() {
    let _ = Foo;
}
