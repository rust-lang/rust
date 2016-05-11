#![feature(plugin)]
#![plugin(clippy)]
#![deny(clippy,similar_names)]
#![allow(unused)]

struct Foo {
    apple: i32,
    bpple: i32,
}

fn main() {
    let Foo { apple, bpple } = unimplemented!();
    let Foo { apple: spring, bpple: sprang } = unimplemented!(); //~ ERROR: name is too similar
}
