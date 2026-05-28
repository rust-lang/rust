//@ check-pass

// Ensure we collect lint levels from pat fields in structs.

#![deny(unused_variables)]

pub struct Foo {
    bar: u32,
    baz: u32,
}

pub fn test(foo: Foo) {
    let Foo {
        #[allow(unused_variables)]
        bar,
        #[allow(unused_variables)]
        baz,
    } = foo;
}

fn main() {}
