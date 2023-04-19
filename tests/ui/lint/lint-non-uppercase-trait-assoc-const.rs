#![deny(non_upper_case_globals)]

trait Trait {
    const item: usize;
    //~^ ERROR associated constant `item` should have an upper case name [non_upper_case_globals]
}

struct Foo;

impl Trait for Foo {
    const item: usize = 5;
    // ^^^ there should be no error here (in the trait `impl`)
}

fn main() {}
