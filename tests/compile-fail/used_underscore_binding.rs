#![feature(plugin)]
#![plugin(clippy)]
#![deny(clippy)]

fn prefix_underscore(_x: u32) -> u32 {
    _x + 1 //~ ERROR used binding which is prefixed with an underscore
}

fn in_macro(_x: u32) {
    println!("{}", _x); //~ ERROR used binding which is prefixed with an underscore
}

fn non_prefix_underscore(some_foo: u32) -> u32 {
    some_foo + 1
}

fn unused_underscore(_foo: u32) -> u32 {
    1
}

fn main() {
    let foo = 0u32;
    // tests of unused_underscore lint
    let _ = prefix_underscore(foo);
    in_macro(foo);
    // possible false positives
    let _ = non_prefix_underscore(foo);
    let _ = unused_underscore(foo);
}

