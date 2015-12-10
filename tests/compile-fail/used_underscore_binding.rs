#![feature(plugin)]
#![plugin(clippy)]
#[deny(used_underscore_binding)]

fn main() {
    let foo = 0u32;
    prefix_underscore(foo); //should fail
    non_prefix_underscore(foo); //should pass
    unused_underscore(foo); //should pass
}

fn prefix_underscore(_x: u32){
    println!("{}", _x + 1); //~Error: Used binding which is prefixed with an underscore
}

fn non_prefix_underscore(some_foo: u32) {
    println!("{}", some_foo + 1);
}

fn unused_underscore(_foo: u32) {
    println!("{}", 1);
}
