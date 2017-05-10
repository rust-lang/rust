#![feature(plugin)]
#![plugin(clippy)]

#![allow(dead_code)]
#![allow(single_match)]
#![allow(unused_variables, similar_names)]
#![deny(blacklisted_name)]

fn test(foo: ()) {}

fn main() {
    let foo = 42;
    let bar = 42;
    let baz = 42;

    let barb = 42;
    let barbaric = 42;

    match (42, Some(1337), Some(0)) {
        (foo, Some(bar), baz @ Some(_)) => (),
        _ => (),
    }
}
