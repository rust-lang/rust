#![feature(plugin)]
#![plugin(clippy)]
#![warn(unit_expr)]
#[allow(unused_variables)]

fn main() {
    let x = {
        "foo";
        "baz";
    };
}
