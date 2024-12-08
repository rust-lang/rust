//@ run-pass
#![allow(non_shorthand_field_patterns)]

struct Foo {
    x: isize,
    y: isize,
}

pub fn main() {
    let a = Foo { x: 1, y: 2 };
    match a {
        Foo { x: x, y: y } => println!("yes, {}, {}", x, y)
    }

    match a {
        Foo { .. } => ()
    }
}
