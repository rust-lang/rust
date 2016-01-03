#![feature(plugin)]
#![plugin(clippy)]

#![deny(unneeded_field_pattern)]
#[allow(dead_code, unused)]

struct Foo {
    a: i32,
    b: i32,
    c: i32,
}

fn main() {
    let f = Foo { a: 0, b: 0, c: 0 };

    match f {
        Foo { a: _, b: 0, .. } => {} //~ERROR You matched a field with a wildcard pattern
                                     //~^ HELP Try with `Foo { b: 0, .. }`
        Foo { a: _, b: _, c: _ } => {} //~ERROR All the struct fields are matched to a
                                       //~^ HELP Try with `Foo { .. }`
    }
    match f {
        Foo { b: 0, .. } => {} // should be OK
        Foo { .. } => {} // and the Force might be with this one
    }
}
