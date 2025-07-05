//@ check-pass

// Make sure we don't have any false positives here.

#![deny(dead_code)]

struct Foo {
    bar: usize,
}

fn get_thing<T>() -> T {
    todo!()
}

pub fn main() {
    let Foo { bar: _x } = get_thing();
}
