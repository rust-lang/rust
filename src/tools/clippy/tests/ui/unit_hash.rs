#![warn(clippy::unit_hash)]

use std::collections::hash_map::DefaultHasher;
use std::hash::Hash;

enum Foo {
    Empty,
    WithValue(u8),
}

fn do_nothing() {}

fn main() {
    let mut state = DefaultHasher::new();
    let my_enum = Foo::Empty;

    match my_enum {
        Foo::Empty => ().hash(&mut state),
        Foo::WithValue(x) => x.hash(&mut state),
    }

    let res = ();
    res.hash(&mut state);

    #[allow(clippy::unit_arg)]
    do_nothing().hash(&mut state);
}
