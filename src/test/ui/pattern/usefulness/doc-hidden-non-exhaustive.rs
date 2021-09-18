// aux-build:hidden.rs

extern crate hidden;

use hidden::Foo;

fn main() {
    match Foo::A {
        Foo::A => {}
        Foo::B => {}
    }
    //~^^^^ non-exhaustive patterns: `_` not covered

    match Foo::A {
        Foo::A => {}
        Foo::C => {}
    }
    //~^^^^ non-exhaustive patterns: `B` not covered

    match Foo::A {
        Foo::A => {}
    }
    //~^^^ non-exhaustive patterns: `B` and `_` not covered

    match None {
        None => {}
        Some(Foo::A) => {}
    }
    //~^^^^ non-exhaustive patterns: `Some(B)` and `Some(_)` not covered
}
