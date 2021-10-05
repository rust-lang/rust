// aux-build:unstable.rs

extern crate unstable;

use unstable::Foo;

fn main() {
    match Foo::Stable {
        Foo::Stable => {}
    }
    //~^^^ non-exhaustive patterns: `Stable2` not covered

    // Ok: all variants are explicitly matched
    match Foo::Stable {
        Foo::Stable => {}
        Foo::Stable2 => {}
    }
}
