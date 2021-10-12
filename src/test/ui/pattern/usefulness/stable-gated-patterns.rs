// aux-build:unstable.rs

extern crate unstable;

use unstable::Foo;

fn main() {
    match Foo::Stable {
        Foo::Stable => {}
    }
    //~^^^ non-exhaustive patterns: `Stable2` and `_` not covered

    match Foo::Stable {
        Foo::Stable => {}
        Foo::Stable2 => {}
    }
    //~^^^^ non-exhaustive patterns: `_` not covered
}
