#![feature(unstable_test_feature)]

// aux-build:unstable.rs

extern crate unstable;

use unstable::Foo;

fn main() {
    match Foo::Stable {
        Foo::Stable => {}
        Foo::Stable2 => {}
    }
    //~^^^^ non-exhaustive patterns: `Unstable` not covered

    // Ok: all variants are explicitly matched
    match Foo::Stable {
        Foo::Stable => {}
        Foo::Stable2 => {}
        Foo::Unstable => {}
    }
}
