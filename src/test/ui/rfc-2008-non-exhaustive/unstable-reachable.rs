#![feature(unstable_test_feature)]

// aux-build:unstable.rs

extern crate unstable;

use unstable::Foo;

fn main() {
    #[deny(non_exhaustive_omitted_patterns)]
    match Foo::Stable {
        Foo::Stable => {}
        _ => {}
    }
    //~^^ some variants are not matched explicitly

    // Ok: all variants are explicitly matched
    #[deny(non_exhaustive_omitted_patterns)]
    match Foo::Stable {
        Foo::Stable => {}
        Foo::Stable2 => {}
        Foo::Unstable => {}
        _ => {}
    }
}
