//@ edition: 2021
//@ compile-flags: --crate-type lib --extern circular_dependencies={{build-base}}/circular-dependencies/libcircular_dependencies.rmeta --emit dep-info,metadata

use circular_dependencies::Foo;

pub fn consume_foo(_: Foo) {}

pub fn produce_foo() -> Foo {
    Foo
}
