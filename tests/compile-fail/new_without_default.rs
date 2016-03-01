#![feature(plugin)]
#![plugin(clippy)]

#![allow(dead_code)]
#![deny(new_without_default)]

struct Foo;

impl Foo {
    fn new() -> Foo { Foo } //~ERROR: you should consider adding a `Default` implementation for `Foo`
}

struct Bar;

impl Bar {
    fn new() -> Self { Bar } //~ERROR: you should consider adding a `Default` implementation for `Bar`
}

struct Ok;

impl Ok {
    fn new() -> Self { Ok }
}

impl Default for Ok {
    fn default() -> Self { Ok }
}

struct Params;

impl Params {
    fn new(_: u32) -> Self { Params }
}

fn main() {}
