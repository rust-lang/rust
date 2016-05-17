#![feature(plugin)]
#![plugin(clippy)]

#![allow(dead_code)]
#![deny(new_without_default, new_without_default_derive)]

struct Foo;

impl Foo {
    fn new() -> Foo { Foo } //~ERROR: you should consider deriving a `Default` implementation for `Foo`
}

struct Bar;

impl Bar {
    fn new() -> Self { Bar } //~ERROR: you should consider deriving a `Default` implementation for `Bar`
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

struct GenericsOk<T> {
    bar: T,
}

impl<U> Default for GenericsOk<U> {
    fn default() -> Self { unimplemented!(); }
}

impl<'c, V> GenericsOk<V> {
    fn new() -> GenericsOk<V> { unimplemented!() }
}

struct LtOk<'a> {
    foo: &'a bool,
}

impl<'b> Default for LtOk<'b> {
    fn default() -> Self { unimplemented!(); }
}

impl<'c> LtOk<'c> {
    fn new() -> LtOk<'c> { unimplemented!() }
}

struct LtKo<'a> {
    foo: &'a bool,
}

impl<'c> LtKo<'c> {
    fn new() -> LtKo<'c> { unimplemented!() } //~ERROR: you should consider adding a `Default` implementation for
}

fn main() {}
